"""Perception Distortion Engine and Sensor Exactness Modeling for Balanced RL (BRL).

Models the reality gap by allowing the Tweaker and Balancer to regulate the correctness
of the agent's perception across:
1. Morphology / self-model knowledge (how accurately the agent knows its leg lengths,
   mass, and physical parameters in its context vector).
2. Proprioceptive joint encoders (calibration zero-offset bias, scale drift, noise).
3. Vestibular / IMU & torso kinematics (tilt bias, gyro drift, accelerometer noise).
4. Contact & force load cells (gain calibration error, dead-zone threshold, contact noise).
5. Sensor quantization & resolution discretization (finite ADC/encoder bit-depth).
6. Observation latency & sensor dropout (temporal transport lag and packet loss).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ModalitySliceSpec:
    """Specifies observation indices corresponding to distinct sensor modalities."""

    obs_dim: int
    imu_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    joint_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contact_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def qpos_slice(self) -> Optional[slice]:
        if len(self.joint_indices) > 0:
            half = max(1, len(self.joint_indices) // 2)
            return slice(int(self.joint_indices[0]), int(self.joint_indices[half - 1] + 1))
        return None

    @property
    def qvel_slice(self) -> Optional[slice]:
        if len(self.joint_indices) > 0:
            half = max(1, len(self.joint_indices) // 2)
            return slice(int(self.joint_indices[half]), int(self.joint_indices[-1] + 1))
        return None

    @property
    def imu_slice(self) -> Optional[slice]:
        if len(self.imu_indices) > 0:
            return slice(int(self.imu_indices[0]), int(self.imu_indices[-1] + 1))
        return None

    @property
    def cfrc_slice(self) -> Optional[slice]:
        if len(self.contact_indices) > 0:
            return slice(int(self.contact_indices[0]), int(self.contact_indices[-1] + 1))
        return None

    @classmethod
    def create_ant_spec(cls, use_contact_forces: bool = False) -> ModalitySliceSpec:
        """Constructs ModalitySliceSpec for Morphological Ant (27D or 105D).

        Canonical 27D layout:
          0..4: Torso position (z + 4 quat elements) [IMU]
          5..12: Leg joint angles (8 angles) [Joints]
          13..18: Torso velocities (3 linear + 3 angular) [IMU]
          19..26: Leg joint velocities (8 velocities) [Joints]
        Contact forces (if enabled, dims 27..104):
          27..32: Torso cfrc (6)
          33..104: Leg links cfrc (72)
        """
        imu_idx = np.array([0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18], dtype=np.int64)
        joint_idx = np.array(
            [5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26], dtype=np.int64
        )
        if use_contact_forces:
            contact_idx = np.arange(27, 105, dtype=np.int64)
            obs_dim = 105
        else:
            contact_idx = np.zeros(0, dtype=np.int64)
            obs_dim = 27
        return cls(
            obs_dim=obs_dim,
            imu_indices=imu_idx,
            joint_indices=joint_idx,
            contact_indices=contact_idx,
        )

    @classmethod
    def create_dog_spec(cls) -> ModalitySliceSpec:
        """Constructs ModalitySliceSpec for Unitree Go1 Quadruped Dog (48D).

        Layout:
          0..11: Torso z, roll/pitch, projected gravity, linear vel, angular vel [IMU]
          12..23: 12 joint positions [Joints]
          24..35: 12 joint velocities [Joints]
          36..47: 4 feet 3D contact forces [Contact]
        """
        imu_idx = np.arange(0, 12, dtype=np.int64)
        joint_idx = np.arange(12, 36, dtype=np.int64)
        contact_idx = np.arange(36, 48, dtype=np.int64)
        return cls(
            obs_dim=48,
            imu_indices=imu_idx,
            joint_indices=joint_idx,
            contact_indices=contact_idx,
        )

    @classmethod
    def create_humanoid_spec(cls, obs_dim: int = 348) -> ModalitySliceSpec:
        """Constructs ModalitySliceSpec for Morphological Humanoid (270D or 348D).

        Gymnasium Humanoid-v5 layout:
          0..21: Torso height, orientations, velocities, and center of mass [IMU]
          22..180: Joint positions, velocities, and actuators [Joints]
          181..obs_dim-1: External contact forces cfrc_ext [Contact]
        """
        imu_idx = np.arange(0, min(22, obs_dim), dtype=np.int64)
        split_pt = min(180, obs_dim)
        joint_idx = np.arange(22, split_pt, dtype=np.int64) if obs_dim > 22 else np.zeros(0, dtype=np.int64)
        contact_idx = np.arange(split_pt, obs_dim, dtype=np.int64) if obs_dim > split_pt else np.zeros(0, dtype=np.int64)
        return cls(
            obs_dim=obs_dim,
            imu_indices=imu_idx,
            joint_indices=joint_idx,
            contact_indices=contact_idx,
        )


class PerceptionDistortionFilter:
    """Applies controlled perception inexactness to observations and agent context."""

    def __init__(
        self,
        spec: Optional[ModalitySliceSpec] = None,
        slice_spec: Optional[ModalitySliceSpec] = None,
        obs_dim: Optional[int] = None,
        max_latency_steps: int = 4,
    ) -> None:
        self.spec = slice_spec if slice_spec is not None else spec
        self.obs_dim = obs_dim or (self.spec.obs_dim if self.spec else 0)
        self.max_latency_steps = max_latency_steps
        self._rng = np.random.default_rng()

        # Calibration parameters set per episode
        self.current_tweak = np.zeros(18, dtype=np.float64)
        self.perceived_tweak_vector = np.zeros(18, dtype=np.float32)

        self.true_leg_scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.perceived_leg_scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.leg_length_exactness: float = 1.0
        self.last_leg_length_exactness: float = 1.0

        # Episode-constant calibration biases and scale errors
        self._joint_bias = np.zeros(0, dtype=np.float32)
        self._joint_scale = 1.0
        self._imu_bias = np.zeros(0, dtype=np.float32)
        self._imu_scale = 1.0
        self._contact_scale = 1.0
        self._contact_deadzone = 0.0

        # Sensor noise and discretization
        self._step_joint_noise_std = 0.0
        self._step_imu_noise_std = 0.0
        self._step_contact_noise_std = 0.0
        self._quant_step = 0.0

        # Observation latency and packet dropout
        self._dropout_prob = 0.0
        self._latency_steps = 0
        self._obs_history: deque[np.ndarray] = deque(maxlen=max_latency_steps)
        self._last_emitted_obs: Optional[np.ndarray] = None

        # Limb position & pointing perception calibration
        self._limb_pos_bias: Optional[np.ndarray] = None
        self._limb_rot_bias: Optional[np.ndarray] = None
        self._limb_pos_bias_std: float = 0.0
        self._limb_rot_bias_std: float = 0.0
        self._step_limb_pos_noise_std: float = 0.0
        self._step_limb_rot_noise_std: float = 0.0
        self.limb_pos_rmse: float = 0.0
        self.limb_pointing_rmse: float = 0.0
        self.limb_kinematics_exactness: float = 1.0
        self.last_perceived_kinematics: Optional[np.ndarray] = None

        # Running telemetry
        self._step_sq_errors: List[float] = []
        self._ep_steps: int = 0

    @property
    def perceived_context_vector(self) -> np.ndarray:
        return self.perceived_tweak_vector

    def reset_episode(
        self,
        tweak_vector: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        true_leg_scales: Optional[Sequence[float]] = None,
        obs_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Initializes episode-level perception distortion parameters.

        Args:
            tweak_vector: Raw or projected tweak vector of length >= 12 (up to 18).
            rng: Random number generator for deterministic noise sampling.
            true_leg_scales: Optional ground truth leg length scales (s_front, s_back, s_left, s_right).
            obs_dim: Total observation dimension.
            seed: Optional random seed.

        Returns:
            perceived_tweak_vector: 18D vector representing what the agent perceives.
            info: Perception metadata dictionary.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif rng is not None:
            self._rng = rng

        t_in = np.asarray(tweak_vector, dtype=np.float64).flatten()
        if t_in.shape[0] < 18:
            t_in = np.pad(t_in, (0, 18 - t_in.shape[0]))
        self.current_tweak = np.clip(t_in[:18], 0.0, 1000.0)

        # Normalize perception controls u_12..u_17 in [0, 1]
        u_morph = float(self.current_tweak[12]) / 1000.0    # w13: Morphology / self-knowledge exactness
        u_joint = float(self.current_tweak[13]) / 1000.0    # w14: Joint encoder exactness
        u_imu = float(self.current_tweak[14]) / 1000.0      # w15: IMU / Torso kinematics exactness
        u_contact = float(self.current_tweak[15]) / 1000.0  # w16: Contact force exactness
        u_quant = float(self.current_tweak[16]) / 1000.0    # w17: Sensor quantization
        u_lat = float(self.current_tweak[17]) / 1000.0      # w18: Observation latency & dropout

        # ----------------------------------------------------------------------
        # 1. Morphology & Leg-Length Knowledge Exactness (w[12])
        # ----------------------------------------------------------------------
        if true_leg_scales is not None and len(true_leg_scales) > 0:
            self.true_leg_scales = np.array(true_leg_scales, dtype=np.float32)
        else:
            self.true_leg_scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        if u_morph <= 1e-6:
            # 100% Exact perception
            self.perceived_leg_scales = self.true_leg_scales.copy()
            self.leg_length_exactness = 1.0
            self.last_leg_length_exactness = 1.0
            perceived_phys = self.current_tweak[:12].copy()
        else:
            # Perturb agent's perceived leg length scales
            leg_arr = np.array(self.true_leg_scales, dtype=np.float64)
            # Calibration scale bias (systematic over/underestimation) + leg-specific jitter
            bias_scale = self._rng.uniform(-0.30, 0.30) * u_morph
            leg_noise = self._rng.normal(0.0, 0.15 * u_morph, size=leg_arr.shape)
            perc_legs = np.clip(leg_arr * (1.0 + bias_scale + leg_noise), 0.20, 3.50)
            self.perceived_leg_scales = np.array(perc_legs, dtype=np.float32)

            # Relative error in leg length estimation
            scale_norm = float(np.linalg.norm(leg_arr)) + 1e-6
            err_norm = float(np.linalg.norm(perc_legs - leg_arr))
            rel_err = err_norm / scale_norm
            self.leg_length_exactness = float(np.clip(1.0 - rel_err, 0.0, 1.0))
            self.last_leg_length_exactness = self.leg_length_exactness

            # Perturb physical context vector w_0..w_11 perceived by the agent
            phys_tweak = self.current_tweak[:12].copy()
            # Multiplicative distortion and additive noise proportional to u_morph
            phys_distort = 1.0 + self._rng.normal(0.0, 0.25 * u_morph, size=12)
            phys_bias = self._rng.normal(0.0, 75.0 * u_morph, size=12)
            perceived_phys = np.clip(phys_tweak * phys_distort + phys_bias, 0.0, 1000.0)

        # Form full 18D perceived tweak vector
        perc_slots = self.current_tweak[12:18].copy()
        self.perceived_tweak_vector = np.concatenate([perceived_phys, perc_slots]).astype(np.float32)

        # ----------------------------------------------------------------------
        # 2. Sensor Modality Calibration & Distortion Parameters (w[13]..w[17])
        # ----------------------------------------------------------------------
        target_dim = obs_dim or (self.spec.obs_dim if self.spec is not None else 27)

        # Joint encoders (w[13])
        n_joints = len(self.spec.joint_indices) if (self.spec and len(self.spec.joint_indices) > 0) else max(1, target_dim // 2)
        if u_joint > 1e-6:
            self._joint_bias = self._rng.normal(0.0, 0.15 * u_joint, size=n_joints).astype(np.float32)
            self._joint_scale = float(1.0 + self._rng.uniform(-0.20, 0.20) * u_joint)
            self._step_joint_noise_std = float(0.05 * u_joint)
        else:
            self._joint_bias = np.zeros(n_joints, dtype=np.float32)
            self._joint_scale = 1.0
            self._step_joint_noise_std = 0.0

        # IMU & Torso Kinematics (w[14])
        n_imu = len(self.spec.imu_indices) if (self.spec and len(self.spec.imu_indices) > 0) else max(1, target_dim // 4)
        if u_imu > 1e-6:
            self._imu_bias = self._rng.normal(0.0, 0.10 * u_imu, size=n_imu).astype(np.float32)
            self._imu_scale = float(1.0 + self._rng.uniform(-0.15, 0.15) * u_imu)
            self._step_imu_noise_std = float(0.04 * u_imu)
        else:
            self._imu_bias = np.zeros(n_imu, dtype=np.float32)
            self._imu_scale = 1.0
            self._step_imu_noise_std = 0.0

        # Contact & Force Load Cells (w[15])
        if u_contact > 1e-6:
            self._contact_scale = float(1.0 + self._rng.uniform(-0.35, 0.35) * u_contact)
            self._contact_deadzone = float(0.08 * u_contact)
            self._step_contact_noise_std = float(0.08 * u_contact)
        else:
            self._contact_scale = 1.0
            self._contact_deadzone = 0.0
            self._step_contact_noise_std = 0.0

        # Quantization grid step (w[16])
        if u_quant > 1e-6:
            self._quant_step = float(0.15 * (u_quant**2))
        else:
            self._quant_step = 0.0

        # Latency & Packet Dropout (w[17])
        if u_lat > 1e-6:
            self._dropout_prob = float(0.35 * u_lat)
            self._latency_steps = 1 if u_lat < 0.60 else 2
        else:
            self._dropout_prob = 0.0
            self._latency_steps = 0

        # ----------------------------------------------------------------------
        # 3. Limb Kinematics Perception (Position: w[12] / Pointing: w[13])
        # ----------------------------------------------------------------------
        if u_morph > 1e-6:
            self._step_limb_pos_noise_std = float(0.015 * u_morph)
            self._limb_pos_bias_std = float(0.05 * u_morph)
        else:
            self._step_limb_pos_noise_std = 0.0
            self._limb_pos_bias_std = 0.0

        if u_joint > 1e-6:
            self._step_limb_rot_noise_std = float(0.02 * u_joint)
            self._limb_rot_bias_std = float(0.08 * u_joint)
        else:
            self._step_limb_rot_noise_std = 0.0
            self._limb_rot_bias_std = 0.0

        self._limb_pos_bias = None
        self._limb_rot_bias = None
        self.limb_pos_rmse = 0.0
        self.limb_pointing_rmse = 0.0
        self.limb_kinematics_exactness = 1.0
        self.last_perceived_kinematics = None

        self._obs_history.clear()
        self._last_emitted_obs = None
        self._step_sq_errors.clear()
        self._ep_steps = 0

        info = self.get_telemetry()
        info["perceived_tweak_vector"] = self.perceived_tweak_vector.copy()
        return self.perceived_tweak_vector, info

    def distort_observation(
        self, obs: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies configured perception distortions to the raw environment observation.

        Args:
            obs: Clean canonical observation vector o_t.
            rng: Optional RNG.

        Returns:
            distorted_obs: Perceived observation vector o_tilde_t seen by the agent.
            metrics: Current perception telemetry dictionary.
        """
        if rng is not None:
            self._rng = rng

        clean_obs = np.asarray(obs, dtype=np.float32)
        distorted = clean_obs.copy()
        d_len = distorted.shape[0]

        # 1. Modality-specific distortion (bias, scale, and sensor noise)
        if self.spec is not None and len(self.spec.joint_indices) > 0 and len(self.spec.imu_indices) > 0:
            # Joint slice
            j_valid = self.spec.joint_indices[self.spec.joint_indices < d_len]
            if len(j_valid) > 0:
                b_slice = self._joint_bias[: len(j_valid)]
                distorted[j_valid] = (distorted[j_valid] * self._joint_scale) + b_slice
                if self._step_joint_noise_std > 1e-6:
                    j_noise = self._rng.normal(0.0, self._step_joint_noise_std, size=len(j_valid)).astype(np.float32)
                    distorted[j_valid] += j_noise

            # IMU slice
            imu_valid = self.spec.imu_indices[self.spec.imu_indices < d_len]
            if len(imu_valid) > 0:
                b_imu = self._imu_bias[: len(imu_valid)]
                distorted[imu_valid] = (distorted[imu_valid] * self._imu_scale) + b_imu
                if self._step_imu_noise_std > 1e-6:
                    imu_noise = self._rng.normal(0.0, self._step_imu_noise_std, size=len(imu_valid)).astype(np.float32)
                    distorted[imu_valid] += imu_noise

            # Contact slice
            c_valid = self.spec.contact_indices[self.spec.contact_indices < d_len]
            if len(c_valid) > 0:
                distorted[c_valid] = distorted[c_valid] * self._contact_scale
                if self._contact_deadzone > 1e-6:
                    small_c = np.abs(distorted[c_valid]) < self._contact_deadzone
                    distorted[c_valid][small_c] = 0.0
                if self._step_contact_noise_std > 1e-6:
                    c_noise = self._rng.normal(0.0, self._step_contact_noise_std, size=len(c_valid)).astype(np.float32)
                    distorted[c_valid] += c_noise
        else:
            # Fallback when no modality slices are configured: apply overall joint & IMU distortions across halves
            half = d_len // 2
            if self._joint_scale != 1.0 or len(self._joint_bias) > 0:
                distorted[half:] *= self._joint_scale
            if self._step_joint_noise_std > 1e-6:
                distorted[half:] += self._rng.normal(0.0, self._step_joint_noise_std, size=d_len - half).astype(np.float32)

        # 2. Sensor Quantization & Discretization (w[16])
        if self._quant_step > 1e-6:
            distorted = np.round(distorted / self._quant_step) * self._quant_step

        # 3. Observation Latency & Sensor Dropout (w[17])
        self._obs_history.append(distorted.copy())
        if self._latency_steps > 0 and len(self._obs_history) > self._latency_steps:
            delayed_obs = self._obs_history[-(1 + self._latency_steps)].copy()
        else:
            delayed_obs = distorted.copy()

        if self._dropout_prob > 1e-6 and self._last_emitted_obs is not None:
            drop_mask = self._rng.uniform(0.0, 1.0, size=d_len) < self._dropout_prob
            delayed_obs[drop_mask] = self._last_emitted_obs[drop_mask]

        self._last_emitted_obs = delayed_obs.copy()
        output_obs = np.clip(delayed_obs, -100.0, 100.0).astype(np.float32)

        # Track distortion error metrics
        sq_err = float(np.mean(np.square(output_obs - clean_obs)))
        self._step_sq_errors.append(sq_err)
        self._ep_steps += 1
        return output_obs, self.get_telemetry()

    def distort_limb_kinematics(
        self, true_kinematics: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies configured perception noise to body-relative limb anchor position (x, y, z)
        and instantaneous pointing (yaw, pitch).

        Limb position inexactness is governed by w[12] (u_morph).
        Limb pointing inexactness is governed by w[13] (u_joint).

        Args:
            true_kinematics: Array of shape (n_limbs, 5) where columns are [x, y, z, yaw, pitch].
            rng: Optional RNG.

        Returns:
            perceived_kinematics: Distorted array of shape (n_limbs, 5).
            metrics: Telemetry dictionary including limb_pos_rmse, limb_pointing_rmse, limb_kinematics_exactness.
        """
        if rng is not None:
            self._rng = rng

        kin = np.asarray(true_kinematics, dtype=np.float32)
        if kin.ndim != 2 or kin.shape[1] < 5:
            return kin, {
                "limb_pos_rmse": 0.0,
                "limb_pointing_rmse": 0.0,
                "limb_kinematics_exactness": 1.0,
            }

        n_limbs = kin.shape[0]
        # Lazily allocate persistent calibration biases for current limb count if not set
        if self._limb_pos_bias is None or self._limb_pos_bias.shape[0] != n_limbs:
            if self._limb_pos_bias_std > 1e-6:
                self._limb_pos_bias = self._rng.normal(
                    0.0, self._limb_pos_bias_std, size=(n_limbs, 3)
                ).astype(np.float32)
            else:
                self._limb_pos_bias = np.zeros((n_limbs, 3), dtype=np.float32)

        if self._limb_rot_bias is None or self._limb_rot_bias.shape[0] != n_limbs:
            if self._limb_rot_bias_std > 1e-6:
                self._limb_rot_bias = self._rng.normal(
                    0.0, self._limb_rot_bias_std, size=(n_limbs, 2)
                ).astype(np.float32)
            else:
                self._limb_rot_bias = np.zeros((n_limbs, 2), dtype=np.float32)

        distorted = kin.copy()

        # 1. Distort Limb Anchor Positions (x, y, z) via w[12]
        if self._limb_pos_bias_std > 1e-6 or self._step_limb_pos_noise_std > 1e-6:
            pos_noise = (
                self._rng.normal(0.0, self._step_limb_pos_noise_std, size=(n_limbs, 3)).astype(np.float32)
                if self._step_limb_pos_noise_std > 1e-6
                else np.zeros((n_limbs, 3), dtype=np.float32)
            )
            distorted[:, 0:3] = kin[:, 0:3] + self._limb_pos_bias + pos_noise

        # 2. Distort Limb Pointing Direction (yaw, pitch) via w[13]
        if self._limb_rot_bias_std > 1e-6 or self._step_limb_rot_noise_std > 1e-6:
            rot_noise = (
                self._rng.normal(0.0, self._step_limb_rot_noise_std, size=(n_limbs, 2)).astype(np.float32)
                if self._step_limb_rot_noise_std > 1e-6
                else np.zeros((n_limbs, 2), dtype=np.float32)
            )
            raw_yaw = kin[:, 3] + self._limb_rot_bias[:, 0] + rot_noise[:, 0]
            raw_pitch = kin[:, 4] + self._limb_rot_bias[:, 1] + rot_noise[:, 1]
            distorted[:, 3] = raw_yaw % 1.0
            distorted[:, 4] = np.clip(raw_pitch, 0.0, 1.0)

        # Telemetry
        pos_sq_err = float(np.mean(np.square(distorted[:, 0:3] - kin[:, 0:3])))
        rot_sq_err = float(np.mean(np.square(distorted[:, 3:5] - kin[:, 3:5])))
        self.limb_pos_rmse = float(math.sqrt(pos_sq_err))
        self.limb_pointing_rmse = float(math.sqrt(rot_sq_err))
        self.limb_kinematics_exactness = float(
            1.0 / (1.0 + 10.0 * self.limb_pos_rmse + 5.0 * self.limb_pointing_rmse)
        )
        self.last_perceived_kinematics = distorted.copy()

        metrics = {
            "limb_pos_rmse": self.limb_pos_rmse,
            "limb_pointing_rmse": self.limb_pointing_rmse,
            "limb_kinematics_exactness": self.limb_kinematics_exactness,
        }
        return distorted.astype(np.float32), metrics

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns instantaneous and episode-level perception telemetry."""
        if len(self._step_sq_errors) > 0:
            obs_rmse = float(math.sqrt(np.mean(self._step_sq_errors)))
        else:
            obs_rmse = 0.0

        obs_exactness = float(1.0 / (1.0 + obs_rmse))
        leg_exactness = float(self.leg_length_exactness)
        limb_exactness = float(self.limb_kinematics_exactness)
        composite_correctness = float(
            0.30 * leg_exactness + 0.50 * obs_exactness + 0.20 * limb_exactness
        )

        mean_phys = float(np.mean(self.current_tweak[:12]))
        mean_perc = float(np.mean(self.current_tweak[12:18]))

        return {
            "perception_correctness": composite_correctness,
            "leg_length_exactness": leg_exactness,
            "limb_kinematics_exactness": limb_exactness,
            "limb_pos_rmse": self.limb_pos_rmse,
            "limb_pointing_rmse": self.limb_pointing_rmse,
            "obs_exactness": obs_exactness,
            "obs_rmse": obs_rmse,
            "true_leg_scales": [float(s) for s in self.true_leg_scales],
            "perceived_leg_scales": [float(s) for s in self.perceived_leg_scales],
            "mean_physics_tweak": mean_phys,
            "mean_perception_tweak": mean_perc,
            "morph_inexactness": float(self.current_tweak[12]),
            "joint_inexactness": float(self.current_tweak[13]),
            "imu_inexactness": float(self.current_tweak[14]),
            "contact_inexactness": float(self.current_tweak[15]),
            "quant_inexactness": float(self.current_tweak[16]),
            "latency_inexactness": float(self.current_tweak[17]),
        }
