"""Morphological Quadruped Dog Environment (Unitree Go1) applying 12D BRL Tweak Vector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from brl_core.interfaces import WinEvaluator
from brl_core.perception import (
    ModalitySliceSpec,
    PerceptionDistortionFilter,
    compute_torso_leg_angle_diff,
)


def _quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Converts a quaternion [w, x, y, z] to Euler roll, pitch, yaw (radians)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _rotate_vec_by_quat_conj(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotates 3D vector v by the conjugate of quaternion q=[w, x, y, z] (world to body frame)."""
    w, x, y, z = q[0], -q[1], -q[2], -q[3]  # conjugate
    qv = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


class MorphologicalDogBRLWrapper(gym.Env):
    """Self-contained Gymnasium Environment for 12-DoF Unitree Go1 Quadruped with 12D BRL Tweaks.

    Action space: Box(-1.0, 1.0, shape=(12,), float32)
      - FR (Hip, Thigh, Calf), FL (Hip, Thigh, Calf), RR (Hip, Thigh, Calf), RL (Hip, Thigh, Calf)
    Observation space: Box(-inf, inf, shape=(48,), float32)
      - Base z (1)
      - Base roll, pitch (2)
      - Projected gravity vector gx, gy, gz in body frame (3)
      - Base linear velocity vx, vy, vz (3)
      - Base angular velocity wx, wy, wz (3)
      - 12 joint positions (12)
      - 12 joint velocities (12)
      - 4 feet 3D contact forces (12)
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        max_episode_steps: int = 1000,
        frame_skip: int = 10,
        healthy_z_range: Tuple[float, float] = (0.18, 0.60),
        max_action_capacity: int = 12,
        render_mode: Optional[str] = None,
        camera_name: Optional[str] = "track",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.camera_name = camera_name
        self._renderer: Optional[mujoco.Renderer] = None
        if xml_path is None:
            xml_path = str(Path(__file__).parent / "assets" / "unitree_go1.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.healthy_z_range = healthy_z_range
        self.max_action_capacity = int(max_action_capacity)
        self.act_dim = 12
        self.canonical_obs_dim = 48

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.canonical_obs_dim,), dtype=np.float32
        )

        # Baseline body IDs
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.front_legs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["fr_hip", "fr_thigh", "fr_calf", "fl_hip", "fl_thigh", "fl_calf"]
        ]
        self.rear_legs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["rr_hip", "rr_thigh", "rr_calf", "rl_hip", "rl_thigh", "rl_calf"]
        ]
        self.left_legs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["fl_hip", "fl_thigh", "fl_calf", "rl_hip", "rl_thigh", "rl_calf"]
        ]
        self.right_legs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["fr_hip", "fr_thigh", "fr_calf", "rr_hip", "rr_thigh", "rr_calf"]
        ]

        # Foot geoms for contact forces
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["fr_foot", "fl_foot", "rr_foot", "rl_foot"]
        ]

        # Cache baseline physical parameters
        self._default_body_masses = np.array(self.model.body_mass, copy=True)
        self._default_dof_damping = np.array(self.model.dof_damping, copy=True)
        self._default_geom_friction = np.array(self.model.geom_friction, copy=True)
        self._default_actuator_gear = np.array(self.model.actuator_gear, copy=True)
        self._default_gravity = np.array(self.model.opt.gravity, copy=True)

        # Standing pose
        # root pos: [0, 0, 0.38], root quat: [1, 0, 0, 0]
        # 12 joints: hip=0.0, thigh=0.8, calf=-1.6 for each of the 4 legs
        self._standing_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self._standing_qpos[0:3] = [0.0, 0.0, 0.38]
        self._standing_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for leg_idx in range(4):
            idx = 7 + leg_idx * 3
            self._standing_qpos[idx] = 0.0      # hip
            self._standing_qpos[idx + 1] = 0.8  # thigh
            self._standing_qpos[idx + 2] = -1.6 # calf

        self.perception_filter = PerceptionDistortionFilter(
            slice_spec=ModalitySliceSpec.create_dog_spec(),
            obs_dim=self.canonical_obs_dim,
            max_latency_steps=4,
        )
        self._rng = np.random.default_rng()
        self._current_tweak = np.zeros(18, dtype=np.float32)
        self._action_mask = np.ones(self.act_dim, dtype=np.float32)
        self._wind_state = np.zeros(3, dtype=np.float64)
        self._sensor_noise_std = 0.0
        self._prev_x = 0.0
        self._step_count = 0
        self._ep_energy = 0.0
        self._healthy_steps = 0

    def apply_tweak_vector(self, tweak: np.ndarray, disabled_leg: Optional[int] = None) -> None:
        """Applies the 18D Knapsack physical + perception tweak vector w_e to the MuJoCo simulation."""
        tweak_arr = np.asarray(tweak, dtype=np.float32).flatten()
        if tweak_arr.shape[0] < 18:
            tweak_arr = np.pad(tweak_arr, (0, 18 - tweak_arr.shape[0]))
        self._current_tweak = np.clip(tweak_arr[:18], 0.0, 1000.0)
        norm_tweak = self._current_tweak[:12] / 1000.0

        # Reset to defaults
        self.model.body_mass[:] = self._default_body_masses.copy()
        self.model.dof_damping[:] = self._default_dof_damping.copy()
        self.model.geom_friction[:] = self._default_geom_friction.copy()
        self.model.actuator_gear[:] = self._default_actuator_gear.copy()
        self.model.opt.gravity[:] = self._default_gravity.copy()

        # 0: Torso payload / mass (+0% to +80%)
        torso_factor = 1.0 + 0.80 * float(norm_tweak[0])
        self.model.body_mass[self.torso_body_id] *= torso_factor

        # 1: Leg mass scale (0.7x to 1.5x)
        leg_mass_factor = 1.0 + 0.50 * float(norm_tweak[1])
        all_leg_bodies = self.front_legs_body_ids + self.rear_legs_body_ids
        for b_id in all_leg_bodies:
            self.model.body_mass[b_id] *= leg_mass_factor

        # 2: Front vs Rear asymmetry (tilt mass forward or backward: +/- 30%)
        asym_fr = float(norm_tweak[2]) * 0.30
        for b_id in self.front_legs_body_ids:
            self.model.body_mass[b_id] *= (1.0 + asym_fr)
        for b_id in self.rear_legs_body_ids:
            self.model.body_mass[b_id] *= (1.0 - asym_fr)

        # 3: Lateral asymmetry (Left vs Right: +/- 30%)
        asym_lat = float(norm_tweak[3]) * 0.30
        for b_id in self.left_legs_body_ids:
            self.model.body_mass[b_id] *= (1.0 + asym_lat)
        for b_id in self.right_legs_body_ids:
            self.model.body_mass[b_id] *= (1.0 - asym_lat)

        # 4: Joint damping multiplier (1.0x to 3.0x)
        damping_mult = 1.0 + 2.0 * float(norm_tweak[4])
        self.model.dof_damping[6:] *= damping_mult

        # 5: Foot friction (1.0 down to 0.25 on ice)
        friction_factor = max(0.15, 1.0 - 0.75 * float(norm_tweak[5]))
        self.model.geom_friction[:, 0] *= friction_factor

        # 6: Gravity scale (0.8x to 1.3x)
        grav_scale = 1.0 + 0.30 * (float(norm_tweak[6]) - 0.5)
        self.model.opt.gravity[2] = -9.81 * grav_scale

        # 7: Actuator gear / torque authority (1.0 down to 0.5)
        torque_factor = max(0.4, 1.0 - 0.50 * float(norm_tweak[7]))
        self.model.actuator_gear[:, 0] *= torque_factor

        # 8: Lateral wind force magnitude (0 to 15 N)
        self._wind_lateral_mag = float(norm_tweak[8]) * 15.0

        # 9: Headwind force magnitude (0 to 12 N)
        self._wind_head_mag = float(norm_tweak[9]) * 12.0

        # 10: Sensor noise std (0.0 to 0.05)
        self._sensor_noise_std = float(norm_tweak[10]) * 0.05

        # 11: Tripod / disabled leg weakness: if disabled_leg specified or tweak[11] > 750
        self._action_mask = np.ones(self.act_dim, dtype=np.float32)
        if disabled_leg is not None and 0 <= disabled_leg < 4:
            leg_start = disabled_leg * 3
            self._action_mask[leg_start : leg_start + 3] = 0.0
        elif norm_tweak[11] > 0.75:
            # Disable Front-Right leg (tripod stress test)
            self._action_mask[0:3] = 0.0

        self._wind_state = np.zeros(3, dtype=np.float64)

    def _get_foot_forces(self) -> np.ndarray:
        """Extracts 3D contact forces for the 4 feet [FR, FL, RR, RL] (12 dims)."""
        forces = np.zeros(12, dtype=np.float32)
        # MuJoCo contacts
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1, geom2 = con.geom1, con.geom2
            for leg_idx, foot_gid in enumerate(self.foot_geom_ids):
                if geom1 == foot_gid or geom2 == foot_gid:
                    c_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    # Normal contact force along contact frame
                    forces[leg_idx * 3] += float(c_force[0])
                    forces[leg_idx * 3 + 1] += float(c_force[1])
                    forces[leg_idx * 3 + 2] += float(c_force[2])
        # Scale to manageable magnitude
        return np.clip(forces / 100.0, -5.0, 5.0)

    def _get_obs(self) -> np.ndarray:
        """Constructs canonical aligned 48-dimensional observation vector."""
        # 1: Torso z
        torso_z = float(self.data.qpos[2])

        # 2: Torso roll, pitch
        quat = self.data.qpos[3:7]
        roll, pitch, _ = _quat_to_euler(quat)

        # 3: Projected gravity in body frame (gx, gy, gz)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        proj_grav = _rotate_vec_by_quat_conj(world_gravity, quat)

        # 4: Torso linear velocity (3)
        linvel = self.data.qvel[0:3]

        # 5: Torso angular velocity (3)
        angvel = self.data.qvel[3:6]

        # 6: 12 joint positions (12)
        joint_pos = self.data.qpos[7:19]

        # 7: 12 joint velocities (12)
        joint_vel = self.data.qvel[6:18]

        # 8: 4 feet 3D contact forces (12)
        foot_forces = self._get_foot_forces()

        obs = np.concatenate([
            [torso_z],
            [roll, pitch],
            proj_grav,
            linvel,
            angvel,
            joint_pos,
            joint_vel,
            foot_forces,
        ]).astype(np.float32)

        return obs

    def _add_sensor_noise(self, obs: np.ndarray) -> np.ndarray:
        """Applies calibrated proprioceptive Gaussian noise w_11 (norm_tweak[10]) to observation vector."""
        w11_n = float(self._current_tweak[10]) / 1000.0
        if w11_n <= 1e-6:
            return obs.astype(np.float32, copy=False)
        sigma_obs = 0.25 * w11_n
        obs_scale = np.maximum(np.abs(obs), 1.0)
        noise = self._rng.normal(0.0, sigma_obs * obs_scale, size=obs.shape)
        noisy_obs = obs + noise
        return np.clip(noisy_obs, -100.0, 100.0).astype(np.float32)

    def get_limb_kinematics(self) -> np.ndarray:
        """Computes body-relative anchor (x,y,z) and instantaneous pointing (yaw, pitch) for 4 legs."""
        torso_xpos = self.data.xpos[self.torso_body_id]
        torso_xmat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        legs = ["fr", "fl", "rr", "rl"]
        k_list = []
        for l in legs:
            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{l}_hip")
            f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{l}_calf")
            if a_id >= 0 and f_id >= 0:
                a_pos = torso_xmat.T @ (self.data.xpos[a_id] - torso_xpos)
                d_pos = torso_xmat.T @ (self.data.xpos[f_id] - torso_xpos)
                v = d_pos - a_pos
                u = v / max(float(np.linalg.norm(v)), 1e-6)
                yaw = float((np.arctan2(u[1], u[0]) + np.pi) / (2.0 * np.pi)) % 1.0
                pitch = float((np.arcsin(np.clip(u[2], -1.0, 1.0)) + np.pi / 2.0) / np.pi)
                diff_rx, diff_ry, diff_rz = compute_torso_leg_angle_diff(
                    torso_xmat, self.data.xmat[f_id]
                )
                k_list.append(
                    [float(a_pos[0]), float(a_pos[1]), float(a_pos[2]), yaw, pitch, diff_rx, diff_ry, diff_rz]
                )
            else:
                k_list.append([0.188, -0.08, 0.0, 0.064, 0.286, 0.0, 0.0, 0.0])
        return np.array(k_list, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Parse options
        tweak = np.zeros(12, dtype=np.float32)
        disabled_leg = None
        if options is not None:
            if "tweak_vector" in options:
                tweak = np.asarray(options["tweak_vector"], dtype=np.float32)
            if "disabled_leg" in options:
                disabled_leg = options["disabled_leg"]

        self.apply_tweak_vector(tweak, disabled_leg=disabled_leg)

        # Reset MuJoCo state to standing posture + small perturbation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._standing_qpos.copy()
        # Small noise for initial exploration
        self.data.qpos[7:19] += self._rng.uniform(-0.05, 0.05, size=12)
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._prev_x = float(self.data.qpos[0])
        self._step_count = 0
        self._ep_energy = 0.0
        self._healthy_steps = 0

        # Compute true leg scales from tweak asymmetry
        norm_tweak = self._current_tweak[:12] / 1000.0
        asym_fr = float(norm_tweak[2]) * 0.30
        asym_lr = float(norm_tweak[3]) * 0.30
        true_scales = np.array([
            (1.0 + asym_fr) * (1.0 - asym_lr),
            (1.0 + asym_fr) * (1.0 + asym_lr),
            (1.0 - asym_fr) * (1.0 - asym_lr),
            (1.0 - asym_fr) * (1.0 + asym_lr),
        ], dtype=np.float32)
        self.perception_filter.reset_episode(
            tweak_vector=self._current_tweak,
            true_leg_scales=true_scales,
            seed=seed,
        )

        raw_obs = self._get_obs()
        obs_noisy = self._add_sensor_noise(raw_obs)
        obs_distorted, perc_metrics = self.perception_filter.distort_observation(obs_noisy)
        true_kin = self.get_limb_kinematics()
        perc_kin, kin_metrics = self.perception_filter.distort_limb_kinematics(true_kin)
        perc_metrics.update(kin_metrics)

        info = {
            "morphology_mask": self._action_mask.copy(),
            "tweak_vector": self._current_tweak.copy(),
            "perceived_tweak_vector": self.perception_filter.perceived_context_vector.astype(np.float32),
            "perceived_leg_scales": self.perception_filter.perceived_leg_scales.astype(np.float32),
            "perception_metrics": perc_metrics,
            "limb_kinematics": perc_kin,
            "true_limb_kinematics": true_kin,
            "perceived_limb_kinematics": perc_kin,
            "torso_mass": float(self.model.body_mass[self.torso_body_id]),
            "healthy": True,
            "brl_metrics": {
                "x_velocity": 0.0,
                "total_distance": 0.0,
                "energy_cost": 0.0,
                "healthy_steps": 0,
                "tweak_vector": self._current_tweak.copy(),
                "perceived_tweak_vector": self.perception_filter.perceived_context_vector.astype(np.float32),
                "perceived_leg_scales": self.perception_filter.perceived_leg_scales.astype(np.float32),
                "perception_metrics": perc_metrics,
            },
        }
        return obs_distorted, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw_action.size < self.act_dim:
            padded = np.zeros(self.act_dim, dtype=np.float32)
            padded[: raw_action.size] = raw_action
            raw_action = padded
        else:
            raw_action = raw_action[: self.act_dim]

        # Apply morphology action mask (e.g. disabled leg in Tripod)
        effective_action = np.clip(raw_action, -1.0, 1.0) * self._action_mask

        # Apply lateral and headwind forces to torso body via Ornstein-Uhlenbeck process
        if self._wind_lateral_mag > 1e-3 or self._wind_head_mag > 1e-3:
            # OU update: dx = -theta * x * dt + sigma * dW
            wind_noise = self._rng.normal(0.0, 1.0, size=3)
            self._wind_state[0] += -0.1 * self._wind_state[0] + 0.3 * wind_noise[0]  # headwind
            self._wind_state[1] += -0.1 * self._wind_state[1] + 0.3 * wind_noise[1]  # lateral
            wind_x = -self._wind_head_mag * (1.0 + 0.2 * self._wind_state[0])
            wind_y = self._wind_lateral_mag * self._wind_state[1]
            self.data.xfrc_applied[self.torso_body_id, 0] = wind_x
            self.data.xfrc_applied[self.torso_body_id, 1] = wind_y
        else:
            self.data.xfrc_applied[self.torso_body_id, :] = 0.0

        # Sub-step simulation
        energy = float(np.sum(np.square(effective_action)))
        self._ep_energy += energy

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = effective_action * 25.0  # ctrlrange is [-25, 25] Nm
            mujoco.mj_step(self.model, self.data)

        # Post-step kinematics
        curr_x = float(self.data.qpos[0])
        dt = self.frame_skip * self.model.opt.timestep
        vx = (curr_x - self._prev_x) / dt
        self._prev_x = curr_x

        torso_z = float(self.data.qpos[2])
        roll, pitch, _ = _quat_to_euler(self.data.qpos[3:7])

        # Health checks: torso height and safe tilt
        is_healthy = (
            (self.healthy_z_range[0] <= torso_z <= self.healthy_z_range[1])
            and (abs(roll) < 1.0)
            and (abs(pitch) < 1.0)
        )
        if is_healthy:
            self._healthy_steps += 1

        # Reward components
        forward_reward = float(vx) * 1.5
        healthy_reward = 1.0 if is_healthy else 0.0
        ctrl_cost = -0.005 * energy
        lateral_cost = -0.1 * abs(float(self.data.qvel[1]))
        orientation_cost = -0.5 * (roll * roll + pitch * pitch)

        reward = forward_reward + healthy_reward + ctrl_cost + lateral_cost + orientation_cost

        terminated = not is_healthy
        truncated = self._step_count >= self.max_episode_steps

        raw_obs = self._get_obs()
        obs_noisy = self._add_sensor_noise(raw_obs)
        obs_distorted, perc_metrics = self.perception_filter.distort_observation(obs_noisy)
        true_kin = self.get_limb_kinematics()
        perc_kin, kin_metrics = self.perception_filter.distort_limb_kinematics(true_kin)
        perc_metrics.update(kin_metrics)

        info = {
            "morphology_mask": self._action_mask.copy(),
            "tweak_vector": self._current_tweak.copy(),
            "perceived_tweak_vector": self.perception_filter.perceived_context_vector.astype(np.float32),
            "perceived_leg_scales": self.perception_filter.perceived_leg_scales.astype(np.float32),
            "perception_metrics": perc_metrics,
            "limb_kinematics": perc_kin,
            "true_limb_kinematics": true_kin,
            "perceived_limb_kinematics": perc_kin,
            "healthy": is_healthy,
            "brl_metrics": {
                "forward_reward": float(forward_reward),
                "healthy_reward": float(healthy_reward),
                "ctrl_cost": float(ctrl_cost),
                "x_velocity": float(vx),
                "total_distance": float(curr_x),
                "energy_cost": float(self._ep_energy),
                "healthy_steps": int(self._healthy_steps),
                "tweak_vector": self._current_tweak.copy(),
                "perceived_tweak_vector": self.perception_filter.perceived_context_vector.astype(np.float32),
                "perceived_leg_scales": self.perception_filter.perceived_leg_scales.astype(np.float32),
                "perception_metrics": perc_metrics,
            },
        }
        return obs_distorted, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders current frame using offscreen MuJoCo renderer."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=480)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self) -> None:
        """Cleans up offscreen renderer resources."""
        if getattr(self, "_renderer", None) is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        super().close()


def evaluate_dog_benchmark_suite(
    agent: Any,
    env: MorphologicalDogBRLWrapper,
    max_steps: int = 500,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluates the agent deterministically on Standard Dog and OOD Dog stress variations (including Tripod)."""
    suite_specs: Dict[str, Dict[str, Any]] = {
        "Standard-Go1": {
            "tweak": np.zeros(12, dtype=np.float32),
            "disabled_leg": None,
        },
        "Tripod-Dog": {
            "tweak": np.zeros(12, dtype=np.float32),
            "disabled_leg": 0,  # Front-Right leg disabled: test 3-legged recovery & brittleness!
        },
        "Heavy-Payload-Dog": {
            "tweak": np.array([800, 0, 0, 0, 0, 0, 300, 0, 0, 0, 0, 0], dtype=np.float32),
            "disabled_leg": None,
        },
        "Ice-Surface-Dog": {
            "tweak": np.array([0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "disabled_leg": None,
        },
        "Windy-Gale-Dog": {
            "tweak": np.array([0, 0, 0, 0, 0, 0, 0, 0, 700, 600, 0, 0], dtype=np.float32),
            "disabled_leg": None,
        },
    }

    import inspect
    results: Dict[str, Dict[str, float]] = {}
    supports_ctx_kin = (
        hasattr(agent, "set_episode_context")
        and "limb_kinematics" in inspect.signature(agent.set_episode_context).parameters
    )
    supports_act_kin = "limb_kinematics" in inspect.signature(agent.select_action).parameters
    for name, spec in suite_specs.items():
        obs, info = env.reset(options={"tweak_vector": spec["tweak"], "disabled_leg": spec.get("disabled_leg")})
        morph_mask = info.get("morphology_mask", np.ones(agent.cfg.max_action_dim, dtype=np.float32))
        if hasattr(agent, "set_episode_context"):
            ctx_kwargs: Dict[str, Any] = {
                "tweak_vector": info.get("perceived_tweak_vector", spec["tweak"]),
                "action_mask": morph_mask,
            }
            if supports_ctx_kin:
                ctx_kwargs["limb_kinematics"] = info.get("limb_kinematics")
            agent.set_episode_context(**ctx_kwargs)

        ep_return = 0.0
        steps = 0
        last_info = info
        while steps < max_steps:
            if supports_act_kin:
                action = agent.select_action(
                    obs, deterministic=True, limb_kinematics=last_info.get("limb_kinematics")
                )
            else:
                action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_return += float(reward)
            steps += 1
            if terminated or truncated:
                break

        brl_m = last_info.get("brl_metrics", {})
        results[name] = {
            "return": float(ep_return),
            "steps": float(steps),
            "x_velocity": float(brl_m.get("x_velocity", 0.0)),
            "total_distance": float(brl_m.get("total_distance", 0.0)),
        }
        dist_val = float(brl_m.get("total_distance", 0.0))
        vel_val = float(brl_m.get("x_velocity", 0.0))
        suite_line = (
            f"    * {name:<22}: Return = {ep_return:+7.2f} | Steps = {steps:4d}/{max_steps} | "
            f"Dist = {dist_val:+6.2f}m | Vel = {vel_val:+6.3f} m/s"
        )
        if verbose:
            print(suite_line, flush=True)
        if log_fn is not None:
            log_fn(suite_line)

    return results
