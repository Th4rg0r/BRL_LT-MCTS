"""Unit and Integration Tests for Perception Correctness & Distortion in BRL Locomotion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
from brl_core.balancer import DDPGBalancer
from brl_core.knapsack_proj import DifferentiableKnapsackProjection
from brl_core.orchestrator import BRLTrainingOrchestrator
from brl_core.perception import ModalitySliceSpec, PerceptionDistortionFilter
from brl_core.tweaker import StochasticTweakerActorCritic
from config import BRLConfig, LAMCTSConfig, SystemConfig
from dog_morph_env.wrapper import MorphologicalDogBRLWrapper
from humanoid_morph_env.wrapper import MorphologicalHumanoidBRLWrapper
from la_mcts.agent import LatentActionMCTSAgent, EpisodeTrajectory, build_step_history
from la_mcts.backbone import SelfAttentionBackbone


def test_modality_slice_spec_creation() -> None:
    """Verifies modality slice specifications across Ant, Dog, and Humanoid."""
    # Ant spec without contact forces (27D)
    ant_spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=False)
    assert ant_spec.obs_dim == 27
    assert len(ant_spec.imu_indices) == 11
    assert len(ant_spec.joint_indices) == 16
    assert len(ant_spec.contact_indices) == 0

    # Ant spec with contact forces (105D)
    ant_cfrc_spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=True)
    assert ant_cfrc_spec.obs_dim == 105
    assert len(ant_cfrc_spec.contact_indices) == 78

    # Dog spec (48D)
    dog_spec = ModalitySliceSpec.create_dog_spec()
    assert dog_spec.obs_dim == 48
    assert len(dog_spec.imu_indices) == 12
    assert len(dog_spec.joint_indices) == 24
    assert len(dog_spec.contact_indices) == 12
    assert dog_spec.imu_slice == slice(0, 12)
    assert dog_spec.cfrc_slice == slice(36, 48)

    # Humanoid spec
    humanoid_spec = ModalitySliceSpec.create_humanoid_spec(obs_dim=348)
    assert humanoid_spec.obs_dim == 348
    assert len(humanoid_spec.imu_indices) == 22
    assert len(humanoid_spec.joint_indices) == 158
    assert len(humanoid_spec.contact_indices) == 168


def test_exact_perception_zero_distortion() -> None:
    """Verifies that w_12..w_17 = 0 produces exact, lossless perception."""
    spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=False)
    pfilter = PerceptionDistortionFilter(slice_spec=spec, obs_dim=27)

    # 18D tweak vector with 0 in perception slots 12..17
    tweak = np.zeros(18, dtype=np.float32)
    tweak[:12] = np.array([100, 200, 150, 50, 300, 100, 50, 80, 120, 0, 50, 10], dtype=np.float32)
    true_leg_scales = np.array([1.12, 0.87, 1.06, 0.94], dtype=np.float32)

    pfilter.reset_episode(tweak_vector=tweak, true_leg_scales=true_leg_scales, seed=42)

    # Perception exactness metrics should be 1.0
    assert np.allclose(pfilter.perceived_leg_scales, true_leg_scales, atol=1e-6)
    assert np.allclose(pfilter.perceived_context_vector, tweak, atol=1e-6)

    # Clean observation
    obs = np.linspace(-1.0, 1.0, 27, dtype=np.float32)
    distorted_obs, metrics = pfilter.distort_observation(obs)

    assert np.allclose(distorted_obs, obs, atol=1e-6)
    assert metrics["perception_correctness"] == 1.0
    assert metrics["leg_length_exactness"] == 1.0
    assert metrics["obs_exactness"] == 1.0
    assert metrics["obs_rmse"] == 0.0


def test_inexact_leg_length_perception() -> None:
    """Verifies that w_12 > 0 distorts perceived leg scales while leaving physical dynamics true."""
    spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=False)
    pfilter = PerceptionDistortionFilter(slice_spec=spec, obs_dim=27)

    tweak = np.zeros(18, dtype=np.float32)
    tweak[0] = 500.0  # s_front base
    tweak[1] = 500.0  # s_back base
    tweak[2] = 500.0  # s_left / s_right base
    tweak[12] = 800.0  # Severe leg-length perception inexactness

    true_leg_scales = np.array([1.60, 0.675, 1.225, 0.775], dtype=np.float32)
    pfilter.reset_episode(tweak_vector=tweak, true_leg_scales=true_leg_scales, seed=123)

    # Perceived leg scales must deviate from true leg scales
    assert not np.allclose(pfilter.perceived_leg_scales, true_leg_scales, atol=1e-3)
    # Leg length exactness should drop below 1.0
    assert pfilter.last_leg_length_exactness < 0.95
    # Perceived context vector should differ in leg geometry indices (0, 1, 2)
    assert pfilter.perceived_context_vector[12] == tweak[12]
    assert not np.allclose(pfilter.perceived_context_vector[:3], tweak[:3], atol=1e-2)


def test_modality_specific_corruptions() -> None:
    """Verifies that individual inexactness knobs isolate perturbations to their target modalities."""
    spec = ModalitySliceSpec.create_dog_spec()
    pfilter = PerceptionDistortionFilter(slice_spec=spec, obs_dim=48)

    base_obs = np.ones(48, dtype=np.float32)

    # Test 1: Joint encoder inexactness (w_13 > 0, all others 0)
    tweak_joints = np.zeros(18, dtype=np.float32)
    tweak_joints[13] = 750.0  # Joint corruption
    pfilter.reset_episode(tweak_vector=tweak_joints, seed=10)
    dist_joints, _ = pfilter.distort_observation(base_obs)

    # Joint positions (12:24) and velocities (24:36) should be corrupted
    assert not np.allclose(dist_joints[12:36], base_obs[12:36], atol=1e-3)
    # IMU (0:12) and contact forces (36:48) should remain uncorrupted by joint knob
    assert np.allclose(dist_joints[0:12], base_obs[0:12], atol=1e-5)
    assert np.allclose(dist_joints[36:48], base_obs[36:48], atol=1e-5)

    # Test 2: IMU inexactness (w_14 > 0, all others 0)
    tweak_imu = np.zeros(18, dtype=np.float32)
    tweak_imu[14] = 750.0
    pfilter.reset_episode(tweak_vector=tweak_imu, seed=20)
    dist_imu, _ = pfilter.distort_observation(base_obs)

    assert not np.allclose(dist_imu[0:12], base_obs[0:12], atol=1e-3)
    assert np.allclose(dist_imu[12:48], base_obs[12:48], atol=1e-5)

    # Test 3: Contact sensor inexactness (w_15 > 0, all others 0)
    tweak_contact = np.zeros(18, dtype=np.float32)
    tweak_contact[15] = 750.0
    pfilter.reset_episode(tweak_vector=tweak_contact, seed=30)
    dist_contact, _ = pfilter.distort_observation(base_obs)

    assert not np.allclose(dist_contact[36:48], base_obs[36:48], atol=1e-3)
    assert np.allclose(dist_contact[0:36], base_obs[0:36], atol=1e-5)


def test_quantization_and_latency() -> None:
    """Verifies quantization discretization and temporal latency buffer."""
    spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=False)
    pfilter = PerceptionDistortionFilter(slice_spec=spec, obs_dim=27, max_latency_steps=4)

    # Test quantization (w_16 = 1000 => coarsest grid)
    tweak_quant = np.zeros(18, dtype=np.float32)
    tweak_quant[16] = 1000.0
    pfilter.reset_episode(tweak_vector=tweak_quant, seed=1)

    obs = np.array([0.1234567] * 27, dtype=np.float32)
    dist_quant, _ = pfilter.distort_observation(obs)
    # Coarse quantization step at 1000 is ~0.15, value must snap to discrete grid
    assert not np.isclose(dist_quant[0], obs[0], atol=1e-4)

    # Test latency and dropout (w_17 = 500 => latency_steps = 1)
    tweak_lat = np.zeros(18, dtype=np.float32)
    tweak_lat[17] = 500.0
    pfilter.reset_episode(tweak_vector=tweak_lat, seed=1)
    pfilter._dropout_prob = 0.0

    obs1 = np.ones(27, dtype=np.float32) * 1.0
    obs2 = np.ones(27, dtype=np.float32) * 5.0
    dist1, _ = pfilter.distort_observation(obs1)
    dist2, _ = pfilter.distort_observation(obs2)
    # Under 1-step latency, step 2 should return a stale observation from history buffer
    assert np.allclose(dist2, obs1, atol=1e-4)

    # Test dropout: 100% dropout should retain previous emitted observation
    pfilter._dropout_prob = 1.0
    obs3 = np.ones(27, dtype=np.float32) * 99.0
    dist3, _ = pfilter.distort_observation(obs3)
    assert np.allclose(dist3, dist2, atol=1e-4)


def test_18d_differentiable_knapsack_projection() -> None:
    """Verifies that DifferentiableKnapsackProjection satisfies all constraints on P=18."""
    rng = np.random.default_rng(999)
    P = 18

    limits_np = rng.uniform(100.0, 1000.0, size=(P,))
    costs_np = rng.uniform(0.5, 5.0, size=(P,))
    u_np = rng.uniform(0.2, 0.8, size=(P,))
    w_raw_np = limits_np * u_np
    raw_spend = float(np.sum(w_raw_np * costs_np))
    budget_val = raw_spend * 0.60  # Require active projection

    w_raw = torch.tensor(w_raw_np, dtype=torch.float64, requires_grad=True)
    costs = torch.tensor(costs_np, dtype=torch.float64)
    budget = torch.tensor([budget_val], dtype=torch.float64)
    limits = torch.tensor(limits_np, dtype=torch.float64)

    w_proj = DifferentiableKnapsackProjection.apply(w_raw, costs, budget, limits)

    # Primal feasibility
    assert float(w_proj.min().item()) >= -1e-6
    assert float((w_proj - limits).max().item()) <= 1e-5
    total_cost = float((w_proj * costs).sum().item())
    assert total_cost <= budget_val + 1e-4

    # Autograd gradcheck
    torch.autograd.gradcheck(
        lambda w: DifferentiableKnapsackProjection.apply(w, costs, budget, limits),
        (w_raw,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_limb_kinematics_perception_distortion() -> None:
    """Verifies that w_13 distorts limb position and w_14 distorts limb pointing."""
    spec = ModalitySliceSpec.create_ant_spec(use_contact_forces=False)
    pfilter = PerceptionDistortionFilter(slice_spec=spec, obs_dim=27)

    # 1. Zero tweak -> 100% exact limb kinematics
    tweak_zero = np.zeros(18, dtype=np.float32)
    pfilter.reset_episode(tweak_vector=tweak_zero, seed=42)
    clean_kin = np.array([
        [0.2, 0.2, 0.0, 0.625, 0.5],
        [-0.2, 0.2, 0.0, 0.875, 0.5],
        [-0.2, -0.2, 0.0, 0.125, 0.5],
        [0.2, -0.2, 0.0, 0.375, 0.5],
    ], dtype=np.float32)

    perc_kin, metrics = pfilter.distort_limb_kinematics(clean_kin)
    assert np.allclose(perc_kin, clean_kin, atol=1e-6)
    assert metrics["limb_kinematics_exactness"] == 1.0
    assert metrics["limb_pos_rmse"] == 0.0
    assert metrics["limb_pointing_rmse"] == 0.0

    # 2. Position distortion only (w[12] = 800)
    tweak_pos = np.zeros(18, dtype=np.float32)
    tweak_pos[12] = 800.0
    pfilter.reset_episode(tweak_vector=tweak_pos, seed=42)
    perc_kin_pos, m_pos = pfilter.distort_limb_kinematics(clean_kin)
    assert m_pos["limb_pos_rmse"] > 0.005
    assert m_pos["limb_pointing_rmse"] == 0.0  # pointing unchanged
    assert m_pos["limb_kinematics_exactness"] < 1.0

    # 3. Pointing distortion only (w[13] = 800)
    tweak_rot = np.zeros(18, dtype=np.float32)
    tweak_rot[13] = 800.0
    pfilter.reset_episode(tweak_vector=tweak_rot, seed=42)
    perc_kin_rot, m_rot = pfilter.distort_limb_kinematics(clean_kin)
    assert m_rot["limb_pos_rmse"] == 0.0  # position unchanged
    assert m_rot["limb_pointing_rmse"] > 0.005
    assert m_rot["limb_kinematics_exactness"] < 1.0


def test_ant_wrapper_perception_integration() -> None:
    """Tests MorphologicalAntBRLWrapper with 18D tweak vector and perception metrics."""
    env = MorphologicalAntBRLWrapper(env_id="Ant-v5", use_contact_forces=False)
    try:
        tweak_18d = np.zeros(18, dtype=np.float32)
        tweak_18d[0] = 300.0   # Front leg length
        tweak_18d[12] = 600.0  # Perception leg inexactness & limb position noise
        tweak_18d[13] = 400.0  # Proprioceptive joint noise & limb pointing noise

        obs, info = env.reset(options={"tweak_vector": tweak_18d})
        assert obs.shape == (27,)
        assert "perceived_tweak_vector" in info
        assert "perception_metrics" in info
        assert info["perception_metrics"]["leg_length_exactness"] < 1.0
        assert "limb_kinematics" in info
        assert "true_limb_kinematics" in info
        assert "limb_kinematics_exactness" in info["perception_metrics"]

        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        next_obs, reward, term, trunc, step_info = env.step(action)
        assert next_obs.shape == (27,)
        assert "perception_metrics" in step_info
        assert "perceived_tweak_vector" in step_info["brl_metrics"]
        assert "limb_kinematics" in step_info

        # Backward compatibility: verify 12D input is seamlessly accepted
        tweak_12d = np.ones(12, dtype=np.float32) * 100.0
        obs_12, info_12 = env.reset(options={"tweak_vector": tweak_12d})
        assert obs_12.shape == (27,)
        assert info_12["perceived_tweak_vector"].shape == (18,)
    finally:
        env.close()


def test_dog_wrapper_perception_integration() -> None:
    """Tests MorphologicalDogBRLWrapper with 18D tweak vector and perception metrics."""
    env = MorphologicalDogBRLWrapper()
    try:
        tweak_18d = np.zeros(18, dtype=np.float32)
        tweak_18d[12] = 500.0  # Limb position noise
        tweak_18d[13] = 500.0  # Joint noise & limb pointing noise
        tweak_18d[15] = 400.0  # Contact force noise

        obs, info = env.reset(options={"tweak_vector": tweak_18d})
        assert obs.shape == (48,)
        assert "perceived_tweak_vector" in info
        assert "perception_metrics" in info
        assert "limb_kinematics" in info
        assert "limb_kinematics_exactness" in info["perception_metrics"]
        assert "brl_metrics" in info

        action = np.zeros(12, dtype=np.float32)
        next_obs, reward, term, trunc, step_info = env.step(action)
        assert next_obs.shape == (48,)
        assert "perception_metrics" in step_info
        assert "limb_kinematics" in step_info
        assert "perceived_leg_scales" in step_info
    finally:
        env.close()


def test_humanoid_wrapper_perception_integration() -> None:
    """Tests MorphologicalHumanoidBRLWrapper with 18D tweak vector and perception metrics."""
    env = MorphologicalHumanoidBRLWrapper(include_cfrc_ext=True)
    try:
        tweak_18d = np.zeros(18, dtype=np.float32)
        tweak_18d[0] = 300.0   # Torso mass
        tweak_18d[12] = 600.0  # Limb position noise
        tweak_18d[13] = 400.0  # Joint noise & limb pointing noise

        obs, info = env.reset(options={"tweak_vector": tweak_18d})
        assert obs.shape == (348,)
        assert "perceived_tweak_vector" in info
        assert "perception_metrics" in info
        assert "limb_kinematics" in info
        assert info["limb_kinematics"].shape == (5, 5)
        assert "limb_kinematics_exactness" in info["perception_metrics"]
        assert "brl_metrics" in info

        action = np.zeros(17, dtype=np.float32)
        next_obs, reward, term, trunc, step_info = env.step(action)
        assert next_obs.shape == (348,)
        assert "perception_metrics" in step_info
        assert "limb_kinematics" in step_info
        assert "perceived_leg_scales" in step_info
        assert "perceived_leg_scales" in step_info["brl_metrics"]
    finally:
        env.close()


def test_agent_dual_context_and_history_adapter() -> None:
    """Verifies that SelfAttentionBackbone and Agent operate with both context and rolling history."""
    cfg = LAMCTSConfig(
        obs_dim=27,
        context_dim=18,
        history_len=4,
        max_action_dim=8,
        batch_size=4,
        unroll_steps=2,
    )
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")

    # Set 18D episode context
    ctx = np.ones(18, dtype=np.float32) * 500.0
    agent.set_episode_context(tweak_vector=ctx)

    # Initial action selection (history starts empty/zero-padded)
    obs0 = np.random.randn(27).astype(np.float32)
    act0 = agent.select_action(obs0, deterministic=False)
    assert act0.shape == (8,)

    # Observe multiple transitions to populate rolling history buffer
    for i in range(5):
        obs = np.random.randn(27).astype(np.float32)
        act = np.random.randn(8).astype(np.float32)
        agent.observe_transition(obs, act, 1.0, obs, False, False, {})

    # Reanalyze batch with history
    agent.reanalyze_batch(num_states=2)

    # Unroll training step with history
    loss_dict = agent.train_step()
    assert "loss_total" in loss_dict


def test_orchestrator_perception_telemetry() -> None:
    """Verifies that BRLTrainingOrchestrator collects perception telemetry and passes perceived tweaks."""
    env = MorphologicalAntBRLWrapper(env_id="Ant-v5", use_contact_forces=False)
    try:
        sys_cfg = SystemConfig()
        sys_cfg.la_mcts.obs_dim = 27
        sys_cfg.la_mcts.context_dim = 18
        sys_cfg.la_mcts.max_action_dim = 8
        sys_cfg.brl.num_params = 18

        agent = LatentActionMCTSAgent(config=sys_cfg.la_mcts, device="cpu")
        tweaker = StochasticTweakerActorCritic(config=sys_cfg.brl, device="cpu")
        balancer = DDPGBalancer(config=sys_cfg.brl, device="cpu")

        orchestrator = BRLTrainingOrchestrator(
            env=env,
            agent=agent,
            tweaker=tweaker,
            balancer=balancer,
            config=sys_cfg,
        )

        summary = orchestrator.run_episode(episode_idx=1, max_steps=5, train_agent_every=5)

        assert "perception_correctness" in summary
        assert "leg_length_exactness" in summary
        assert "obs_exactness" in summary
        assert "obs_rmse" in summary
        assert "mean_physics_tweak" in summary
        assert "mean_perception_tweak" in summary
        assert orchestrator.last_perception_correctness <= 1.0
    finally:
        env.close()


def test_balancer_mismatched_tweak_failure_correlations() -> None:
    """Verifies that DDPGBalancerReplayBuffer gracefully handles mismatched lengths on resume."""
    cfg = BRLConfig(num_params=18)
    balancer = DDPGBalancer(config=cfg, device="cpu")
    buf = balancer.replay_buffer

    # Case 1: recent_tweaks has 3 items, recent_failures has 50 items (the resume bug)
    buf.recent_failures.extend([1.0 if i % 2 == 0 else 0.0 for i in range(50)])
    buf.recent_tweaks.extend([np.random.uniform(0, 100, size=18).astype(np.float32) for _ in range(3)])
    corr = buf.get_failure_correlations()
    assert corr.shape == (18,)
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0)

    # Case 2: recent_tweaks has 50 items, recent_failures has 3 items
    buf.recent_tweaks.clear()
    buf.recent_failures.clear()
    buf.recent_tweaks.extend([np.random.uniform(0, 100, size=18).astype(np.float32) for _ in range(50)])
    buf.recent_failures.extend([1.0, 0.0, 1.0])
    corr = buf.get_failure_correlations()
    assert corr.shape == (18,)

    # Case 3: recent_tweaks has 0 items, recent_failures has 50 items
    buf.recent_tweaks.clear()
    buf.recent_failures.clear()
    buf.recent_failures.extend([1.0 for _ in range(50)])
    corr = buf.get_failure_correlations()
    assert corr.shape == (18,)
    assert np.all(corr == 0.0)

    # Case 4: build_state works without broadcast crash
    state = balancer.build_state(
        ema_win_rate=0.5,
        last_win=0,
        dominance_score=0.2,
    )
    assert state.shape == (balancer.state_dim,)

