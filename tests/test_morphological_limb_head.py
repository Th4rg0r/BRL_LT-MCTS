"""Comprehensive test suite for 3D Kinematic Anchor + Pointing Direction Morphological Limb Head."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
from config import LAMCTSConfig
from dog_morph_env.wrapper import MorphologicalDogBRLWrapper
from humanoid_morph_env.wrapper import MorphologicalHumanoidBRLWrapper
from la_mcts.agent import LatentActionMCTSAgent
from la_mcts.gmm_heads import MorphologicalLimbHead, MultiPeakGMMActorHead
from la_mcts.sampled_mcts import SampledContinuousMCTS


def test_harmonic_kinematic_embedding():
    """Verifies that encode_kinematics maps (x, y, z, yaw, pitch) to 8D with harmonic continuity."""
    kin = torch.tensor([
        [0.2, 0.2, 0.0, 0.0, 0.5],   # yaw = 0.0
        [0.2, 0.2, 0.0, 1.0, 0.5],   # yaw = 1.0 (equivalent to 0.0)
    ], dtype=torch.float32)

    emb = MorphologicalLimbHead.encode_kinematics(kin)
    assert emb.shape == (2, 8)

    # cos(0) == cos(2pi) == 1, sin(0) == sin(2pi) == 0
    assert torch.allclose(emb[0, 4], emb[1, 4], atol=1e-5)  # cos(yaw)
    assert torch.allclose(emb[0, 5], emb[1, 5], atol=1e-5)  # sin(yaw)
    assert torch.allclose(emb[0, 6], emb[1, 6], atol=1e-5)  # pitch
    assert torch.allclose(emb[0, 7], emb[1, 7], atol=1e-5)  # sin(pitch)


def test_limb_head_single_limb_query():
    """Verifies single limb query returns valid mu and log_std parameters."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=4, hidden_dim=128)
    z_t = torch.randn(256)

    # Query 2 joints (e.g. Ant leg)
    mu_ant, log_std_ant = head.forward_single_limb(
        z_t, anchor=(0.2, 0.2, 0.0), yaw=0.125, pitch=0.5, num_joints=2
    )
    assert mu_ant.shape == (5, 2)
    assert log_std_ant.shape == (5, 2)

    # Query 3 joints (e.g. Dog leg)
    mu_dog, log_std_dog = head.forward_single_limb(
        z_t, anchor=(0.188, -0.08, 0.0), yaw=0.875, pitch=0.25, num_joints=3
    )
    assert mu_dog.shape == (5, 3)
    assert log_std_dog.shape == (5, 3)


def test_pitch_and_z_distinguishes_arm_from_leg():
    """Verifies that Humanoid pelvis anchor (z=-0.46, pitch=0.0) vs shoulder anchor (z=+0.06, pitch=0.31) produce distinct outputs."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=4, hidden_dim=128)
    z_t = torch.randn(256)

    # Leg query (pelvis socket, downward)
    mu_leg, _ = head.forward_single_limb(
        z_t, anchor=(-0.016, 0.106, -0.464), yaw=0.15, pitch=0.0, num_joints=3
    )
    # Arm query (shoulder socket, outward/upward)
    mu_arm, _ = head.forward_single_limb(
        z_t, anchor=(-0.002, 0.169, 0.063), yaw=0.63, pitch=0.31, num_joints=3
    )

    diff = torch.norm(mu_leg - mu_arm).item()
    assert diff > 1e-4, f"Limb head failed to distinguish arm from leg: diff={diff}"


def test_anchor_xy_distinguishes_dog_front_from_rear():
    """Verifies Dog front hip (+x) vs rear hip (-x) produces distinct specialized movements."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=3, hidden_dim=128)
    z_t = torch.randn(256)

    mu_front, _ = head.forward_single_limb(
        z_t, anchor=(0.188, 0.08, 0.0), yaw=0.125, pitch=0.25, num_joints=3
    )
    mu_rear, _ = head.forward_single_limb(
        z_t, anchor=(-0.188, 0.08, 0.0), yaw=0.375, pitch=0.25, num_joints=3
    )

    diff = torch.norm(mu_front - mu_rear).item()
    assert diff > 1e-4, f"Limb head failed to distinguish front from rear: diff={diff}"


def test_parameter_sharing_across_all_limbs():
    """Verifies that all limbs share identical Transformer Encoder-Decoder weights in MorphologicalLimbHead."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=4, hidden_dim=128)
    num_params = sum(p.numel() for p in head.parameters())
    assert num_params < 600000, f"Too many parameters for shared Transformer limb head: {num_params}"

    # Parameter count is strictly invariant to number of queried limbs (4 vs 6)
    z_t = torch.randn(2, 256)
    kin_4 = torch.randn(2, 4, 5)
    kin_6 = torch.randn(2, 6, 5)
    mu_4, _ = head.forward_all_limbs(z_t, kin_4)
    mu_6, _ = head.forward_all_limbs(z_t, kin_6)
    assert mu_4.shape == (2, 5, 16)
    assert mu_6.shape == (2, 5, 24)


def test_inter_limb_coordination_attention():
    """Verifies that changing another limb's position/angle alters a limb's output via Transformer attention."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=2, hidden_dim=128)
    head.eval()
    z_t = torch.randn(1, 256)

    kin_a = torch.tensor([[
        [ 0.2,  0.2, 0.0, 0.125, 0.50],
        [-0.2,  0.2, 0.0, 0.875, 0.50],
        [-0.2, -0.2, 0.0, 0.375, 0.50],
        [ 0.2, -0.2, 0.0, 0.625, 0.50],
    ]], dtype=torch.float32)

    kin_b = kin_a.clone()
    # Perturb limb 2 (index 1) significantly
    kin_b[0, 1] = torch.tensor([-0.2, 0.2, 0.0, 0.25, 0.90])

    with torch.no_grad():
        mu_a, _ = head.forward_all_limbs(z_t, kin_a, limb_joint_counts=[2, 2, 2, 2])
        mu_b, _ = head.forward_all_limbs(z_t, kin_b, limb_joint_counts=[2, 2, 2, 2])

    # Limb 1 (joints 0:2) must respond to Limb 2's state change
    limb1_diff = torch.norm(mu_a[0, :, :2] - mu_b[0, :, :2]).item()
    assert limb1_diff > 1e-4, f"Limb 1 failed to attend to Limb 2's kinematics: diff={limb1_diff}"


def test_actor_head_all_environments_shape():
    """Verifies MultiPeakGMMActorHead forward passes across Ant, Dog, and Humanoid shapes."""
    z = torch.randn(4, 256)

    # 1. Ant (8 action dims)
    actor_ant = MultiPeakGMMActorHead(latent_dim=256, action_dim=8, num_peaks=5)
    dist_ant = actor_ant(z)
    samples_ant = dist_ant.sample()
    assert samples_ant.shape == (4, 8)
    assert torch.all(samples_ant >= -1.0) and torch.all(samples_ant <= 1.0)

    # 2. Dog (12 action dims)
    actor_dog = MultiPeakGMMActorHead(latent_dim=256, action_dim=12, num_peaks=5)
    dist_dog = actor_dog(z)
    samples_dog = dist_dog.sample()
    assert samples_dog.shape == (4, 12)

    # 3. Humanoid (17 action dims)
    actor_hum = MultiPeakGMMActorHead(latent_dim=256, action_dim=17, num_peaks=5)
    dist_hum = actor_hum(z)
    samples_hum = dist_hum.sample()
    assert samples_hum.shape == (4, 17)


def test_env_wrappers_provide_limb_kinematics():
    """Verifies that Ant, Dog, and Humanoid wrappers provide limb_kinematics in reset() and step()."""
    # Ant
    ant_env = MorphologicalAntBRLWrapper()
    obs_ant, info_ant = ant_env.reset()
    assert "limb_kinematics" in info_ant
    kin_ant = info_ant["limb_kinematics"]
    assert kin_ant.shape == (4, 5)
    _, _, _, _, step_info_ant = ant_env.step(np.zeros(8, dtype=np.float32))
    assert "limb_kinematics" in step_info_ant
    assert step_info_ant["limb_kinematics"].shape == (4, 5)

    # Dog
    dog_env = MorphologicalDogBRLWrapper()
    obs_dog, info_dog = dog_env.reset()
    assert "limb_kinematics" in info_dog
    kin_dog = info_dog["limb_kinematics"]
    assert kin_dog.shape == (4, 5)
    _, _, _, _, step_info_dog = dog_env.step(np.zeros(12, dtype=np.float32))
    assert "limb_kinematics" in step_info_dog
    assert step_info_dog["limb_kinematics"].shape == (4, 5)

    # Humanoid
    hum_env = MorphologicalHumanoidBRLWrapper()
    obs_hum, info_hum = hum_env.reset()
    assert "limb_kinematics" in info_hum
    kin_hum = info_hum["limb_kinematics"]
    assert kin_hum.shape == (5, 5)
    _, _, _, _, step_info_hum = hum_env.step(np.zeros(17, dtype=np.float32))
    assert "limb_kinematics" in step_info_hum
    assert step_info_hum["limb_kinematics"].shape == (5, 5)


def test_agent_get_limb_movement():
    """Verifies that LatentActionMCTSAgent.get_limb_movement produces valid limb actions."""
    cfg = LAMCTSConfig(obs_dim=27, max_action_dim=8, n_sim=5, k_act=4)
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")

    obs = np.zeros(27, dtype=np.float32)
    # Query front-left leg
    limb_action = agent.get_limb_movement(
        obs=obs,
        anchor=(0.2, 0.2, 0.0),
        yaw=0.125,
        pitch=0.50,
        num_joints=2,
        deterministic=True,
    )
    assert isinstance(limb_action, np.ndarray)
    assert limb_action.shape == (2,)
    assert np.all(limb_action >= -1.0) and np.all(limb_action <= 1.0)


def test_mcts_search_with_kinematic_limb_head():
    """Verifies that MCTS search runs seamlessly with limb kinematics."""
    cfg = LAMCTSConfig(obs_dim=27, max_action_dim=8, n_sim=10, k_act=8)
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")

    kin = np.array([
        [ 0.2,  0.2, 0.0, 0.125, 0.50],
        [-0.2,  0.2, 0.0, 0.875, 0.50],
        [-0.2, -0.2, 0.0, 0.375, 0.50],
        [ 0.2, -0.2, 0.0, 0.625, 0.50],
    ], dtype=np.float32)

    agent.set_episode_context(tweak_vector=np.zeros(18), limb_kinematics=kin)
    obs = np.zeros(27, dtype=np.float32)

    action = agent.select_action(obs, deterministic=False)
    assert action.shape == (8,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_end_to_end_training_unroll_gradient():
    """Verifies gradient propagation back through the morphological limb head during training."""
    cfg = LAMCTSConfig(obs_dim=27, max_action_dim=8, n_sim=5, k_act=4, unroll_steps=2, batch_size=2)
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")

    # Record dummy transitions ending with terminated=True
    for step in range(5):
        obs = np.zeros(27, dtype=np.float32)
        act = np.zeros(8, dtype=np.float32)
        is_done = (step == 4)
        agent.observe_transition(obs, act, 1.0, obs, is_done, False, {})

    # Verify trajectory in replay buffer
    assert len(agent.replay_buffer) > 0

    # Execute a train step
    metrics = agent.train_step()
    assert "loss_total" in metrics

    # Verify that MorphologicalLimbHead parameters received gradients
    has_grads = any(
        p.grad is not None and torch.norm(p.grad) > 0
        for p in agent.actor_head.limb_head.parameters()
    )
    assert has_grads, "MorphologicalLimbHead parameters received no gradients during train step!"


@pytest.mark.parametrize(
    "obs_dim,act_dim,n_limbs",
    [
        (27, 8, 4),    # Ant
        (43, 12, 4),   # Dog (Unitree Go1)
        (376, 17, 5),  # Humanoid
    ],
)
def test_per_step_limb_kinematics_propagation(obs_dim: int, act_dim: int, n_limbs: int):
    """Verifies per-step limb_kinematics storage, unroll batching, reanalyze, and gradients across Ant, Dog, and Humanoid."""
    cfg = LAMCTSConfig(
        obs_dim=obs_dim,
        max_action_dim=act_dim,
        n_sim=4,
        k_act=4,
        unroll_steps=2,
        batch_size=2,
    )
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")
    kin_init = np.random.randn(n_limbs, 5).astype(np.float32)
    agent.set_episode_context(tweak_vector=np.zeros(18), limb_kinematics=kin_init)

    for step in range(5):
        obs = np.zeros(obs_dim, dtype=np.float32)
        step_kin = np.random.randn(n_limbs, 5).astype(np.float32)
        act = agent.select_action(obs, deterministic=False, limb_kinematics=step_kin)
        agent.observe_transition(obs, act, 1.0, obs, step == 4, False, {"limb_kinematics": step_kin})

    assert len(agent.replay_buffer) == 1
    assert len(agent.replay_buffer.trajectories[0].limb_kinematics_seq) == 5

    batch = agent.replay_buffer.sample_unroll_batch(batch_size=2, device=torch.device("cpu"))
    kin_batch = batch[9]
    assert kin_batch is not None
    assert kin_batch.shape == (2, 3, n_limbs, 5)

    agent.reanalyze_batch(num_states=2)
    metrics = agent.train_step()
    assert "loss_total" in metrics
    assert any(
        p.grad is not None and torch.norm(p.grad) > 0
        for p in agent.actor_head.limb_head.parameters()
    )


def test_compositional_peak_pos_encoding_and_weights():
    """Verifies peak positional encoding [0..K-1], per-limb per-peak weights, and composition selection."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=2, hidden_dim=128)
    z_t = torch.randn(3, 256)
    kin = torch.randn(3, 4, 5)
    peak_logits, mu, log_std, limb_weights = head.forward_all_limbs(
        z_t, kin, limb_joint_counts=[2, 2, 2, 2], return_weights=True
    )
    assert peak_logits.shape == (3, 5)
    assert mu.shape == (3, 5, 8)
    assert log_std.shape == (3, 5, 8)
    assert limb_weights.shape == (3, 5, 4)
    assert not torch.allclose(mu[:, 0, :], mu[:, 1, :])
    actor = MultiPeakGMMActorHead(latent_dim=256, action_dim=8, num_peaks=5)
    action, selected_peak, peak_probs = actor.select_composition(z_t, limb_kinematics=kin)
    assert action.shape == (3, 8)
    assert selected_peak.shape == (3,)
    assert peak_probs.shape == (3, 5)
    assert torch.allclose(peak_probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_entropy_regularization_and_adaptive_exploration():
    """Verifies differentiable policy entropy proxy and adaptive MCTS exploration scaling."""
    actor = MultiPeakGMMActorHead(latent_dim=256, action_dim=8, num_peaks=4)
    z_t = torch.randn(2, 256, requires_grad=True)
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32)
    tot_ent, peak_ent, mean_std = actor.entropy_proxy(z_t, action_mask=mask)
    assert tot_ent.shape == (2,)
    assert peak_ent.shape == (2,)
    assert mean_std.shape == (2,)
    tot_ent.mean().backward()
    assert z_t.grad is not None and torch.norm(z_t.grad) > 0

    cfg = LAMCTSConfig(
        obs_dim=27,
        max_action_dim=8,
        n_sim=4,
        k_act=4,
        unroll_steps=2,
        batch_size=2,
        adaptive_exploration=True,
        min_policy_std_target=0.55,
        exploration_boost_max=2.0,
    )
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")
    agent._diagnostics["policy_std"] = 0.275  # half of target -> boost should be 2.0
    obs = np.zeros(27, dtype=np.float32)
    _ = agent.select_action(obs, deterministic=False)
    assert abs(agent.get_diagnostics()["exploration_scale"] - 2.0) < 1e-5

