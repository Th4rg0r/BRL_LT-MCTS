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
    """Verifies that all limbs share identical MLP parameters in MorphologicalLimbHead."""
    head = MorphologicalLimbHead(latent_dim=256, num_peaks=5, max_joints_per_limb=4, hidden_dim=128)
    # Total parameter count should be strictly that of one 2-layer MLP + mu_head + log_std_head
    num_params = sum(p.numel() for p in head.parameters())
    # in_dim = 256 + 8 = 264. mlp.0: 264*128 + 128 = 33920. mlp.3: 128*128 + 128 = 16512.
    # heads: 2 * (128 * 20 + 20) = 5160. Total ~ 55k parameters.
    assert num_params < 70000, f"Too many parameters for single shared limb head: {num_params}"


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
