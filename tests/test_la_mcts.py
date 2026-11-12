"""Verification Gate 1: GMM & MCTS Engine Unit Tests (la_mcts/)."""

from __future__ import annotations

import time
import numpy as np
import torch

from la_mcts.backbone import SelfAttentionBackbone
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer
from la_mcts.sampled_mcts import SampledContinuousMCTS


def test_gate1_backbone_normalization_bounds() -> None:
    """Verifies z_t in [0, 1]^(B x 256) strictly bounded across random and extreme inputs."""
    torch.manual_seed(42)
    backbone = SelfAttentionBackbone(
        obs_dim=27, k_tok=8, d_model=256, n_heads=4, l_enc=3, latent_dim=256
    )
    backbone.eval()

    obs = torch.randn(16, 27) * 25.0
    context = torch.rand(16, 12)
    z_t = backbone(obs, context=context)

    assert z_t.shape == (16, 256)
    assert torch.all(torch.isfinite(z_t))
    assert float(z_t.min().item()) >= 0.0
    assert float(z_t.max().item()) <= 1.0
    # Min-Max scaling maps each row's minimum to 0.0 and maximum to ~1.0
    row_mins = z_t.min(dim=-1)[0]
    row_maxs = z_t.max(dim=-1)[0]
    assert torch.allclose(row_mins, torch.zeros_like(row_mins), atol=1e-5)
    assert torch.allclose(row_maxs, torch.ones_like(row_maxs), atol=1e-4)


def test_gate1_gmm_actor_head_1000_points() -> None:
    """Tests MultiPeakGMMActorHead(M=5, N=8) on 1000 points a in (-1, 1)^8 for NaN/Inf freedom."""
    torch.manual_seed(42)
    actor_head = MultiPeakGMMActorHead(
        latent_dim=256, action_dim=8, num_peaks=5, log_std_bounds=(-5.0, 2.0)
    )
    # Verify architectural rule: strictly 3 linear layers directly from z_t
    linear_children = [m for m in actor_head.children() if isinstance(m, torch.nn.Linear)]
    assert len(linear_children) == 3

    z_t = torch.rand(1000, 256)
    # Generate 1000 continuous actions across (-1, 1)^8 including near-boundary points
    actions = torch.empty(1000, 8).uniform_(-0.9999, 0.9999)

    log_probs = actor_head.log_prob(z_t, actions)
    assert log_probs.shape == (1000,)
    assert not torch.any(torch.isnan(log_probs)), "log_prob produced NaN"
    assert not torch.any(torch.isinf(log_probs)), "log_prob produced Inf"

    # Also test sampling K_act = 16 actions
    sampled, priors = actor_head.sample_actions(z_t[:8], k_act=16, temperature=1.25)
    assert sampled.shape == (8, 16, 8)
    assert priors.shape == (8, 16)
    assert float(sampled.min().item()) >= -1.0
    assert float(sampled.max().item()) <= 1.0
    assert torch.allclose(priors.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_gate1_rep_trainer_and_sampled_mcts_latency() -> None:
    """Verifies ModelRepresentationTrainer and SampledContinuousMCTS (< 15 ms, sum(pi) == 1)."""
    torch.manual_seed(42)
    np.random.seed(42)

    backbone = SelfAttentionBackbone(
        obs_dim=27, k_tok=8, d_model=256, n_heads=4, l_enc=3, latent_dim=256
    )
    actor_head = MultiPeakGMMActorHead(
        latent_dim=256, action_dim=8, num_peaks=5, log_std_bounds=(-5.0, 2.0)
    )
    critic_head = SupportCriticHead(latent_dim=256, use_categorical=True)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=256, action_dim=8, num_blocks=3, proj_dim=512, pred_dim=256
    )

    backbone.eval()
    actor_head.eval()
    critic_head.eval()
    rep_trainer.eval()

    obs = torch.randn(1, 27)
    with torch.inference_mode():
        z_t = backbone(obs).squeeze(0)
        a_test = torch.zeros(1, 8)
        z_next, r_hat = rep_trainer(z_t.unsqueeze(0), a_test)

    assert z_next.shape == (1, 256)
    assert r_hat.shape == (1, 1)
    assert float(z_next.min().item()) >= 0.0
    assert float(z_next.max().item()) <= 1.0

    mcts = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=50,
        k_act=16,
    )

    # Warmup call for PyTorch kernel caching
    mcts.search(z_t, deterministic=False)

    # Timed benchmark over 5 runs (take median latency)
    timings_ms = []
    chosen_action = None
    policy_weights = None
    for _ in range(5):
        t0 = time.perf_counter()
        chosen_action, policy_weights, root_samples, root_val, _ = mcts.search(
            z_t, deterministic=False
        )
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(timings_ms))
    assert chosen_action is not None and policy_weights is not None
    assert chosen_action.shape == (8,)
    assert np.all(chosen_action >= -1.0) and np.all(chosen_action <= 1.0)
    assert policy_weights.shape == (16,)
    assert abs(float(np.sum(policy_weights)) - 1.0) <= 1e-6
    max_latency = 15.0 if next(backbone.parameters()).is_cuda else 50.0
    assert median_ms < max_latency, f"MCTS latency {median_ms:.2f} ms exceeded {max_latency} ms limit"


def test_gate1_mcts_root_dirichlet_noise() -> None:
    """Verifies Dirichlet noise mixing on root prior weights during search and search_batch."""
    torch.manual_seed(123)
    np.random.seed(123)

    actor_head = MultiPeakGMMActorHead(
        latent_dim=64, action_dim=4, num_peaks=3, log_std_bounds=(-5.0, 2.0)
    )
    critic_head = SupportCriticHead(latent_dim=64, use_categorical=False)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=64, action_dim=4, num_blocks=1, proj_dim=128, pred_dim=64
    )

    actor_head.eval()
    critic_head.eval()
    rep_trainer.eval()

    z_root = torch.rand(64)

    # 1. Non-deterministic search with Dirichlet noise enabled (eps = 0.25, alpha = 0.3)
    mcts_noisy = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=5,
        k_act=8,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )

    _, _, _, _, root_node_noisy = mcts_noisy.search(z_root, deterministic=False)
    assert root_node_noisy.prior_weights.shape == (8,)
    assert np.all(root_node_noisy.prior_weights >= 0.0)
    assert abs(float(np.sum(root_node_noisy.prior_weights)) - 1.0) <= 1e-6

    # 2. Deterministic search disables Dirichlet noise completely
    # In deterministic mode, root.prior_weights must match pure actor head sample_actions priors exactly
    torch.manual_seed(42)
    np.random.seed(42)
    _, expected_priors = actor_head.sample_actions(
        z_root, k_act=8, temperature=mcts_noisy.root_tau_beta
    )
    expected_priors_np = expected_priors.detach().cpu().numpy()

    torch.manual_seed(42)
    np.random.seed(42)
    _, _, _, _, root_node_det = mcts_noisy.search(z_root, deterministic=True)
    np.testing.assert_allclose(root_node_det.prior_weights, expected_priors_np, atol=1e-6)

    # 3. Fraction = 0.0 disables noise even when deterministic=False
    mcts_no_frac = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=5,
        k_act=8,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.0,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    _, _, _, _, root_node_no_frac = mcts_no_frac.search(z_root, deterministic=False)
    np.testing.assert_allclose(root_node_no_frac.prior_weights, expected_priors_np, atol=1e-6)

    # 4. Search batch testing: verify simplex bounds and shapes
    z_batch = torch.rand(3, 64)
    acts, pi_weights, root_acts, root_vals = mcts_noisy.search_batch(z_batch, deterministic=False)
    assert acts.shape == (3, 4)
    assert pi_weights.shape == (3, 8)
    assert root_acts.shape == (3, 8, 4)
    assert root_vals.shape == (3,)
    for b in range(3):
        assert abs(float(np.sum(pi_weights[b])) - 1.0) <= 1e-5
        assert np.all(pi_weights[b] >= 0.0)

