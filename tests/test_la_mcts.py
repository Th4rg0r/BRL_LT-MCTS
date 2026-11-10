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
    assert median_ms < 15.0, f"MCTS latency {median_ms:.2f} ms exceeded 15 ms limit"
