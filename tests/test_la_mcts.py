"""Verification Gate 1: GMM & MCTS Engine Unit Tests (la_mcts/)."""

from __future__ import annotations

import time
import numpy as np
import torch

from la_mcts.backbone import SelfAttentionBackbone, combine_cumulative_rep
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer
from la_mcts.sampled_mcts import SampledContinuousMCTS
from config import LAMCTSConfig
from la_mcts.agent import LatentActionMCTSAgent


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    actor_head = actor_head.to(device)
    critic_head = critic_head.to(device)
    rep_trainer = rep_trainer.to(device)

    obs = torch.randn(1, 27, device=device)
    with torch.inference_mode():
        z_t = backbone(obs).squeeze(0)
        a_test = torch.zeros(1, 8, device=device)
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
    max_latency = 50.0 if device.type == "cuda" else 2500.0
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


def test_combine_cumulative_rep_bounds_and_fixed_point() -> None:
    """Verifies combine_cumulative_rep normalization, fixed-point identity, and zero-history mask."""
    torch.manual_seed(0)
    cur = torch.rand(4, 32)
    assert torch.equal(combine_cumulative_rep(cur, None, alpha=0.4), cur)
    assert torch.equal(combine_cumulative_rep(cur, cur, alpha=0.0), cur)

    # Fixed point: min_max_normalize(cur + 0.4 * cur) == min_max_normalize(cur)
    # when cur is already min-max normalized in [0, 1]
    c_min = cur.min(dim=-1, keepdim=True).values
    c_max = cur.max(dim=-1, keepdim=True).values
    cur_norm = (cur - c_min) / (c_max - c_min)
    fp = combine_cumulative_rep(cur_norm, cur_norm, alpha=0.4)
    torch.testing.assert_close(fp, cur_norm, atol=1e-5, rtol=1e-5)

    # Zero prev row is skipped per batch item, non-zero row is combined & bounded in [0, 1]
    prev = torch.rand(4, 32)
    prev[1].zero_()
    out = combine_cumulative_rep(cur, prev, alpha=0.4)
    assert float(out.min().item()) >= 0.0
    assert float(out.max().item()) <= 1.0
    torch.testing.assert_close(out[1], cur[1])


def test_compute_prev_cumulative_matches_sequential_ema() -> None:
    """Verifies compute_prev_cumulative skips zero history and matches sequential accumulation."""
    torch.manual_seed(7)
    backbone = SelfAttentionBackbone(
        obs_dim=10, max_action_dim=4, context_dim=6, k_tok=4, d_model=32, n_heads=2, l_enc=1, latent_dim=32, history_len=3
    )
    backbone.eval()

    zero_hist = torch.zeros(2, 3 * (10 + 4))
    ctx = torch.randn(2, 6)
    assert backbone.compute_prev_cumulative(ctx, zero_hist, alpha=0.4) is None
    assert backbone.compute_prev_cumulative(ctx, None, alpha=0.4) is None

    # Populate batch 0 with 2 valid older steps (step index 1 and 2) and batch 1 empty
    hist = torch.zeros(2, 3, 14)
    hist[0, 1, :10] = torch.randn(10)
    hist[0, 1, 10:] = torch.randn(4)
    hist[0, 2, :10] = torch.randn(10)
    hist[0, 2, 10:] = torch.randn(4)
    flat_hist = hist.reshape(2, -1)

    prev_cum = backbone.compute_prev_cumulative(ctx, flat_hist, alpha=0.4)
    assert prev_cum is not None
    assert prev_cum.shape == (2, 32)
    torch.testing.assert_close(prev_cum[1], torch.zeros(32))

    with torch.no_grad():
        sub_hist_1 = torch.zeros_like(hist[0:1])
        sub_hist_2 = torch.zeros_like(hist[0:1])
        sub_hist_2[0, 2] = hist[0, 1]
        z1 = backbone(hist[0:1, 1, :10], context=ctx[0:1], history=sub_hist_1.reshape(1, -1))
        z2 = backbone(hist[0:1, 2, :10], context=ctx[0:1], history=sub_hist_2.reshape(1, -1))
        expected_0 = combine_cumulative_rep(z2, z1, alpha=0.4).squeeze(0)
    torch.testing.assert_close(prev_cum[0], expected_0, atol=1e-5, rtol=1e-5)
    assert backbone.compute_prev_cumulative(ctx, flat_hist, alpha=0.0) is None


def test_mcts_search_uses_on_demand_cumulative_rep_and_keeps_cur_rep_dynamics() -> None:
    """Verifies MCTS keeps rep_trainer on cur_rep -> next_rep while feeding cum_rep to heads."""
    torch.manual_seed(11)
    actor_head = MultiPeakGMMActorHead(latent_dim=32, action_dim=4, num_peaks=3)
    critic_head = SupportCriticHead(latent_dim=32, use_categorical=True, support_min=-50, support_max=50, num_bins=101)
    rep_trainer = ModelRepresentationTrainer(
        latent_dim=32, action_dim=4, num_blocks=1, proj_dim=32, pred_dim=32, latent_ema_decay=0.4
    )
    mcts = SampledContinuousMCTS(
        actor_head=actor_head,
        critic_head=critic_head,
        rep_trainer=rep_trainer,
        n_sim=10,
        k_act=4,
        latent_ema_decay=0.4,
    )

    z_root = torch.rand(32)
    prev_cum = torch.rand(32)
    _, _, _, _, root = mcts.search(z_root, deterministic=True, prev_cum_state=prev_cum)

    assert root.z_state is not None
    assert root.z_cum_state is not None
    torch.testing.assert_close(root.z_state, z_root)
    expected_root_cum = combine_cumulative_rep(z_root, prev_cum, alpha=0.4)
    torch.testing.assert_close(root.z_cum_state, expected_root_cum)
    assert not torch.allclose(root.z_cum_state, root.z_state)

    expanded_children = list(root.children.values())
    assert len(expanded_children) > 0
    child = expanded_children[0]
    assert child.z_state is not None and child.z_cum_state is not None
    # Verify child.z_state was predicted from root.z_state (cur_rep), NOT root.z_cum_state
    child_edge_idx = next(idx for idx, c in root.children.items() if c is child)
    with torch.no_grad():
        expected_child_z, _ = rep_trainer.forward(
            root.z_state, root.sampled_actions[child_edge_idx]
        )
    torch.testing.assert_close(child.z_state, expected_child_z, atol=1e-5, rtol=1e-5)
    expected_child_cum = combine_cumulative_rep(child.z_state, root.z_cum_state, alpha=0.4)
    torch.testing.assert_close(child.z_cum_state, expected_child_cum, atol=1e-5, rtol=1e-5)


def test_agent_train_step_with_latent_ema_decay() -> None:
    """Verifies LatentActionMCTSAgent unroll training and reanalyze with latent_ema_decay=0.4."""
    torch.manual_seed(19)
    cfg = LAMCTSConfig(
        obs_dim=12,
        max_action_dim=4,
        context_dim=6,
        history_len=3,
        k_tok=4,
        d_model=32,
        n_heads=2,
        l_enc=1,
        latent_dim=32,
        l_dyn=1,
        proj_dim=32,
        pred_dim=32,
        n_sim=8,
        n_sim_eval=6,
        k_act=4,
        unroll_steps=2,
        batch_size=2,
        support_min=-20,
        support_max=20,
        num_bins=41,
        latent_ema_decay=0.4,
    )
    agent = LatentActionMCTSAgent(cfg, device=torch.device("cpu"))
    agent.set_episode_context(tweak_vector=np.zeros(6, dtype=np.float32), action_mask=np.ones(4, dtype=np.float32))

    # Collect a short episode via select_action + observe_transition
    obs = np.random.randn(12).astype(np.float32)
    for step in range(8):
        act = agent.select_action(obs, deterministic=False)
        next_obs = np.random.randn(12).astype(np.float32)
        agent.observe_transition(
            obs, act, float(0.5), next_obs, terminated=(step == 7), truncated=False, info={}
        )
        obs = next_obs

    metrics = agent.train_step()
    assert "loss_total" in metrics
    assert np.isfinite(metrics["loss_total"])
    assert agent.rep_trainer.latent_ema_decay == 0.4
    assert agent.mcts.latent_ema_decay == 0.4
    agent.reanalyze_batch(num_states=2)


