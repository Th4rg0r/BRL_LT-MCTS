"""Verification Gate 4: End-to-End Integration, Loss Convergence, and Benchmark Pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from config import load_config
from experiments.train_ant_brl import load_training_checkpoint, main as run_train_main
from la_mcts.backbone import SelfAttentionBackbone
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer


def test_gate4_representation_consistency_convergence() -> None:
    """Verifies that K-step unrolled training drives SimSiam representation loss toward 0."""
    torch.manual_seed(42)
    B, K, obs_dim, act_dim, latent_dim = 8, 5, 27, 8, 256
    K_act = 16

    backbone = SelfAttentionBackbone(obs_dim=obs_dim, latent_dim=latent_dim)
    actor_head = MultiPeakGMMActorHead(latent_dim=latent_dim, action_dim=act_dim)
    critic_head = SupportCriticHead(latent_dim=latent_dim, use_categorical=True)
    rep_trainer = ModelRepresentationTrainer(latent_dim=latent_dim, action_dim=act_dim)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters())
        + list(actor_head.parameters())
        + list(critic_head.parameters())
        + list(rep_trainer.parameters()),
        lr=1e-3,
    )

    obs_seq = torch.randn(B, K + 1, obs_dim)
    act_seq = torch.tanh(torch.randn(B, K, act_dim))
    rew_seq = torch.randn(B, K)
    val_seq = torch.randn(B, K + 1) * 5.0
    policy_samples = torch.tanh(torch.randn(B, K + 1, K_act, act_dim))
    policy_weights = torch.softmax(torch.randn(B, K + 1, K_act), dim=-1)

    initial_rep_loss = None
    final_rep_loss = None

    for step in range(25):
        optimizer.zero_grad()
        total_loss, metrics = rep_trainer.compute_unrolled_loss(
            obs_seq=obs_seq,
            action_seq=act_seq,
            reward_seq=rew_seq,
            target_value_seq=val_seq,
            mcts_policy_samples=policy_samples,
            mcts_policy_weights=policy_weights,
            backbone=backbone,
            actor_head=actor_head,
            critic_head=critic_head,
            lambda_rep=4.0,
        )
        total_loss.backward()
        optimizer.step()
        if step == 0:
            initial_rep_loss = metrics["loss_rep"]
        final_rep_loss = metrics["loss_rep"]

    assert initial_rep_loss is not None and final_rep_loss is not None
    assert final_rep_loss < initial_rep_loss * 0.5, (
        f"Representation loss did not converge: {initial_rep_loss:.4f} -> {final_rep_loss:.4f}"
    )


def test_gate4_end_to_end_multi_seed_pipeline(tmp_path) -> None:
    """Runs the full 3-seed end-to-end training & OOD benchmark suite verification."""
    cfg_Check = load_config("config/default_config.yaml")
    assert cfg_Check.la_mcts.num_peaks == 4

    out_dir = tmp_path / "gate4_outputs"
    results = run_train_main(
        [
            "--config",
            "config/default_config.yaml",
            "--seeds",
            "3",
            "--episodes",
            "4",
            "--max-steps",
            "15",
            "--n-sim",
            "10",
            "--checkpoint-interval",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert len(results["seeds"]) == 3
    assert results["best_overall_checkpoint_path"] is not None
    assert results["final_overall_checkpoint_path"] is not None
    assert Path(results["best_overall_checkpoint_path"]).exists()
    assert Path(results["final_overall_checkpoint_path"]).exists()

    for seed_res in results["seeds"]:
        ema_w = seed_res["final_ema_win_rate"]
        assert 0.40 <= ema_w <= 0.60, f"Balancer EMA win rate {ema_w} drifted out of bounds"
        final_bench = seed_res["benchmarks"][4]
        assert ("Standard-Ant-v5" in final_bench or "Standard-Ant-v4" in final_bench)
        assert "Tripod-Ant" in final_bench
        assert "Spider-Ant" in final_bench
        assert "Heavy-Ice-Ant" in final_bench
        assert "Hexapod-Ant" in final_bench
        assert Path(seed_res["best_checkpoint_path"]).exists()
        assert Path(seed_res["final_checkpoint_path"]).exists()
        assert len(seed_res["periodic_checkpoint_paths"]) == 1
        assert Path(seed_res["periodic_checkpoint_paths"][0]).exists()

    # Verify loading the saved checkpoint restores Agent, Tweaker, and Balancer cleanly
    loaded_agent, loaded_twk, loaded_bal, loaded_cfg, payload = load_training_checkpoint(
        results["final_overall_checkpoint_path"], device="cpu"
    )
    assert payload["episode"] == 4
    act = loaded_agent.select_action(np.zeros(loaded_cfg.la_mcts.obs_dim, dtype=np.float32), deterministic=True)
    assert act.shape == (loaded_cfg.ant_morph.max_action_capacity,)


def test_train_resume_and_extension_step_tracking(tmp_path) -> None:
    """Tests crash recovery, cumulative step tracking, and episode extension (e.g. 2 -> 4 episodes)."""
    import json
    out_dir = tmp_path / "resume_test"

    # Phase 1: Run 2 episodes
    r1 = run_train_main(
        [
            "--config", "config/default_config.yaml",
            "--seeds", "1",
            "--episodes", "2",
            "--max-steps", "10",
            "--n-sim", "6",
            "--output-dir", str(out_dir),
        ]
    )
    seed_res1 = r1["seeds"][0]
    assert seed_res1["completed_episodes"] == 2
    assert len(seed_res1["episodes"]) == 2
    steps_phase1 = seed_res1["total_env_steps"]
    assert steps_phase1 > 0

    prog_file = out_dir / "checkpoints" / "seed_42" / "progress.json"
    latest_ckpt = out_dir / "checkpoints" / "seed_42" / "latest_checkpoint.pt"
    assert prog_file.exists()
    assert latest_ckpt.exists()

    with prog_file.open("r", encoding="utf-8") as f:
        meta1 = json.load(f)
    assert meta1["completed_episodes"] == 2
    assert meta1["total_env_steps"] == steps_phase1
    assert meta1["resume_count"] == 0
    assert not meta1["is_resumed"]

    # Phase 2: Resume and extend from 2 to 4 episodes
    r2 = run_train_main(
        [
            "--config", "config/default_config.yaml",
            "--seeds", "1",
            "--episodes", "4",
            "--max-steps", "10",
            "--n-sim", "6",
            "--resume",
            "--output-dir", str(out_dir),
        ]
    )
    seed_res2 = r2["seeds"][0]
    assert seed_res2["completed_episodes"] == 4
    assert seed_res2["is_resumed"]
    assert seed_res2["resume_count"] == 1
    # Only 2 additional episodes were executed in this session!
    assert len(seed_res2["episodes"]) == 2
    assert seed_res2["episodes"][0]["episode"] == 3
    assert seed_res2["episodes"][1]["episode"] == 4
    # Cumulative steps continued growing
    assert seed_res2["total_env_steps"] > steps_phase1

    with prog_file.open("r", encoding="utf-8") as f:
        meta2 = json.load(f)
    assert meta2["completed_episodes"] == 4
    assert meta2["target_episodes"] == 4
    assert meta2["remaining_episodes"] == 0
    assert meta2["resume_count"] == 1
    assert meta2["is_resumed"]

    # Phase 3: Resume again when already completed 4 episodes -> should skip cleanly
    r3 = run_train_main(
        [
            "--config", "config/default_config.yaml",
            "--seeds", "1",
            "--episodes", "4",
            "--max-steps", "10",
            "--n-sim", "6",
            "--resume",
            "--output-dir", str(out_dir),
        ]
    )
    seed_res3 = r3["seeds"][0]
    assert seed_res3.get("skipped", False)
    assert len(seed_res3["episodes"]) == 0
    assert seed_res3["completed_episodes"] == 4


def test_step_milestones_and_eval(tmp_path) -> None:
    """Verifies that crossing cumulative step thresholds triggers dedicated evaluation and checkpoints."""
    out_dir = tmp_path / "milestone_test"
    r = run_train_main(
        [
            "--config", "config/default_config.yaml",
            "--seeds", "1",
            "--episodes", "3",
            "--max-steps", "15",
            "--n-sim", "6",
            "--step-milestones", "10,25",
            "--output-dir", str(out_dir),
        ]
    )
    seed_res = r["seeds"][0]
    ckpt_dir = out_dir / "checkpoints" / "seed_42"
    ms_ckpts = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    assert len(ms_ckpts) >= 1
    assert "10" in seed_res["milestone_checkpoint_paths"] or "25" in seed_res["milestone_checkpoint_paths"]


def test_parallel_multi_seed_logging(tmp_path) -> None:
    """Verifies that 3 parallel seeds log without crashing or garbling, and create dedicated log files."""
    import json
    out_dir = tmp_path / "parallel_log_test"
    r = run_train_main(
        [
            "--config", "config/default_config.yaml",
            "--seeds", "3",
            "--parallel-seeds", "3",
            "--episodes", "2",
            "--max-steps", "8",
            "--n-sim", "6",
            "--output-dir", str(out_dir),
        ]
    )
    assert len(r["seeds"]) == 3

    logs_dir = out_dir / "logs"
    assert logs_dir.exists()
    for s_val in (42, 43, 44):
        log_file = logs_dir / f"seed_{s_val}.log"
        jsonl_file = logs_dir / f"seed_{s_val}_progress.jsonl"
        assert log_file.exists(), f"Missing {log_file}"
        assert jsonl_file.exists(), f"Missing {jsonl_file}"
        assert log_file.stat().st_size > 0
        assert jsonl_file.stat().st_size > 0

    status_file = out_dir / "training_status.json"
    assert status_file.exists()
    with status_file.open("r", encoding="utf-8") as f:
        status_data = json.load(f)
    assert status_data["num_seeds"] == 3
    assert "42" in status_data["seeds"]
    assert "43" in status_data["seeds"]
    assert "44" in status_data["seeds"]
    for s_key in ("42", "43", "44"):
        assert status_data["seeds"][s_key]["completed_episodes"] == 2
        assert status_data["seeds"][s_key]["total_env_steps"] > 0


def test_agent_ema_target_networks() -> None:
    """Verifies LatentActionMCTSAgent EMA target networks, n-step TD bootstrap, and checkpoints."""
    from config import LAMCTSConfig
    from la_mcts.agent import EpisodeTrajectory, LatentActionMCTSAgent

    torch.manual_seed(7)
    np.random.seed(7)
    cfg = LAMCTSConfig(
        obs_dim=27,
        max_action_dim=8,
        latent_dim=64,
        d_model=64,
        n_heads=2,
        l_enc=1,
        l_dyn=1,
        proj_dim=64,
        pred_dim=32,
        unroll_steps=3,
        td_steps=2,
        gamma=0.9,
        target_tau=0.05,
        batch_size=4,
        n_sim=4,
        k_act=4,
        lr=1e-2,
    )
    agent = LatentActionMCTSAgent(config=cfg, device="cpu")

    # Target networks must be frozen (requires_grad=False, eval mode) and initially match online
    for mod in (agent.target_backbone, agent.target_critic_head, agent.target_projector):
        assert not mod.training
        assert all(not p.requires_grad for p in mod.parameters())
    for p_targ, p_on in zip(agent.target_backbone.parameters(), agent.backbone.parameters()):
        assert torch.allclose(p_targ, p_on)

    # Build a synthetic trajectory longer than td_steps + unroll_steps
    T = 8
    traj = EpisodeTrajectory(
        obs=[np.random.randn(27).astype(np.float32) for _ in range(T)],
        actions=[np.tanh(np.random.randn(8)).astype(np.float32) for _ in range(T)],
        rewards=[1.0 for _ in range(T)],
        mcts_values=[999.0 for _ in range(T)],  # Stale values intentionally extreme
        mcts_policy_samples=[np.zeros((cfg.k_act, 8), dtype=np.float32) for _ in range(T)],
        mcts_policy_weights=[np.full(cfg.k_act, 1.0 / cfg.k_act, dtype=np.float32) for _ in range(T)],
        context=np.zeros(cfg.context_dim, dtype=np.float32),
        action_mask=np.ones(8, dtype=np.float32),
    )
    agent.replay_buffer.add_trajectory(traj)

    # _compute_target_td_values should use target critic rather than stale mcts_values (999.0),
    # and handle terminal steps (step=7 -> boot_idx=9 >= T -> pure reward 1.0)
    td_vals = agent._compute_target_td_values([(traj, 0), (traj, 6)])
    assert td_vals.shape == (2, cfg.unroll_steps + 1)
    assert torch.isfinite(td_vals).all()
    assert float(td_vals[0, 0].abs().item()) < 50.0  # Not using stale 999.0
    # Sample 1 starts at 6 -> k=1 reaches step 7 (last step), boot_idx=9 >= T -> exact reward 1.0
    assert abs(float(td_vals[1, 1].item()) - 1.0) < 1e-5

    bb_init = [p.clone() for p in agent.backbone.parameters()]
    targ_init = [p.clone() for p in agent.target_backbone.parameters()]

    metrics = agent.train_step()
    assert "loss_total" in metrics
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.backbone.parameters())

    # Polyak EMA: target moved slowly toward updated online weights
    moved_online = any(
        not torch.allclose(p0, p1)
        for p0, p1 in zip(bb_init, agent.backbone.parameters())
    )
    moved_target = any(
        not torch.allclose(t0, t1)
        for t0, t1 in zip(targ_init, agent.target_backbone.parameters())
    )
    assert moved_online and moved_target
    for t0, t1, o1 in zip(targ_init, agent.target_backbone.parameters(), agent.backbone.parameters()):
        expected = (1.0 - cfg.target_tau) * t0 + cfg.target_tau * o1
        assert torch.allclose(t1, expected, atol=1e-6)

    # Checkpoint round-trip and backward compatibility when target keys are absent
    ckpt = agent.state_dict()
    assert {"target_backbone", "target_critic_head", "target_projector"}.issubset(ckpt.keys())

    agent2 = LatentActionMCTSAgent(config=cfg, device="cpu")
    agent2.load_state_dict(ckpt)
    for p1, p2 in zip(agent.target_backbone.parameters(), agent2.target_backbone.parameters()):
        assert torch.allclose(p1, p2)

    legacy_ckpt = {k: v for k, v in ckpt.items() if not k.startswith("target_")}
    agent3 = LatentActionMCTSAgent(config=cfg, device="cpu")
    agent3.load_state_dict(legacy_ckpt)
    for p_on, p_t in zip(agent3.backbone.parameters(), agent3.target_backbone.parameters()):
        assert torch.allclose(p_on, p_t)

