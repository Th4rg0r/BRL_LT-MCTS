"""Verification Gate 4: End-to-End Integration, Loss Convergence, and Benchmark Pipeline."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import numpy as np
import torch

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
    assert cfg_Check.la_mcts.num_peaks == 5

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
