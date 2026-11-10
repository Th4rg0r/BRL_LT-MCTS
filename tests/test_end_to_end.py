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
        assert "Standard-Ant-v4" in final_bench
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
