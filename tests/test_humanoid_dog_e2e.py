"""End-to-End Smoke and Resume Tests for Humanoid and Quadruped Dog Training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import pytest

from config import load_config
from experiments.train_humanoid_brl import run_single_seed as run_humanoid_seed
from experiments.train_dog_brl import run_single_seed as run_dog_seed


def test_humanoid_training_smoke_and_resume(tmp_path: Path) -> None:
    cfg = load_config("config/humanoid_optimal_config.yaml")
    # Scale down for fast CPU unit test
    cfg = replace(
        cfg,
        la_mcts=replace(
            cfg.la_mcts,
            n_sim=2,
            n_sim_eval=2,
            batch_size=4,
            k_act=2,
            d_model=32,
            latent_dim=32,
            proj_dim=64,
            pred_dim=32,
        ),
        brl=replace(
            cfg.brl,
            warmup_episodes=1,
            brl_episodes=2,
            total_episodes=3,
        ),
    )

    ckpt_dir = tmp_path / "humanoid_ckpts"
    log_file = tmp_path / "humanoid.log"

    # Run 2 episodes
    res1 = run_humanoid_seed(
        seed=101,
        cfg=cfg,
        episodes=2,
        max_steps=5,
        eval_interval=2,
        device="cpu",
        verbose=False,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1,
        resume=False,
        log_file=log_file,
    )
    assert res1["completed_episodes"] == 2
    assert (ckpt_dir / "seed_101" / "latest_checkpoint.pt").exists()
    assert (ckpt_dir / "seed_101" / "progress.json").exists()

    # Resume and run to 3 episodes
    res2 = run_humanoid_seed(
        seed=101,
        cfg=cfg,
        episodes=3,
        max_steps=5,
        eval_interval=3,
        device="cpu",
        verbose=False,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1,
        resume=True,
        log_file=log_file,
    )
    assert res2["is_resumed"] is True
    assert res2["completed_episodes"] == 3
    assert (ckpt_dir / "seed_101" / "final_model.pt").exists()


def test_dog_training_smoke_and_resume(tmp_path: Path) -> None:
    cfg = load_config("config/dog_optimal_config.yaml")
    # Scale down for fast CPU unit test
    cfg = replace(
        cfg,
        la_mcts=replace(
            cfg.la_mcts,
            n_sim=2,
            n_sim_eval=2,
            batch_size=4,
            k_act=2,
            d_model=32,
            latent_dim=32,
            proj_dim=64,
            pred_dim=32,
        ),
        brl=replace(
            cfg.brl,
            warmup_episodes=1,
            brl_episodes=2,
            total_episodes=3,
        ),
    )

    ckpt_dir = tmp_path / "dog_ckpts"
    log_file = tmp_path / "dog.log"

    # Run 2 episodes
    res1 = run_dog_seed(
        seed=202,
        cfg=cfg,
        episodes=2,
        max_steps=5,
        eval_interval=2,
        device="cpu",
        verbose=False,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1,
        resume=False,
        log_file=log_file,
    )
    assert res1["completed_episodes"] == 2
    assert (ckpt_dir / "seed_202" / "latest_checkpoint.pt").exists()
    assert (ckpt_dir / "seed_202" / "progress.json").exists()

    # Resume and run to 3 episodes
    res2 = run_dog_seed(
        seed=202,
        cfg=cfg,
        episodes=3,
        max_steps=5,
        eval_interval=3,
        device="cpu",
        verbose=False,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=1,
        resume=True,
        log_file=log_file,
    )
    assert res2["is_resumed"] is True
    assert res2["completed_episodes"] == 3
    assert (ckpt_dir / "seed_202" / "final_model.pt").exists()
