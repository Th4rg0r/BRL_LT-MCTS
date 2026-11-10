"""Unit and End-to-End Tests for the Multi-Fidelity Hyperparameter Optimizer."""

from __future__ import annotations

from pathlib import Path
import optuna
import torch

from config import SystemConfig, load_config, save_config
from experiments.hpo.importance import (
    compute_blended_importances,
    partition_parameters_by_importance,
)
from experiments.hpo.search_space import (
    HYPERPARAM_SPECS,
    apply_params_to_config,
    compute_narrowed_bounds,
    default_param_dict,
    estimate_static_model_stats,
    sample_trial_params,
)
from experiments.optimize_hyperparams import main as run_hpo_main
from experiments.train_ant_brl import main as run_train_main
from la_mcts.agent import LatentActionMCTSAgent


def test_hpo_search_space_and_static_estimator() -> None:
    """Verifies that sampled configurations are valid, build models cleanly, and respect constraints."""
    base_cfg = SystemConfig()
    defaults = default_param_dict(base_cfg)
    assert len(defaults) == len(HYPERPARAM_SPECS)

    study = optuna.create_study(direction="maximize")
    all_names = [s.name for s in HYPERPARAM_SPECS]

    sampled_dicts = []
    for i in range(10):
        trial = study.ask()
        params = sample_trial_params(
            trial=trial,
            active_names=all_names,
            frozen_params=defaults,
        )
        cfg = apply_params_to_config(base_cfg, params)
        assert cfg.la_mcts.d_model == cfg.la_mcts.latent_dim
        assert cfg.la_mcts.d_model % cfg.la_mcts.n_heads == 0

        stats = estimate_static_model_stats(cfg)
        assert stats["param_count"] > 50_000
        assert stats["param_count"] < 25_000_000
        assert stats["est_vram_mb"] < 6144.0

        # Instantiate agent on first two samples to verify layer compatibility
        if i < 2:
            agent = LatentActionMCTSAgent(config=cfg.la_mcts, device="cpu")
            z = agent.backbone(torch.randn(2, cfg.la_mcts.obs_dim))
            assert z.shape == (2, cfg.la_mcts.latent_dim)

        study.tell(trial, float(i))
        sampled_dicts.append(params)

    narrowed = compute_narrowed_bounds(sampled_dicts[:4], shrink_factor=0.5)
    assert "lr" in narrowed
    assert narrowed["lr"][0] >= 5e-5
    assert narrowed["lr"][1] <= 2e-3


def test_hpo_importance_blending_and_partitioning() -> None:
    """Verifies PED-ANOVA + fANOVA + Prior blending and Stage 2/3 parameter partitioning."""
    base_cfg = SystemConfig()
    defaults = default_param_dict(base_cfg)
    all_names = [s.name for s in HYPERPARAM_SPECS]

    study = optuna.create_study(direction="maximize")
    for i in range(12):
        trial = study.ask()
        params = sample_trial_params(trial, all_names, defaults)
        # Synthesize objective strongly correlated with lr and train_every_steps
        score = -abs(params["lr"] - 3e-4) * 1e4 - abs(params["train_every_steps"] - 20)
        study.tell(trial, float(score))

    importances = compute_blended_importances(study, prior_blend=0.50)
    assert len(importances) == len(HYPERPARAM_SPECS)
    total_blend = sum(r.blended_score for r in importances)
    assert abs(total_blend - 1.0) < 1e-5

    focused, cma_names, minor_names = partition_parameters_by_importance(
        importances, max_focused_params=14, max_cma_params=8
    )
    assert "lr" in focused
    assert "train_every_steps" in focused
    assert len(cma_names) <= 8
    assert len(minor_names) >= 1


def test_hpo_config_roundtrip_and_end_to_end_smoke(tmp_path: Path) -> None:
    """Runs the 5-stage HPO pipeline in --smoke mode, verifies --resume, and runs train_ant_brl."""
    out_dir = tmp_path / "hpo_run"
    opt_cfg_path = tmp_path / "optimal_config.yaml"

    res = run_hpo_main(
        [
            "--base-config",
            "config/default_config.yaml",
            "--output-config",
            str(opt_cfg_path),
            "--output-dir",
            str(out_dir),
            "--budget-hours",
            "0.05",
            "--device",
            "cpu",
            "--final-seeds",
            "10",
            "--final-max-steps",
            "1000",
            "--final-hours",
            "24.0",
            "--smoke",
        ]
    )

    assert opt_cfg_path.exists()
    assert Path(res["report_md_path"]).exists()
    assert Path(res["summary_json_path"]).exists()

    # Verify saved optimal_config.yaml loads cleanly and contains the full Ant-v4 reference protocol
    # (3,000 episodes x 1,000 steps = 3,000,000 steps/seed across 10 seeds = 30,000,000 steps total)
    loaded_opt = load_config(opt_cfg_path)
    assert loaded_opt.run.seeds == 10
    assert loaded_opt.run.max_steps == 1000
    assert loaded_opt.run.episodes == 3000
    assert loaded_opt.run.episodes * loaded_opt.run.max_steps == 3_000_000
    assert loaded_opt.brl.total_episodes == 3000
    assert loaded_opt.brl.warmup_episodes == 50
    assert loaded_opt.brl.brl_episodes == 2500
    assert loaded_opt.ant_morph.max_episode_steps == 1000
    assert loaded_opt.ant_morph.env_id == "Ant-v5"
    assert loaded_opt.la_mcts.train_every_steps >= 5

    # Verify --resume reuses completed trials from JournalFileStorage without error
    res_resumed = run_hpo_main(
        [
            "--base-config",
            "config/default_config.yaml",
            "--output-config",
            str(opt_cfg_path),
            "--output-dir",
            str(out_dir),
            "--budget-hours",
            "0.05",
            "--device",
            "cpu",
            "--resume",
            "--smoke",
        ]
    )
    assert Path(res_resumed["optimal_config_path"]).exists()

    # Verify train_ant_brl.py runs with the generated optimal_config.yaml
    train_out = tmp_path / "train_verify"
    train_res = run_train_main(
        [
            "--config",
            str(opt_cfg_path),
            "--seeds",
            "1",
            "--episodes",
            "2",
            "--max-steps",
            "12",
            "--n-sim",
            "8",
            "--device",
            "cpu",
            "--output-dir",
            str(train_out),
        ]
    )
    assert len(train_res["seeds"]) == 1
