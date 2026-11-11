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
from experiments.optimize_hyperparams import (
    auto_detect_resume_stage,
    cleanup_interrupted_running_trials,
    load_baseline_calibration,
    main as run_hpo_main,
    parse_stage_arg,
    save_baseline_calibration,
)
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


def test_hpo_progress_monitor_and_diagnostics(capsys) -> None:
    """Verifies that HPOProgressMonitor formats and prints heartbeats, trial stats, and dashboards."""
    from experiments.hpo import (
        BaselineStats,
        HPOProgressMonitor,
        TrialEvalResult,
        ascii_progress_bar,
        format_time_duration,
    )

    assert format_time_duration(45.0) == "45s"
    assert format_time_duration(125.0) == "2m 05s"
    assert format_time_duration(3665.0) == "1h 01m"

    bar = ascii_progress_bar(0.5, width=10)
    assert len(bar) == 10
    assert "█" in bar and "░" in bar

    mon = HPOProgressMonitor(total_budget_sec=100.0, total_stages=5, debug=True, quiet=False)
    b_stats = BaselineStats(
        raw_mean=50.0,
        raw_std=5.0,
        lcb_score=47.5,
        sec_per_step=0.015,
        param_count=100_000,
        peak_vram_mb=500.0,
        standard_ant_mean=60.0,
        ood_suite_mean=40.0,
    )
    mon.set_baseline(b_stats)
    mon.start_stage(
        stage_idx=1,
        stage_name="stage1_screen",
        target_trials=4,
        timeout_sec=20.0,
        episodes=10,
        max_steps=50,
        seeds_count=1,
    )

    # Test heartbeat output
    mon.record_trial_heartbeat(
        trial_num=1,
        ep=5,
        ep_total=10,
        seed_idx=1,
        total_seeds=1,
        intermediate_score=55.0,
        sec_per_step=0.015,
        peak_mb=512.0,
    )
    captured = capsys.readouterr()
    assert "[Trial # 1" in captured.out
    assert "Eval Score:  +55.00" in captured.out

    # Test trial result output and NEW BEST detection
    res = TrialEvalResult(
        raw_score_mean=80.0,
        raw_score_std=4.0,
        standard_ant_mean=85.0,
        ood_suite_mean=75.0,
        penalized_objective=78.0,
        cost_penalty=2.0,
        sec_per_step=0.016,
        total_env_steps=500,
        param_count=120_000,
        peak_gpu_mem_mb=520.0,
        total_wall_sec=10.0,
        ema_win_rate=0.75,
        feasible=True,
        pruned=False,
    )
    mon.record_trial_result(
        trial_num=1,
        eval_res=res,
        params={"lr": 3e-4},
        cfg_summary={"lr": 3e-4, "n_sim": 12},
    )
    captured = capsys.readouterr()
    assert "NEW BEST" in captured.out
    assert "stage1_screen" in captured.out
    assert "Obj =  +78.00" in captured.out

    # Test stage dashboard
    mon.print_stage_dashboard(
        [
            {
                "params": {"lr": 3e-4, "n_sim": 12},
                "score": 78.0,
                "standard_ant_mean": 85.0,
                "ood_suite_mean": 75.0,
            }
        ]
    )
    captured = capsys.readouterr()
    assert "STAGE 1 SUMMARY" in captured.out
    assert "Throughput" in captured.out

    # Test final head-to-head table
    winner = {
        "label": "Winner-Model",
        "lcb_score": 76.0,
        "raw_mean": 80.0,
        "standard_ant_mean": 85.0,
        "ood_suite_mean": 75.0,
        "sec_per_step": 0.016,
        "param_count": 120_000,
        "peak_gpu_mem_mb": 520.0,
    }
    baseline_s4 = {
        "label": "Baseline-Default",
        "lcb_score": 47.5,
        "raw_mean": 50.0,
        "standard_ant_mean": 60.0,
        "ood_suite_mean": 40.0,
        "sec_per_step": 0.015,
        "param_count": 100_000,
        "peak_gpu_mem_mb": 500.0,
    }
    mon.print_final_head_to_head(winner, baseline_s4)
    captured = capsys.readouterr()
    assert "FINAL OPTIMIZATION SUMMARY" in captured.out
    assert "Improvement (Delta)" in captured.out
    assert "+28.50" in captured.out


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
            "--debug",
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


def test_hpo_stage_parsing_and_autodetect(tmp_path: Path) -> None:
    """Verifies stage identifier parsing and auto-detection logic."""
    assert parse_stage_arg("0") == 0
    assert parse_stage_arg("calib") == 0
    assert parse_stage_arg("1") == 1
    assert parse_stage_arg("screen") == 1
    assert parse_stage_arg("2") == 2
    assert parse_stage_arg("focus") == 2
    assert parse_stage_arg("3") == 3
    assert parse_stage_arg("finetune") == 3
    assert parse_stage_arg("4") == 4
    assert parse_stage_arg("select") == 4
    assert parse_stage_arg("auto") is None
    assert parse_stage_arg(None) is None

    import pytest
    with pytest.raises(ValueError, match="Unknown stage"):
        parse_stage_arg("invalid_stage_99")

    # Test auto-detect with empty dir
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert auto_detect_resume_stage(None, empty_dir) == 0


def test_hpo_baseline_calibration_and_stale_trial_cleanup(tmp_path: Path) -> None:
    """Verifies baseline calibration save/load and stale trial cleanup."""
    from experiments.hpo import BaselineStats
    from optuna.trial import TrialState
    from optuna.storages import InMemoryStorage

    calib_file = tmp_path / "baseline_calibration.json"
    stats = BaselineStats(
        raw_mean=12.5,
        raw_std=3.5,
        lcb_score=10.75,
        sec_per_step=0.012,
        param_count=250_000,
        peak_vram_mb=400.0,
        standard_ant_mean=15.0,
        ood_suite_mean=10.0,
    )
    save_baseline_calibration(calib_file, stats)
    loaded = load_baseline_calibration(calib_file)
    assert loaded is not None
    assert loaded.raw_mean == 12.5
    assert loaded.sec_per_step == 0.012

    # Verify cleanup of RUNNING trials
    storage = InMemoryStorage()
    study = optuna.create_study(study_name="test_stale", storage=storage)
    trial_id = storage.create_new_trial(study._study_id)
    # Trial starts in RUNNING state
    assert study.trials[0].state == TrialState.RUNNING
    cleaned = cleanup_interrupted_running_trials(study)
    assert cleaned == 1
    assert study.trials[0].state == TrialState.FAIL


def test_hpo_resume_from_start_stage(tmp_path: Path) -> None:
    """Verifies resuming directly from Stage 4 with --start-stage 4 reusing previous stage results."""
    out_dir = tmp_path / "hpo_stage_test"
    out_dir.mkdir()
    opt_cfg_path = out_dir / "optimal_config.yaml"

    # Pre-populate a fake Stage 2 study in optuna journal
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from experiments.hpo import BaselineStats

    journal_path = out_dir / "optuna_journal.log"
    lock_obj = JournalFileOpenLock(str(journal_path))
    storage = JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock_obj))

    # Pre-save baseline calibration
    calib_file = out_dir / "baseline_calibration.json"
    stats = BaselineStats(
        raw_mean=50.0,
        raw_std=5.0,
        lcb_score=47.5,
        sec_per_step=0.005,
        param_count=100_000,
        peak_vram_mb=400.0,
        standard_ant_mean=55.0,
        ood_suite_mean=45.0,
    )
    save_baseline_calibration(calib_file, stats)

    # Pre-populate stage2_focus study with completed trials
    study_s2 = optuna.create_study(study_name="stage2_focus", storage=storage, direction="maximize")
    trial = study_s2.ask()
    trial.set_user_attr("feasible", True)
    trial.set_user_attr("full_params", {"lr": 0.0003, "n_sim": 8})
    study_s2.tell(trial, 100.0)

    # Run directly starting from Stage 4
    res = run_hpo_main(
        [
            "--base-config",
            "config/default_config.yaml",
            "--output-config",
            str(opt_cfg_path),
            "--output-dir",
            str(out_dir),
            "--resume",
            "--start-stage",
            "4",
            "--smoke",
            "--quiet",
        ]
    )

    assert opt_cfg_path.exists()
    assert Path(res["optimal_config_path"]).exists()
    assert res["winner"]["label"] in ("Stage2-Best", "Baseline-Default")
