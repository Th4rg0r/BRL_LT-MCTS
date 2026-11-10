"""State-of-the-Art Multi-Fidelity, Importance-Weighted Hyperparameter Optimizer.

Executes a 5-stage curriculum of hyperparameter optimization tailored for LA-MCTS + BRL:
  - Stage 0 (Calibration): Measures baseline speed, seed noise sigma_seed, and reference score.
  - Stage 1 (Broad Screening, ~25% budget): Many fast trials (Sobol QMC + Multivariate TPE + ASHA).
  - Importance Analysis: Blends Prior Tier weights with PED-ANOVA + fANOVA empirical importances,
    promoting impactful parameters, freezing flat ones, and narrowing search ranges.
  - Stage 2 (Focused BO, ~30% budget): Medium-fidelity 2-seed trials on important parameters.
  - Stage 3 (Gradual Fine-Tuning, ~30% budget): High-fidelity 2-seed trials split between
    warm-started CMA-ES on top continuous parameters (75%) and TPE refinement of minor Tier-C
    parameters (25%).
  - Stage 4 (Robust Selection, ~13% budget): Evaluates finalists on 3 fresh holdout seeds and
    selects the winner by Lower Confidence Bound (mean - 0.5 * std) with cost tie-breaking.

Outputs `config/optimal_config.yaml` (including a calibrated 10-seed submission `run:` block)
and a detailed report in `outputs/hpo/<run_id>/`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import CmaEsSampler, QMCSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.*")
if hasattr(optuna, "exceptions") and hasattr(optuna.exceptions, "ExperimentalWarning"):
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from config import RunConfig, SystemConfig, load_config, save_config
from experiments.hpo.importance import (
    ImportanceResult,
    compute_blended_importances,
    partition_parameters_by_importance,
)
from experiments.hpo.search_space import (
    HYPERPARAM_SPECS,
    SPECS_BY_NAME,
    apply_params_to_config,
    compute_narrowed_bounds,
    default_param_dict,
    estimate_static_model_stats,
    sample_trial_params,
)
from experiments.hpo.trial_worker import (
    FidelityStage,
    TrialEvalResult,
    evaluate_config_at_fidelity,
)
from experiments.train_ant_brl import format_time_duration

optuna.logging.set_verbosity(optuna.logging.WARNING)


def detect_hardware_and_workers(
    requested_workers: str | int = "auto",
    max_gpu_mem_gb: float = 6.0,
) -> Tuple[int, int, float, float, float]:
    """Detects CPU cores, system RAM (GB), GPU VRAM (GB), and optimal parallel worker count.

    Returns:
        workers: Number of parallel trial workers.
        cpu_cores: Detected logical CPU count.
        ram_gb: Detected system RAM in GB.
        gpu_total_gb: Detected total GPU VRAM in GB (or 11.0 nominal if on CPU).
        gpu_mem_fraction_per_worker: Per-process CUDA memory cap fraction.
    """
    cpu_cores = os.cpu_count() or 4
    ram_gb = 16.0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    ram_gb = kb / (1024.0 * 1024.0)
                    break
        except Exception:
            pass

    gpu_total_gb = 11.0
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_total_gb = float(props.total_memory) / (1024.0**3)
        except Exception:
            pass

    if isinstance(requested_workers, str) and requested_workers.lower() == "auto":
        by_cpu = max(1, cpu_cores - 1)
        by_ram = max(1, int(ram_gb // 2.5))
        by_vram = max(1, int(max_gpu_mem_gb // 1.5))
        workers = max(1, min(by_cpu, by_ram, by_vram, 4))
    else:
        workers = max(1, int(requested_workers))

    per_worker_gb = max_gpu_mem_gb / float(max(1, workers))
    gpu_fraction = min(0.85, max(0.15, per_worker_gb / max(1.0, gpu_total_gb)))
    return workers, cpu_cores, ram_gb, gpu_total_gb, gpu_fraction


def build_fidelity_schedule(smoke: bool = False) -> Dict[str, FidelityStage]:
    """Returns the 5-stage multi-fidelity rung schedule."""
    if smoke:
        return {
            "stage0_calib": FidelityStage(
                name="stage0_calib",
                episodes=2,
                max_steps=12,
                eval_interval=2,
                seeds=(42, 43),
                budget_fraction=0.05,
            ),
            "stage1_screen": FidelityStage(
                name="stage1_screen",
                episodes=4,
                max_steps=12,
                eval_interval=2,
                seeds=(42,),
                budget_fraction=0.25,
            ),
            "stage2_focus": FidelityStage(
                name="stage2_focus",
                episodes=4,
                max_steps=15,
                eval_interval=2,
                seeds=(101, 102),
                budget_fraction=0.30,
            ),
            "stage3_finetune": FidelityStage(
                name="stage3_finetune",
                episodes=4,
                max_steps=15,
                eval_interval=2,
                seeds=(201, 202),
                budget_fraction=0.27,
            ),
            "stage4_select": FidelityStage(
                name="stage4_select",
                episodes=4,
                max_steps=15,
                eval_interval=2,
                seeds=(301, 302),
                budget_fraction=0.13,
            ),
        }

    return {
        "stage0_calib": FidelityStage(
            name="stage0_calib",
            episodes=40,
            max_steps=100,
            eval_interval=20,
            seeds=(42, 43),
            budget_fraction=0.02,
        ),
        "stage1_screen": FidelityStage(
            name="stage1_screen",
            episodes=40,
            max_steps=100,
            eval_interval=10,
            seeds=(42,),
            budget_fraction=0.25,
        ),
        "stage2_focus": FidelityStage(
            name="stage2_focus",
            episodes=100,
            max_steps=150,
            eval_interval=25,
            seeds=(101, 102),
            budget_fraction=0.30,
        ),
        "stage3_finetune": FidelityStage(
            name="stage3_finetune",
            episodes=300,
            max_steps=250,
            eval_interval=50,
            seeds=(201, 202),
            budget_fraction=0.30,
        ),
        "stage4_select": FidelityStage(
            name="stage4_select",
            episodes=300,
            max_steps=250,
            eval_interval=50,
            seeds=(301, 302, 303),
            budget_fraction=0.13,
        ),
    }


def extract_top_param_dicts(
    study: optuna.Study,
    base_defaults: Dict[str, Any],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Returns full hyperparameter dicts for the top-k completed feasible trials in `study`."""
    completed = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    feasible = [
        t for t in completed if bool(t.user_attrs.get("feasible", True))
    ]
    pool = feasible if feasible else completed
    pool.sort(key=lambda t: float(t.value), reverse=True)

    out: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for t in pool:
        full_p = dict(base_defaults)
        if "full_params" in t.user_attrs:
            full_p.update(t.user_attrs["full_params"])
        full_p.update(t.params)
        sig = json.dumps(full_p, sort_keys=True)
        if sig not in seen_keys:
            seen_keys.add(sig)
            out.append(full_p)
        if len(out) >= top_k:
            break

    if not out:
        out.append(dict(base_defaults))
    return out


def create_stage_study(
    study_name: str,
    storage: Optional[JournalStorage],
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
) -> optuna.Study:
    """Creates or loads a maximization Optuna study."""
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def run_stage_optimization(
    study: optuna.Study,
    stage: FidelityStage,
    base_cfg: SystemConfig,
    active_names: Sequence[str],
    frozen_params: Dict[str, Any],
    narrowed_bounds: Optional[Dict[str, Tuple[Any, Any]]],
    target_trials: int,
    stage_timeout_sec: float,
    workers: int,
    device: str,
    objective_mode: str,
    baseline_sec_per_step: float,
    baseline_seed_std: float,
    cost_penalty_lambda: float,
    max_gpu_mem_gb: float,
    gpu_mem_fraction_per_worker: float,
    cma_continuous_only: bool = False,
    qmc_warmup_trials: int = 0,
    verbose: bool = True,
) -> None:
    """Runs trials on `study` up to `target_trials` or `stage_timeout_sec`."""
    existing_done = len(
        [
            t
            for t in study.trials
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
        ]
    )
    remaining_trials = max(0, target_trials - existing_done)
    if remaining_trials == 0:
        if verbose:
            print(
                f"  [{stage.name}] Already has {existing_done}/{target_trials} finished trials (resumed).",
                flush=True,
            )
        return

    stage_start = time.time()

    def _objective(trial: optuna.Trial) -> float:
        params = sample_trial_params(
            trial=trial,
            active_names=active_names,
            frozen_params=frozen_params,
            narrowed_bounds=narrowed_bounds,
            cma_continuous_only=cma_continuous_only,
        )
        cfg = apply_params_to_config(base_cfg, params)

        def _prune_cb(ep: int, running_score: float) -> bool:
            trial.report(running_score, step=ep)
            return trial.should_prune()

        eval_res = evaluate_config_at_fidelity(
            cfg=cfg,
            stage=stage,
            device=device,
            objective_mode=objective_mode,
            baseline_sec_per_step=baseline_sec_per_step,
            baseline_seed_std=baseline_seed_std,
            cost_penalty_lambda=cost_penalty_lambda,
            max_gpu_mem_gb=max_gpu_mem_gb,
            gpu_mem_fraction_per_worker=gpu_mem_fraction_per_worker,
            pruning_callback=_prune_cb,
        )

        # Store rich diagnostics and constraint values on the trial
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("raw_score_mean", eval_res.raw_score_mean)
        trial.set_user_attr("raw_score_std", eval_res.raw_score_std)
        trial.set_user_attr("standard_ant_mean", eval_res.standard_ant_mean)
        trial.set_user_attr("ood_suite_mean", eval_res.ood_suite_mean)
        trial.set_user_attr("cost_penalty", eval_res.cost_penalty)
        trial.set_user_attr("sec_per_step", eval_res.sec_per_step)
        trial.set_user_attr("param_count", eval_res.param_count)
        trial.set_user_attr("peak_gpu_mem_mb", eval_res.peak_gpu_mem_mb)
        trial.set_user_attr("feasible", eval_res.feasible)
        trial.set_user_attr(
            "constraints",
            [0.0 if eval_res.feasible else 1.0],
        )

        if verbose:
            elapsed_stage = format_time_duration(time.time() - stage_start)
            status_tag = "OK" if eval_res.feasible else f"INFEASIBLE({eval_res.infeasibility_reason})"
            if eval_res.pruned and eval_res.feasible:
                status_tag = "PRUNED"
            print(
                f"  [{stage.name} | Trial #{trial.number:3d} | {elapsed_stage}] "
                f"Obj: {eval_res.penalized_objective:+7.2f} "
                f"(Raw: {eval_res.raw_score_mean:+7.2f}, Pen: -{eval_res.cost_penalty:5.2f}) | "
                f"n_sim={cfg.la_mcts.n_sim:2d}, d={cfg.la_mcts.d_model:3d}, "
                f"upd={cfg.la_mcts.train_every_steps:3d}, lr={cfg.la_mcts.lr:.1e} | "
                f"Params: {eval_res.param_count / 1e6:.2f}M | {status_tag}",
                flush=True,
            )

        if eval_res.pruned and eval_res.feasible:
            raise optuna.TrialPruned()

        return float(eval_res.penalized_objective)

    # Optional Phase 1a Sobol QMC space-filling warmup before switching to multivariate TPE
    if qmc_warmup_trials > 0 and existing_done < qmc_warmup_trials:
        qmc_to_run = min(qmc_warmup_trials - existing_done, remaining_trials)
        qmc_timeout = stage_timeout_sec * (qmc_to_run / float(max(1, remaining_trials)))
        study.sampler = QMCSampler(qmc_type="sobol", scramble=True, seed=42)
        study.optimize(
            _objective,
            n_trials=qmc_to_run,
            timeout=max(5.0, qmc_timeout),
            n_jobs=workers,
            gc_after_trial=True,
        )

    done_now = len(
        [
            t
            for t in study.trials
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
        ]
    )
    still_needed = max(0, target_trials - done_now)
    time_left = max(1.0, stage_timeout_sec - (time.time() - stage_start))

    if still_needed > 0 and time_left > 2.0:
        if qmc_warmup_trials > 0:
            study.sampler = TPESampler(
                multivariate=True,
                group=True,
                constant_liar=True,
                n_startup_trials=max(4, qmc_warmup_trials),
                constraints_func=lambda t: t.user_attrs.get("constraints", [0.0]),
                seed=42,
            )
        study.optimize(
            _objective,
            n_trials=still_needed,
            timeout=time_left,
            n_jobs=workers,
            gc_after_trial=True,
        )


def compute_final_run_protocol(
    winning_cfg: SystemConfig,
    measured_sec_per_step: float,
    baseline_sec_per_step: float,
    final_seeds: int = 10,
    final_episodes: int = 3000,
    final_max_steps: int = 1000,
    final_hours: float = 24.0,
    parallel_seeds: int = 3,
    device: str = "cuda",
    gpu_mem_fraction: float = 0.25,
) -> Tuple[RunConfig, float, int]:
    """Computes the optimal `RunConfig` for `optimal_config.yaml`.

    By default (`final_episodes=3000`, `final_max_steps=1000`), this sets the full
    Ant-v4 reference benchmark protocol from Section 4.2 of `task.md`:
    3,000 episodes x 1,000 steps = 3,000,000 environment steps per seed
    (30,000,000 steps across 10 seeds). If `final_episodes <= 0`, episodes are
    calibrated automatically from `final_hours`.
    """
    static_stats = estimate_static_model_stats(winning_cfg, max_steps=final_max_steps)
    rel_cost = float(static_stats["rel_cost_factor"])
    eff_sec_per_step = max(
        1e-4,
        0.5 * measured_sec_per_step + 0.5 * (baseline_sec_per_step * rel_cost),
    )

    # Number of sequential seed waves when running `parallel_seeds` concurrently
    waves = math.ceil(final_seeds / float(max(1, parallel_seeds)))
    # Multi-process memory bandwidth contention factor (~1.25x for 3 workers)
    contention = 1.0 + 0.12 * max(0, min(parallel_seeds, final_seeds) - 1)
    sec_budget_per_seed = (final_hours * 3600.0) / (float(max(1, waves)) * contention)

    # Account for periodic benchmark evaluations (5 creatures * final_max_steps every 10 episodes)
    steps_per_episode_incl_eval = float(final_max_steps) * (1.0 + 5.0 / 10.0)
    sec_per_episode = max(0.05, eff_sec_per_step * steps_per_episode_incl_eval)

    if final_episodes > 0:
        recommended_episodes = int(final_episodes)
    else:
        raw_episodes = int(round(sec_budget_per_seed / sec_per_episode))
        recommended_episodes = int(np.clip(raw_episodes, 200, 3000))

    est_total_hours = (
        waves * contention * recommended_episodes * sec_per_episode
    ) / 3600.0

    run_cfg = RunConfig(
        seeds=int(final_seeds),
        seed_offset=42,
        episodes=int(recommended_episodes),
        max_steps=int(final_max_steps),
        device=str(device),
        eval_interval=10,
        parallel_seeds=int(parallel_seeds),
        gpu_mem_fraction=float(round(gpu_mem_fraction, 3)),
    )
    return run_cfg, float(est_total_hours), int(recommended_episodes * final_max_steps)


def format_Config_header(
    winner_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    importances: Sequence[ImportanceResult],
    est_final_hours: float,
    env_steps_per_seed: int,
    run_cfg: RunConfig,
) -> str:
    """Builds the human-readable provenance comment header for `optimal_config.yaml`."""
    top_imp = ", ".join(
        f"{r.name} ({r.blended_score * 100:.1f}% [Tier {r.tier}])"
        for r in list(importances)[:8]
    )
    lines = [
        "=============================================================================",
        "OPTIMAL CONFIGURATION — Generated by experiments/optimize_hyperparams.py",
        "=============================================================================",
        f"Selection Criterion : Robust Multi-Seed LCB (mean - 0.5 * std) + Cost Parsimony",
        f"Winning LCB Score   : {winner_summary['lcb_score']:+.2f} "
        f"(Raw Mean: {winner_summary['raw_mean']:+.2f} +/- {winner_summary['raw_std']:.2f})",
        f"Baseline LCB Score  : {baseline_summary['lcb_score']:+.2f} "
        f"(Raw Mean: {baseline_summary['raw_mean']:+.2f} +/- {baseline_summary['raw_std']:.2f})",
        f"Standard-Ant Return : {winner_summary['standard_ant_mean']:+.2f} | "
        f"OOD Suite Mean: {winner_summary['ood_suite_mean']:+.2f}",
        f"Model Size & Cost   : {winner_summary['param_count']:,} params | "
        f"Peak VRAM ~ {winner_summary['peak_gpu_mem_mb'] / 1024.0:.2f} GB | "
        f"Cost Penalty: {winner_summary['cost_penalty']:.2f}",
        f"Top Wichtigste Params: {top_imp}",
        "-----------------------------------------------------------------------------",
        "Submission Run Protocol (calibrated for 4 CPUs, 16 GB RAM, 11 GB GPU):",
        f"  Seeds             : {run_cfg.seeds} (parallel_seeds={run_cfg.parallel_seeds})",
        f"  Episodes / Seed   : {run_cfg.episodes} x {run_cfg.max_steps} steps "
        f"({env_steps_per_seed:,} env steps/seed)",
        f"  Est. Total Runtime: ~{est_final_hours:.1f} hours on {run_cfg.device}",
        "  Execute with      : python -m experiments.train_ant_brl --config config/optimal_config.yaml",
        "=============================================================================",
    ]
    return "\n".join(lines)


def write_hpo_report(
    out_dir: Path,
    winner_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    stage4_leaderboard: Sequence[Dict[str, Any]],
    importances: Sequence[ImportanceResult],
    winning_cfg: SystemConfig,
    est_final_hours: float,
    total_hpo_sec: float,
) -> Tuple[Path, Path]:
    """Writes `hpo_report.md` and `hpo_summary.json` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / "hpo_report.md"
    summary_json = out_dir / "hpo_summary.json"

    imp_rows = "\n".join(
        f"| `{r.name}` | Tier {r.tier} | {r.prior_weight:.2f} | "
        f"{r.ped_anova_score * 100:.1f}% | {r.fanova_score * 100:.1f}% | "
        f"**{r.blended_score * 100:.1f}%** | {'Yes' if r.promoted_or_active else 'Frozen after S1'} |"
        for r in importances
    )

    lb_rows = "\n".join(
        f"| {idx + 1} | `{item['label']}` | **{item['lcb_score']:+.2f}** | "
        f"{item['raw_mean']:+.2f} ± {item['raw_std']:.2f} | "
        f"{item['standard_ant_mean']:+.2f} | {item['ood_suite_mean']:+.2f} | "
        f"{item['param_count'] / 1e6:.2f}M | {item['sec_per_step'] * 1000:.1f} ms |"
        for idx, item in enumerate(stage4_leaderboard)
    )

    md_text = f"""# Hyperparameter Optimization Report (`LA-MCTS + BRL`)

- **Total Optimization Wall Time**: `{format_time_duration(total_hpo_sec)}`
- **Winning Candidate**: `{winner_summary['label']}` (LCB = `{winner_summary['lcb_score']:+.2f}` vs. Baseline `{baseline_summary['lcb_score']:+.2f}`)
- **Model Footprint**: `{winner_summary['param_count']:,}` parameters, Peak VRAM `~{winner_summary['peak_gpu_mem_mb'] / 1024.0:.2f} GB`
- **Final 10-Seed Submission Run Estimate**: `~{est_final_hours:.1f} hours` (`{winning_cfg.run.episodes}` episodes × `{winning_cfg.run.max_steps}` steps × `{winning_cfg.run.seeds}` seeds, `{winning_cfg.run.parallel_seeds}` parallel workers)

---

## Stage 4 Multi-Seed Finalist Leaderboard

| Rank | Candidate | Robust LCB ($\\mu - 0.5\\sigma$) | Raw Composite ($\\mu \\pm \\sigma$) | Standard-Ant-v4 | OOD Suite Mean | Params | Step Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{lb_rows}

---

## Blended Hyperparameter Importance Ranking (Prior Tier $\\oplus$ PED-ANOVA $\\oplus$ fANOVA)

| Hyperparameter | Prior Tier | Prior Weight | PED-ANOVA (Top Quantile) | fANOVA (Global) | Blended Importance | Active in Stage 2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{imp_rows}
"""
    report_md.write_text(md_text, encoding="utf-8")

    payload = {
        "total_hpo_sec": total_hpo_sec,
        "est_final_hours": est_final_hours,
        "winner": winner_summary,
        "baseline": baseline_summary,
        "stage4_leaderboard": list(stage4_leaderboard),
        "importances": [
            {
                "name": r.name,
                "tier": r.tier,
                "prior_weight": r.prior_weight,
                "ped_anova_score": r.ped_anova_score,
                "fanova_score": r.fanova_score,
                "blended_score": r.blended_score,
                "promoted_or_active": r.promoted_or_active,
            }
            for r in importances
        ],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_md, summary_json


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Multi-Fidelity Importance-Weighted Hyperparameter Optimizer for LA-MCTS + BRL."
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="config/default_config.yaml",
        help="Path to base YAML configuration.",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Destination path for optimal YAML config (defaults to config/optimal_config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hpo",
        help="Directory for Optuna journal storage and HPO reports.",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=24.0,
        help="Target wall-clock budget in hours for the optimization run (default: 24.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default="auto",
        help="Number of concurrent trial workers ('auto' or integer).",
    )
    parser.add_argument(
        "--max-gpu-mem-gb",
        type=float,
        default=6.0,
        help="Hard ceiling on total GPU VRAM usage in GB across workers (default: 6.0 of 11 GB).",
    )
    parser.add_argument(
        "--cost-penalty-lambda",
        type=float,
        default=0.5,
        help="Soft penalty weight per doubling of step latency in units of seed std (default: 0.5).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=("mixed", "standard", "ood"),
        default="mixed",
        help="Optimization target: 'mixed' (0.5 Standard + 0.5 OOD), 'standard', or 'ood'.",
    )
    parser.add_argument(
        "--final-seeds",
        type=int,
        default=10,
        help="Number of seeds to configure in optimal_config.yaml for submission (default: 10).",
    )
    parser.add_argument(
        "--final-episodes",
        type=int,
        default=3000,
        help=(
            "Episodes per seed in optimal_config.yaml matching the Ant-v4 reference protocol "
            "(default: 3000 -> 3,000,000 steps/seed; pass 0 to auto-scale from --final-hours)."
        ),
    )
    parser.add_argument(
        "--final-max-steps",
        type=int,
        default=1000,
        help="Episode horizon to configure in optimal_config.yaml for submission (default: 1000).",
    )
    parser.add_argument(
        "--final-hours",
        type=float,
        default=24.0,
        help="Target wall-clock runtime in hours when --final-episodes=0 (default: 24.0).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Optuna JournalFileStorage in --output-dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast end-to-end smoke test of all 5 stages with tiny fidelities.",
    )
    args = parser.parse_args(argv)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested `--device cuda`, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch wheel on your GPU machine or pass `--device cpu`."
            )
        device = args.device
    else:
        device = "cpu"

    workers, cpu_cores, ram_gb, gpu_total_gb, gpu_fraction = detect_hardware_and_workers(
        requested_workers="1" if args.smoke else args.workers,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_config is not None:
        final_cfg_path = Path(args.output_config)
    elif args.smoke:
        final_cfg_path = out_dir / "optimal_config.smoke.yaml"
    else:
        final_cfg_path = Path("config/optimal_config.yaml")
    partial_cfg_path = out_dir / "optimal_config.partial.yaml"

    journal_path = out_dir / "optuna_journal.log"
    if not args.resume and journal_path.exists():
        journal_path.unlink()
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    base_cfg = load_config(args.base_config)
    base_defaults = default_param_dict(base_cfg)
    schedule = build_fidelity_schedule(smoke=args.smoke)
    total_budget_sec = max(20.0, float(args.budget_hours) * 3600.0)
    hpo_start_time = time.time()

    print("=" * 88)
    print("  LA-MCTS + BRL Multi-Fidelity Hyperparameter Optimizer")
    print(f"  Device & Hardware : {device} | {cpu_cores} CPUs | {ram_gb:.1f} GB RAM | GPU ~{gpu_total_gb:.1f} GB")
    print(f"  Parallel Workers  : {workers} (VRAM Cap: {args.max_gpu_mem_gb:.1f} GB total, {gpu_fraction * 100:.0f}%/worker)")
    print(f"  Time Budget       : {args.budget_hours:.2f} hours ({format_time_duration(total_budget_sec)})")
    print(f"  Objective Mode    : {args.objective} (Cost Penalty lambda={args.cost_penalty_lambda})")
    print(f"  Output Config     : {final_cfg_path}")
    print("=" * 88, flush=True)

    # =========================================================================
    # STAGE 0: Baseline Calibration
    # =========================================================================
    s0 = schedule["stage0_calib"]
    print(f"\n>>> [Stage 0: Baseline Calibration] ({s0.episodes} eps x {s0.max_steps} steps, {len(s0.seeds)} seeds)", flush=True)
    baseline_cfg = apply_params_to_config(base_cfg, base_defaults)
    calib_res = evaluate_config_at_fidelity(
        cfg=baseline_cfg,
        stage=s0,
        device=device,
        objective_mode=args.objective,
        baseline_sec_per_step=0.0,
        baseline_seed_std=10.0,
        cost_penalty_lambda=0.0,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_mem_fraction_per_worker=gpu_fraction,
    )
    baseline_sec_per_step = max(1e-4, calib_res.sec_per_step)
    baseline_seed_std = max(2.0, calib_res.raw_score_std if calib_res.raw_score_std > 0 else abs(calib_res.raw_score_mean) * 0.1 + 5.0)
    print(
        f"  [Stage 0 Done] Baseline Score: {calib_res.raw_score_mean:+.2f} (+/- {calib_res.raw_score_std:.2f}) | "
        f"Step Time: {baseline_sec_per_step * 1000:.2f} ms | Params: {calib_res.param_count:,}",
        flush=True,
    )

    # =========================================================================
    # STAGE 1: Broad Screening (Sobol QMC + Multivariate TPE + Hyperband ASHA)
    # =========================================================================
    s1 = schedule["stage1_screen"]
    s1_budget_sec = max(6.0, total_budget_sec * s1.budget_fraction)
    est_s1_trial_sec = max(0.2, baseline_sec_per_step * s1.episodes * s1.max_steps * len(s1.seeds))
    s1_target_trials = (
        6
        if args.smoke
        else int(np.clip((s1_budget_sec * workers * 1.4) / est_s1_trial_sec, 36, 360))
    )
    qmc_warmup = max(2, s1_target_trials // 3)
    print(
        f"\n>>> [Stage 1: Broad Screening] Target={s1_target_trials} trials "
        f"({s1.episodes} eps x {s1.max_steps} steps, budget={format_time_duration(s1_budget_sec)})",
        flush=True,
    )

    s1_sampler = TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=qmc_warmup,
        constraints_func=lambda t: t.user_attrs.get("constraints", [0.0]),
        seed=42,
    )
    s1_pruner = HyperbandPruner(
        min_resource=max(2, s1.eval_interval),
        max_resource=s1.episodes,
        reduction_factor=2,
    )
    study_s1 = create_stage_study("stage1_screen", storage, s1_sampler, s1_pruner)
    if len(study_s1.trials) == 0:
        study_s1.enqueue_trial(dict(base_defaults))

    all_param_names = [s.name for s in HYPERPARAM_SPECS]
    run_stage_optimization(
        study=study_s1,
        stage=s1,
        base_cfg=base_cfg,
        active_names=all_param_names,
        frozen_params=base_defaults,
        narrowed_bounds=None,
        target_trials=s1_target_trials,
        stage_timeout_sec=s1_budget_sec,
        workers=workers,
        device=device,
        objective_mode=args.objective,
        baseline_sec_per_step=baseline_sec_per_step,
        baseline_seed_std=baseline_seed_std,
        cost_penalty_lambda=args.cost_penalty_lambda,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_mem_fraction_per_worker=gpu_fraction,
        qmc_warmup_trials=qmc_warmup,
        verbose=True,
    )

    # =========================================================================
    # IMPORTANCE ANALYSIS & SEARCH SPACE NARROWING
    # =========================================================================
    importances = compute_blended_importances(study_s1, prior_blend=0.50)
    focused_names, cma_names, minor_names = partition_parameters_by_importance(
        importances, max_focused_params=16, max_cma_params=10
    )
    s1_top_dicts = extract_top_param_dicts(study_s1, base_defaults, top_k=6)
    s1_best_params = dict(s1_top_dicts[0])
    narrowed_s2 = compute_narrowed_bounds(s1_top_dicts, shrink_factor=0.60)

    print("\n>>> [Importance Analysis: Prior Tier + PED-ANOVA + fANOVA]", flush=True)
    for r in importances[:10]:
        print(
            f"    * {r.name:<20} (Tier {r.tier}): Blended={r.blended_score * 100:5.1f}% "
            f"[PED-ANOVA={r.ped_anova_score * 100:4.1f}%, fANOVA={r.fanova_score * 100:4.1f}%]",
            flush=True,
        )
    save_config(
        apply_params_to_config(base_cfg, s1_best_params),
        partial_cfg_path,
        header_comment="Intermediate Best Configuration after Stage 1 (Broad Screening)",
    )

    # =========================================================================
    # STAGE 2: Focused Bayesian Optimization on Important Parameters
    # =========================================================================
    s2 = schedule["stage2_focus"]
    s2_budget_sec = max(6.0, total_budget_sec * s2.budget_fraction)
    est_s2_trial_sec = max(0.3, baseline_sec_per_step * s2.episodes * s2.max_steps * len(s2.seeds))
    s2_target_trials = (
        4
        if args.smoke
        else int(np.clip((s2_budget_sec * workers * 1.3) / est_s2_trial_sec, 16, 90))
    )
    print(
        f"\n>>> [Stage 2: Focused BO] Target={s2_target_trials} trials on {len(focused_names)} key parameters "
        f"({s2.episodes} eps x {s2.max_steps} steps, {len(s2.seeds)} seeds)",
        flush=True,
    )

    s2_sampler = TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=max(3, min(8, s2_target_trials // 3)),
        constraints_func=lambda t: t.user_attrs.get("constraints", [0.0]),
        seed=102,
    )
    s2_pruner = HyperbandPruner(
        min_resource=max(2, s2.eval_interval),
        max_resource=s2.episodes,
        reduction_factor=2,
    )
    study_s2 = create_stage_study("stage2_focus", storage, s2_sampler, s2_pruner)
    if len(study_s2.trials) == 0:
        for top_d in s1_top_dicts[: min(3, s2_target_trials)]:
            study_s2.enqueue_trial({k: top_d[k] for k in focused_names if k in top_d})

    run_stage_optimization(
        study=study_s2,
        stage=s2,
        base_cfg=base_cfg,
        active_names=focused_names,
        frozen_params=s1_best_params,
        narrowed_bounds=narrowed_s2,
        target_trials=s2_target_trials,
        stage_timeout_sec=s2_budget_sec,
        workers=workers,
        device=device,
        objective_mode=args.objective,
        baseline_sec_per_step=baseline_sec_per_step,
        baseline_seed_std=baseline_seed_std,
        cost_penalty_lambda=args.cost_penalty_lambda,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_mem_fraction_per_worker=gpu_fraction,
        verbose=True,
    )

    s2_top_dicts = extract_top_param_dicts(study_s2, s1_best_params, top_k=5)
    s2_best_params = dict(s2_top_dicts[0])
    narrowed_s3 = compute_narrowed_bounds(s2_top_dicts, shrink_factor=0.40)
    save_config(
        apply_params_to_config(base_cfg, s2_best_params),
        partial_cfg_path,
        header_comment="Intermediate Best Configuration after Stage 2 (Focused BO)",
    )

    # =========================================================================
    # STAGE 3: High-Fidelity Gradual Fine-Tuning (3a: CMA-ES Core, 3b: Minor Params)
    # =========================================================================
    s3 = schedule["stage3_finetune"]
    s3_budget_sec = max(6.0, total_budget_sec * s3.budget_fraction)
    est_s3_trial_sec = max(0.4, baseline_sec_per_step * s3.episodes * s3.max_steps * len(s3.seeds))
    s3_total_trials = (
        4
        if args.smoke
        else int(np.clip((s3_budget_sec * workers * 1.2) / est_s3_trial_sec, 8, 28))
    )
    s3a_trials = max(2, int(round(0.70 * s3_total_trials)))
    s3b_trials = max(2, s3_total_trials - s3a_trials)

    print(
        f"\n>>> [Stage 3a: High-Fidelity Core Fine-Tuning] Target={s3a_trials} trials "
        f"({s3.episodes} eps x {s3.max_steps} steps, {len(s3.seeds)} seeds)",
        flush=True,
    )
    # Use warm-started CMA-ES when enough trials exist, else tight multivariate TPE
    if s3a_trials >= 6 and not args.smoke:
        s3a_sampler: optuna.samplers.BaseSampler = CmaEsSampler(
            n_startup_trials=2,
            warn_independent_sampling=False,
            seed=203,
        )
    else:
        s3a_sampler = TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=2,
            seed=203,
        )
    s3_pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=s3.eval_interval)
    study_s3a = create_stage_study("stage3a_cma_core", storage, s3a_sampler, s3_pruner)
    if len(study_s3a.trials) == 0:
        study_s3a.enqueue_trial({k: s2_best_params[k] for k in cma_names if k in s2_best_params})

    run_stage_optimization(
        study=study_s3a,
        stage=s3,
        base_cfg=base_cfg,
        active_names=cma_names,
        frozen_params=s2_best_params,
        narrowed_bounds=narrowed_s3,
        target_trials=s3a_trials,
        stage_timeout_sec=s3_budget_sec * 0.70,
        workers=workers,
        device=device,
        objective_mode=args.objective,
        baseline_sec_per_step=baseline_sec_per_step,
        baseline_seed_std=baseline_seed_std,
        cost_penalty_lambda=args.cost_penalty_lambda,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_mem_fraction_per_worker=gpu_fraction,
        cma_continuous_only=True,
        verbose=True,
    )

    s3a_top_dicts = extract_top_param_dicts(study_s3a, s2_best_params, top_k=3)
    s3a_best_params = dict(s3a_top_dicts[0])

    print(
        f"\n>>> [Stage 3b: Minor Parameter Refinement] Target={s3b_trials} trials on {len(minor_names)} secondary/Tier-C parameters",
        flush=True,
    )
    s3b_sampler = TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=1,
        seed=204,
    )
    study_s3b = create_stage_study("stage3b_minor_refine", storage, s3b_sampler, s3_pruner)
    if len(study_s3b.trials) == 0:
        study_s3b.enqueue_trial({k: s3a_best_params[k] for k in minor_names if k in s3a_best_params})

    run_stage_optimization(
        study=study_s3b,
        stage=s3,
        base_cfg=base_cfg,
        active_names=minor_names,
        frozen_params=s3a_best_params,
        narrowed_bounds=None,
        target_trials=s3b_trials,
        stage_timeout_sec=s3_budget_sec * 0.30,
        workers=workers,
        device=device,
        objective_mode=args.objective,
        baseline_sec_per_step=baseline_sec_per_step,
        baseline_seed_std=baseline_seed_std,
        cost_penalty_lambda=args.cost_penalty_lambda,
        max_gpu_mem_gb=args.max_gpu_mem_gb,
        gpu_mem_fraction_per_worker=gpu_fraction,
        verbose=True,
    )

    # =========================================================================
    # STAGE 4: Robust Multi-Seed Holdout Evaluation & Winner Selection
    # =========================================================================
    s4 = schedule["stage4_select"]
    print(
        f"\n>>> [Stage 4: Robust Multi-Seed Verification] ({s4.episodes} eps x {s4.max_steps} steps, "
        f"{len(s4.seeds)} fresh holdout seeds: {s4.seeds})",
        flush=True,
    )

    s3b_top_dicts = extract_top_param_dicts(study_s3b, s3a_best_params, top_k=2)
    candidate_pool: List[Tuple[str, Dict[str, Any]]] = [
        ("Stage3b-Best", s3b_top_dicts[0]),
        ("Stage3a-Best", s3a_best_params),
        ("Stage2-Best", s2_best_params),
    ]
    if not args.smoke and len(s2_top_dicts) > 1:
        candidate_pool.append(("Stage2-RunnerUp", s2_top_dicts[1]))
    candidate_pool.append(("Baseline-Default", dict(base_defaults)))

    # Deduplicate candidate parameter sets while keeping Baseline-Default explicitly
    unique_candidates: List[Tuple[str, Dict[str, Any]]] = []
    seen_cands: set[str] = set()
    for label, p_dict in candidate_pool:
        sig = json.dumps(p_dict, sort_keys=True)
        if sig not in seen_cands or label == "Baseline-Default":
            seen_cands.add(sig)
            unique_candidates.append((label, p_dict))

    leaderboard: List[Dict[str, Any]] = []
    baseline_s4_summary: Optional[Dict[str, Any]] = None

    for label, p_dict in unique_candidates:
        cand_cfg = apply_params_to_config(base_cfg, p_dict)
        res = evaluate_config_at_fidelity(
            cfg=cand_cfg,
            stage=s4,
            device=device,
            objective_mode=args.objective,
            baseline_sec_per_step=baseline_sec_per_step,
            baseline_seed_std=baseline_seed_std,
            cost_penalty_lambda=args.cost_penalty_lambda,
            max_gpu_mem_gb=args.max_gpu_mem_gb,
            gpu_mem_fraction_per_worker=gpu_fraction,
        )
        lcb_score = float(res.penalized_objective - 0.5 * res.raw_score_std)
        entry = {
            "label": label,
            "params": p_dict,
            "lcb_score": lcb_score,
            "penalized_objective": res.penalized_objective,
            "raw_mean": res.raw_score_mean,
            "raw_std": res.raw_score_std,
            "standard_ant_mean": res.standard_ant_mean,
            "ood_suite_mean": res.ood_suite_mean,
            "cost_penalty": res.cost_penalty,
            "sec_per_step": res.sec_per_step,
            "param_count": res.param_count,
            "peak_gpu_mem_mb": res.peak_gpu_mem_mb,
            "feasible": res.feasible,
        }
        leaderboard.append(entry)
        if label == "Baseline-Default":
            baseline_s4_summary = entry
        print(
            f"  * {label:<18}: LCB = {lcb_score:+7.2f} | Raw = {res.raw_score_mean:+7.2f} +/- {res.raw_score_std:5.2f} "
            f"| Params = {res.param_count / 1e6:.2f}M | Step = {res.sec_per_step * 1000:.1f} ms",
            flush=True,
        )

    feasible_lb = [e for e in leaderboard if e["feasible"]]
    ranked_lb = feasible_lb if feasible_lb else leaderboard
    ranked_lb.sort(
        key=lambda e: (e["lcb_score"], -e["sec_per_step"], -e["param_count"]),
        reverse=True,
    )
    winner = ranked_lb[0]
    if baseline_s4_summary is None:
        baseline_s4_summary = ranked_lb[-1]

    # Build final SystemConfig with the full 3M-steps/seed reference protocol (3000 ep x 1000 steps)
    winning_cfg = apply_params_to_config(base_cfg, winner["params"])
    final_run_cfg, est_final_hours, env_steps_per_seed = compute_final_run_protocol(
        winning_cfg=winning_cfg,
        measured_sec_per_step=float(winner["sec_per_step"]),
        baseline_sec_per_step=baseline_sec_per_step,
        final_seeds=args.final_seeds,
        final_episodes=args.final_episodes,
        final_max_steps=args.final_max_steps,
        final_hours=args.final_hours,
        parallel_seeds=workers if not args.smoke else 3,
        device="cuda" if not args.smoke else device,
        gpu_mem_fraction=gpu_fraction,
    )
    winning_cfg = replace(
        winning_cfg,
        brl=replace(
            winning_cfg.brl,
            total_episodes=final_run_cfg.episodes,
            warmup_episodes=(
                50
                if final_run_cfg.episodes == 3000
                else max(10, int(round(final_run_cfg.episodes * (50.0 / 3000.0))))
            ),
            brl_episodes=(
                2500
                if final_run_cfg.episodes == 3000
                else max(20, int(round(final_run_cfg.episodes * (2500.0 / 3000.0))))
            ),
        ),
        ant_morph=replace(
            winning_cfg.ant_morph,
            max_episode_steps=final_run_cfg.max_steps,
        ),
        run=final_run_cfg,
    )

    header = format_Config_header(
        winner_summary=winner,
        baseline_summary=baseline_s4_summary,
        importances=importances,
        est_final_hours=est_final_hours,
        env_steps_per_seed=env_steps_per_seed,
        run_cfg=final_run_cfg,
    )
    save_config(winning_cfg, final_cfg_path, header_comment=header)

    total_hpo_sec = time.time() - hpo_start_time
    report_md_path, summary_json_path = write_hpo_report(
        out_dir=out_dir,
        winner_summary=winner,
        baseline_summary=baseline_s4_summary,
        stage4_leaderboard=ranked_lb,
        importances=importances,
        winning_cfg=winning_cfg,
        est_final_hours=est_final_hours,
        total_hpo_sec=total_hpo_sec,
    )

    print("\n" + "=" * 88)
    print(f"  HPO COMPLETED IN {format_time_duration(total_hpo_sec)}!")
    print(f"  Winning Candidate : {winner['label']} (LCB = {winner['lcb_score']:+.2f})")
    print(f"  Optimal Config    : {final_cfg_path}")
    print(f"  Detailed Report   : {report_md_path}")
    print("=" * 88, flush=True)

    return {
        "optimal_config_path": str(final_cfg_path),
        "report_md_path": str(report_md_path),
        "summary_json_path": str(summary_json_path),
        "winner": winner,
        "baseline": baseline_s4_summary,
        "importances": importances,
    }


if __name__ == "__main__":
    main()
