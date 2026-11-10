"""Trial Evaluation Worker with Cost/VRAM Constraints and Intermediate ASHA Pruning."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import torch

from config import SystemConfig
from experiments.hpo.search_space import estimate_static_model_stats
from experiments.train_ant_brl import (
    compute_composite_benchmark_score,
    run_single_seed,
)


@dataclass(frozen=True)
class FidelityStage:
    """Defines a multi-fidelity optimization rung."""

    name: str
    episodes: int
    max_steps: int
    eval_interval: int
    seeds: Tuple[int, ...]
    budget_fraction: float


@dataclass
class TrialEvalResult:
    """Aggregated outcome of evaluating a hyperparameter configuration at a FidelityStage."""

    raw_score_mean: float
    raw_score_std: float
    standard_ant_mean: float
    ood_suite_mean: float
    penalized_objective: float
    cost_penalty: float
    sec_per_step: float
    total_wall_sec: float
    total_env_steps: int
    param_count: int
    peak_gpu_mem_mb: float
    ema_win_rate: float
    feasible: bool
    pruned: bool
    infeasibility_reason: str = ""


def evaluate_config_at_fidelity(
    cfg: SystemConfig,
    stage: FidelityStage,
    device: str = "cpu",
    objective_mode: str = "mixed",
    baseline_sec_per_step: float = 0.02,
    baseline_seed_std: float = 10.0,
    cost_penalty_lambda: float = 0.5,
    max_gpu_mem_gb: float = 6.0,
    max_params: int = 25_000_000,
    max_slowdown_factor: float = 3.5,
    gpu_mem_fraction_per_worker: float = 0.30,
    pruning_callback: Optional[Callable[[int, float], bool]] = None,
    heartbeat_callback: Optional[
        Callable[[int, int, int, int, float, float, float], None]
    ] = None,
) -> TrialEvalResult:
    """Evaluates `cfg` across `stage.seeds` with single-thread pinning and cost/VRAM guards.

    Computes the composite benchmark score averaged over the last 2 evaluation checkpoints
    of each seed, checks hard feasibility constraints (parameter count, GPU VRAM, step latency),
    and applies the soft logarithmic slowdown penalty.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # 1. Static pre-check before instantiating environments or running rollouts
    static_stats = estimate_static_model_stats(cfg, max_steps=stage.max_steps)
    est_params = int(static_stats["param_count"])
    est_vram_gb = float(static_stats["est_vram_mb"]) / 1024.0

    if est_params > max_params:
        return TrialEvalResult(
            raw_score_mean=-1e6,
            raw_score_std=0.0,
            standard_ant_mean=-1e6,
            ood_suite_mean=-1e6,
            penalized_objective=-1e6,
            cost_penalty=1e6,
            sec_per_step=0.0,
            total_wall_sec=0.0,
            total_env_steps=0,
            param_count=est_params,
            peak_gpu_mem_mb=float(static_stats["est_vram_mb"]),
            ema_win_rate=0.0,
            feasible=False,
            pruned=True,
            infeasibility_reason=f"Parameter count {est_params:,} exceeds {max_params:,}",
        )

    if max_gpu_mem_gb > 0.0 and est_vram_gb > max_gpu_mem_gb:
        return TrialEvalResult(
            raw_score_mean=-1e6,
            raw_score_std=0.0,
            standard_ant_mean=-1e6,
            ood_suite_mean=-1e6,
            penalized_objective=-1e6,
            cost_penalty=1e6,
            sec_per_step=0.0,
            total_wall_sec=0.0,
            total_env_steps=0,
            param_count=est_params,
            peak_gpu_mem_mb=float(static_stats["est_vram_mb"]),
            ema_win_rate=0.0,
            feasible=False,
            pruned=True,
            infeasibility_reason=f"Estimated VRAM {est_vram_gb:.2f} GB exceeds {max_gpu_mem_gb:.2f} GB",
        )

    cfg_with_cap = replace(
        cfg,
        run=replace(cfg.run, gpu_mem_fraction=gpu_mem_fraction_per_worker),
    )

    seed_scores: List[float] = []
    seed_std_ant: List[float] = []
    seed_ood_ant: List[float] = []
    seed_win_rates: List[float] = []
    total_steps_all = 0
    total_wall_all = 0.0
    max_peak_mb = float(static_stats["est_vram_mb"])
    actual_params = est_params
    was_pruned = False
    infeasible_reason = ""

    for s_idx, seed_val in enumerate(stage.seeds):
        checkpoint_scores: List[float] = []

        def _on_eval(
            ep: int,
            bench_dict: Dict[str, Dict[str, float]],
            telemetry: Dict[str, Any],
        ) -> bool:
            score_ep = compute_composite_benchmark_score(
                bench_dict, objective_mode=objective_mode
            )
            checkpoint_scores.append(score_ep)
            # Trailing-2 checkpoint average for lower noise
            trailing = float(np.mean(checkpoint_scores[-2:]))

            sec_step = float(telemetry.get("sec_per_step", 0.0))
            if heartbeat_callback is not None:
                peak_mb = float(telemetry.get("peak_gpu_mem_mb", 0.0))
                heartbeat_callback(
                    ep,
                    stage.episodes,
                    s_idx + 1,
                    len(stage.seeds),
                    score_ep,
                    sec_step,
                    peak_mb,
                )

            # Check hard slowdown limit after warmup episodes
            if (
                baseline_sec_per_step > 0.0
                and ep >= max(2, stage.eval_interval)
                and sec_step > baseline_sec_per_step * max_slowdown_factor * 1.5
            ):
                return True

            if s_idx == 0 and pruning_callback is not None:
                return bool(pruning_callback(ep, trailing))
            return False

        try:
            seed_res = run_single_seed(
                seed=seed_val,
                cfg=cfg_with_cap,
                episodes=stage.episodes,
                max_steps=stage.max_steps,
                eval_interval=stage.eval_interval,
                device=device,
                seed_idx=s_idx,
                total_seeds=len(stage.seeds),
                verbose=False,
                eval_callback=_on_eval,
                max_gpu_mem_gb=max_gpu_mem_gb,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return TrialEvalResult(
                    raw_score_mean=-1e6,
                    raw_score_std=0.0,
                    standard_ant_mean=-1e6,
                    ood_suite_mean=-1e6,
                    penalized_objective=-1e6,
                    cost_penalty=1e6,
                    sec_per_step=0.0,
                    total_wall_sec=0.0,
                    total_env_steps=total_steps_all,
                    param_count=actual_params,
                    peak_gpu_mem_mb=max_gpu_mem_gb * 1024.0,
                    ema_win_rate=0.0,
                    feasible=False,
                    pruned=True,
                    infeasibility_reason="CUDA Out of Memory",
                )
            raise

        actual_params = int(seed_res.get("param_count", est_params))
        max_peak_mb = max(max_peak_mb, float(seed_res.get("peak_gpu_mem_mb", 0.0)))
        total_steps_all += int(seed_res.get("total_env_steps", 0))
        total_wall_all += float(seed_res.get("total_wall_sec", 0.0))
        seed_win_rates.append(float(seed_res.get("final_ema_win_rate", 0.5)))

        if seed_res.get("mem_exceeded", False):
            infeasible_reason = f"Peak GPU memory {max_peak_mb / 1024.0:.2f} GB > {max_gpu_mem_gb:.2f} GB"
            was_pruned = True
            break

        benchmarks = seed_res.get("benchmarks", {})
        sorted_eps = sorted(benchmarks.keys())
        if sorted_eps:
            last_two_eps = sorted_eps[-2:]
            comp_vals = [
                compute_composite_benchmark_score(
                    benchmarks[e], objective_mode=objective_mode
                )
                for e in last_two_eps
            ]
            std_vals = [
                compute_composite_benchmark_score(
                    benchmarks[e], objective_mode="standard"
                )
                for e in last_two_eps
            ]
            ood_vals = [
                compute_composite_benchmark_score(
                    benchmarks[e], objective_mode="ood"
                )
                for e in last_two_eps
            ]
            seed_scores.append(float(np.mean(comp_vals)))
            seed_std_ant.append(float(np.mean(std_vals)))
            seed_ood_ant.append(float(np.mean(ood_vals)))
        else:
            seed_scores.append(-1e4)
            seed_std_ant.append(-1e4)
            seed_ood_ant.append(-1e4)

        if seed_res.get("pruned", False):
            was_pruned = True
            break

    raw_mean = float(np.mean(seed_scores)) if seed_scores else -1e4
    raw_std = float(np.std(seed_scores)) if len(seed_scores) > 1 else 0.0
    std_ant_mean = float(np.mean(seed_std_ant)) if seed_std_ant else raw_mean
    ood_ant_mean = float(np.mean(seed_ood_ant)) if seed_ood_ant else raw_mean
    ema_w_mean = float(np.mean(seed_win_rates)) if seed_win_rates else 0.5
    sec_per_step = total_wall_all / max(1, total_steps_all)

    # Blend measured step time with static architectural cost factor so noisy OS jitter
    # does not distort the cost penalty while genuine model/MCTS bloat is always penalized
    measured_ratio = (
        sec_per_step / baseline_sec_per_step if baseline_sec_per_step > 1e-6 else 1.0
    )
    eff_slowdown = 0.6 * measured_ratio + 0.4 * float(static_stats["rel_cost_factor"])
    log_slowdown = max(0.0, math.log2(max(1.0, eff_slowdown)))
    par_over = max(0.0, (actual_params - 2_000_000) / 5_000_000.0)
    cost_penalty = (
        cost_penalty_lambda
        * max(1.0, baseline_seed_std)
        * (log_slowdown + 0.25 * par_over)
    )

    feasible = (not bool(infeasible_reason)) and (
        eff_slowdown <= max_slowdown_factor
    )
    if not feasible and not infeasible_reason:
        infeasible_reason = f"Slowdown {eff_slowdown:.2f}x exceeds {max_slowdown_factor:.2f}x limit"

    penalized_obj = raw_mean - cost_penalty
    if not math.isfinite(penalized_obj):
        penalized_obj = -1e6
        feasible = False
        infeasible_reason = "Non-finite objective"

    return TrialEvalResult(
        raw_score_mean=raw_mean,
        raw_score_std=raw_std,
        standard_ant_mean=std_ant_mean,
        ood_suite_mean=ood_ant_mean,
        penalized_objective=penalized_obj,
        cost_penalty=cost_penalty,
        sec_per_step=sec_per_step,
        total_wall_sec=total_wall_all,
        total_env_steps=total_steps_all,
        param_count=actual_params,
        peak_gpu_mem_mb=max_peak_mb,
        ema_win_rate=ema_w_mean,
        feasible=feasible,
        pruned=was_pruned,
        infeasibility_reason=infeasible_reason,
    )
