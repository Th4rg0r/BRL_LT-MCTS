"""Real-Time Progress and Diagnostics Monitor for Multi-Fidelity HPO."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from experiments.hpo.trial_worker import TrialEvalResult


def format_time_duration(seconds: float) -> str:
    """Formats a duration in seconds into a compact, human-readable string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def ascii_progress_bar(fraction: float, width: int = 16) -> str:
    """Renders a simple ASCII progress bar e.g. [██████░░░░░░░░░░]."""
    clamped = max(0.0, min(1.0, float(fraction)))
    filled = int(round(clamped * width))
    return "█" * filled + "░" * (width - filled)


@dataclass
class BaselineStats:
    """Baseline metrics captured from Stage 0 default configuration."""

    raw_mean: float
    raw_std: float
    lcb_score: float
    sec_per_step: float
    param_count: int
    peak_vram_mb: float
    standard_ant_mean: float = 0.0
    ood_suite_mean: float = 0.0


class HPOProgressMonitor:
    """Thread-safe monitor reporting optimization progress, quality gains, and hardware telemetry."""

    def __init__(
        self,
        total_budget_sec: float,
        total_stages: int = 5,
        debug: bool = False,
        quiet: bool = False,
    ) -> None:
        self.total_budget_sec = max(10.0, float(total_budget_sec))
        self.total_stages = total_stages
        self.debug = bool(debug)
        self.quiet = bool(quiet)

        self.hpo_start_time = time.time()
        self.baseline: Optional[BaselineStats] = None
        self.best_trial: Optional[Dict[str, Any]] = None

        self.stage_idx = 0
        self.stage_name = ""
        self.stage_start_time = time.time()
        self.stage_target_trials = 0
        self.stage_timeout_sec = 0.0
        self.stage_episodes = 0
        self.stage_max_steps = 0
        self.stage_seeds_count = 0

        self.stage_completed_trials = 0
        self.stage_pruned_trials = 0
        self.stage_infeasible_trials = 0
        self.stage_best_score = float("-inf")
        self.stage_best_trial_num = -1

        self._lock = threading.Lock()

    def set_baseline(self, stats: BaselineStats) -> None:
        """Stores Stage 0 baseline metrics and prints calibration summary."""
        with self._lock:
            self.baseline = stats
            if not self.quiet:
                print("\n" + "=" * 88)
                print("  STAGE 0: BASELINE CALIBRATION SUMMARY")
                print("=" * 88)
                print(
                    f"  Baseline LCB Score : {stats.lcb_score:+7.2f} "
                    f"(Raw Mean: {stats.raw_mean:+7.2f} +/- {stats.raw_std:5.2f})"
                )
                print(
                    f"  Standard-Ant-v5    : {stats.standard_ant_mean:+7.2f} | "
                    f"OOD Suite Mean: {stats.ood_suite_mean:+7.2f}"
                )
                print(
                    f"  Step Latency       : {stats.sec_per_step * 1000:6.2f} ms/step | "
                    f"Model Size: {stats.param_count / 1e6:5.2f}M params | Peak VRAM: {stats.peak_vram_mb / 1024.0:.2f} GB"
                )
                print("=" * 88, flush=True)

    def start_stage(
        self,
        stage_idx: int,
        stage_name: str,
        target_trials: int,
        timeout_sec: float,
        episodes: int,
        max_steps: int,
        seeds_count: int,
        active_params_count: int = 0,
    ) -> None:
        """Initializes a new optimization stage and displays the stage banner."""
        with self._lock:
            self.stage_idx = stage_idx
            self.stage_name = stage_name
            self.stage_start_time = time.time()
            self.stage_target_trials = max(1, target_trials)
            self.stage_timeout_sec = float(timeout_sec)
            self.stage_episodes = episodes
            self.stage_max_steps = max_steps
            self.stage_seeds_count = seeds_count

            self.stage_completed_trials = 0
            self.stage_pruned_trials = 0
            self.stage_infeasible_trials = 0
            self.stage_best_score = float("-inf")
            self.stage_best_trial_num = -1

            if not self.quiet:
                elapsed_total = time.time() - self.hpo_start_time
                pct_budget = min(100.0, (elapsed_total / self.total_budget_sec) * 100.0)
                active_str = f"on {active_params_count} active parameters" if active_params_count > 0 else "all parameters"
                print("\n" + "-" * 88)
                print(
                    f">>> [Stage {self.stage_idx}/{self.total_stages}: {self.stage_name}] "
                    f"Target={self.stage_target_trials} trials {active_str}"
                )
                print(
                    f"    Fidelity: {self.stage_episodes} eps x {self.stage_max_steps} steps x "
                    f"{self.stage_seeds_count} seeds | Stage Budget: {format_time_duration(self.stage_timeout_sec)}"
                )
                print(
                    f"    Global Clock: {format_time_duration(elapsed_total)} elapsed / "
                    f"{format_time_duration(self.total_budget_sec)} total ({pct_budget:.1f}% budget used)"
                )
                print("-" * 88, flush=True)

    def record_trial_heartbeat(
        self,
        trial_num: int,
        ep: int,
        ep_total: int,
        seed_idx: int,
        total_seeds: int,
        intermediate_score: float,
        sec_per_step: float,
        peak_mb: float,
    ) -> None:
        """Prints live evaluation status inside an active trial when debug is enabled."""
        if not self.debug or self.quiet:
            return
        with self._lock:
            base_sec = self.baseline.sec_per_step if self.baseline else 0.02
            speed_ratio = (sec_per_step / base_sec) if base_sec > 0 else 1.0
            print(
                f"    ↳ [Trial #{trial_num:2d} | Seed {seed_idx}/{total_seeds} | Ep {ep:3d}/{ep_total:3d}] "
                f"Eval Score: {intermediate_score:+7.2f} | Speed: {sec_per_step * 1000:5.1f} ms ({speed_ratio:.1f}x) | "
                f"Peak VRAM: {peak_mb / 1024.0:4.2f} GB",
                flush=True,
            )

    def record_trial_result(
        self,
        trial_num: int,
        eval_res: TrialEvalResult,
        params: Dict[str, Any],
        cfg_summary: Dict[str, Any],
    ) -> None:
        """Records a trial's result, updates counters, and prints the formatted trial progress."""
        with self._lock:
            is_new_best = False
            gain_vs_baseline = 0.0
            gain_pct_str = ""
            base_score = self.baseline.lcb_score if self.baseline else 0.0

            if not eval_res.feasible:
                self.stage_infeasible_trials += 1
            elif eval_res.pruned:
                self.stage_pruned_trials += 1
            else:
                self.stage_completed_trials += 1
                score = eval_res.penalized_objective
                if score > self.stage_best_score:
                    self.stage_best_score = score
                    self.stage_best_trial_num = trial_num

                if self.best_trial is None or score > self.best_trial["score"]:
                    is_new_best = True
                    self.best_trial = {
                        "trial_num": trial_num,
                        "stage_name": self.stage_name,
                        "score": score,
                        "raw_mean": eval_res.raw_score_mean,
                        "raw_std": eval_res.raw_score_std,
                        "standard_ant_mean": eval_res.standard_ant_mean,
                        "ood_suite_mean": eval_res.ood_suite_mean,
                        "cost_penalty": eval_res.cost_penalty,
                        "sec_per_step": eval_res.sec_per_step,
                        "param_count": eval_res.param_count,
                        "peak_gpu_mem_mb": eval_res.peak_gpu_mem_mb,
                        "params": dict(params),
                    }

                gain_vs_baseline = score - base_score
                if abs(base_score) > 1e-3:
                    pct = (gain_vs_baseline / abs(base_score)) * 100.0
                    gain_pct_str = f" ({pct:+.1f}%)"

            if self.quiet:
                return

            # Compute timing & ETA
            now = time.time()
            stage_elapsed = now - self.stage_start_time
            hpo_elapsed = now - self.hpo_start_time
            trials_evaluated = (
                self.stage_completed_trials
                + self.stage_pruned_trials
                + self.stage_infeasible_trials
            )
            pct_stage = min(100.0, (trials_evaluated / float(self.stage_target_trials)) * 100.0)
            bar = ascii_progress_bar(trials_evaluated / float(self.stage_target_trials), width=12)

            # ETA calculation
            rem_budget = max(0.0, self.total_budget_sec - hpo_elapsed)
            eta_str = f"~{format_time_duration(rem_budget)}"

            # Key hyperparameter signature
            hp_sig = ", ".join(
                f"{k}={v:.1e}" if isinstance(v, float) and (v < 0.001 or v > 9999)
                else f"{k}={v}"
                for k, v in list(cfg_summary.items())[:6]
            )

            # Step speed multiplier vs baseline
            base_sec = self.baseline.sec_per_step if self.baseline else 0.02
            speed_mult = (eval_res.sec_per_step / base_sec) if base_sec > 0 else 1.0

            status_tag = "OK"
            if not eval_res.feasible:
                status_tag = f"INFEASIBLE ({eval_res.infeasibility_reason})"
            elif eval_res.pruned:
                status_tag = "PRUNED"

            # Header line with progress bar, stage duration, global elapsed and ETA
            hdr = (
                f"  [{self.stage_name} | Trial {trials_evaluated:2d}/{self.stage_target_trials:2d} "
                f"[{bar}] {pct_stage:4.1f}% | Stage: {format_time_duration(stage_elapsed)} | "
                f"HPO: {format_time_duration(hpo_elapsed)}/{format_time_duration(self.total_budget_sec)} | ETA: {eta_str}]"
            )

            if not eval_res.feasible:
                print(hdr)
                print(f"    ↳ REJECTED : {status_tag} | Config: {hp_sig}", flush=True)
            elif eval_res.pruned:
                print(hdr)
                print(
                    f"    ↳ PRUNED   : Terminated early by ASHA | Config: {hp_sig} | "
                    f"Latency: {eval_res.sec_per_step * 1000:.1f} ms",
                    flush=True,
                )
            else:
                print(hdr)
                print(
                    f"    ↳ Result   : Obj = {eval_res.penalized_objective:+7.2f} "
                    f"(Raw: {eval_res.raw_score_mean:+7.2f}, Cost Pen: -{eval_res.cost_penalty:5.2f}) | "
                    f"Std-Ant: {eval_res.standard_ant_mean:+7.2f} | OOD: {eval_res.ood_suite_mean:+7.2f}"
                )
                print(
                    f"    ↳ Hardware : Latency: {eval_res.sec_per_step * 1000:5.1f} ms ({speed_mult:.1f}x baseline) | "
                    f"VRAM: {eval_res.peak_gpu_mem_mb / 1024.0:.2f} GB | Params: {eval_res.param_count / 1e6:.2f}M | "
                    f"Config: {hp_sig}"
                )
                if is_new_best:
                    print(
                        f"    >>> [★ NEW BEST] Obj = {eval_res.penalized_objective:+7.2f} "
                        f"(vs Baseline: {base_score:+7.2f} -> Gain: {gain_vs_baseline:+7.2f}{gain_pct_str})"
                    )
                print("", flush=True)

    def print_stage_dashboard(self, top_candidates: Sequence[Dict[str, Any]]) -> None:
        """Displays a stage summary dashboard with mini-leaderboard."""
        with self._lock:
            if self.quiet:
                return
            now = time.time()
            stage_elapsed = now - self.stage_start_time
            hpo_elapsed = now - self.hpo_start_time
            rem_budget = max(0.0, self.total_budget_sec - hpo_elapsed)
            base_score = self.baseline.lcb_score if self.baseline else 0.0

            print("\n" + "=" * 88)
            print(f"  STAGE {self.stage_idx} SUMMARY ({self.stage_name}) — Completed in {format_time_duration(stage_elapsed)}")
            print("=" * 88)
            total_trials = (
                self.stage_completed_trials
                + self.stage_pruned_trials
                + self.stage_infeasible_trials
            )
            print(
                f"  Throughput      : {total_trials} trials evaluated "
                f"({self.stage_completed_trials} completed, {self.stage_pruned_trials} pruned by ASHA, "
                f"{self.stage_infeasible_trials} infeasible)"
            )
            if self.stage_best_trial_num >= 0:
                stage_gain = self.stage_best_score - base_score
                print(
                    f"  Stage Best Score: {self.stage_best_score:+7.2f} [Trial #{self.stage_best_trial_num}] "
                    f"(Gain vs Baseline: {stage_gain:+7.2f})"
                )
            if self.best_trial:
                overall_gain = self.best_trial["score"] - base_score
                print(
                    f"  Overall Best    : {self.best_trial['score']:+7.2f} [{self.best_trial['stage_name']} "
                    f"Trial #{self.best_trial['trial_num']}] (Overall Gain: {overall_gain:+7.2f})"
                )
            print(
                f"  Global Progress : {format_time_duration(hpo_elapsed)} elapsed of "
                f"{format_time_duration(self.total_budget_sec)} budget "
                f"({min(100.0, (hpo_elapsed / self.total_budget_sec) * 100.0):.1f}% | Est. Remaining: ~{format_time_duration(rem_budget)})"
            )
            print("-" * 88)
            print(f"  Top {min(3, len(top_candidates))} Configurations in Stage {self.stage_idx}:")
            for idx, cand in enumerate(top_candidates[:3]):
                p = cand.get("params", cand)
                score_val = cand.get("score", cand.get("penalized_objective", 0.0))
                std_ret = cand.get("standard_ant_mean", 0.0)
                ood_ret = cand.get("ood_suite_mean", 0.0)
                sig = f"n_sim={p.get('n_sim', '?')}, d={p.get('d_model', '?')}, upd={p.get('train_every_steps', '?')}, lr={p.get('lr', 0.0):.1e}"
                print(
                    f"    #{idx + 1}: Score = {score_val:+7.2f} | Std-Ant: {std_ret:+7.2f} | "
                    f"OOD: {ood_ret:+7.2f} | {sig}"
                )
            print("=" * 88 + "\n", flush=True)

    def print_final_head_to_head(
        self,
        winner_summary: Dict[str, Any],
        baseline_summary: Dict[str, Any],
    ) -> None:
        """Prints a comprehensive head-to-head comparison table between Winner and Baseline."""
        with self._lock:
            if self.quiet:
                return
            delta_lcb = winner_summary["lcb_score"] - baseline_summary["lcb_score"]
            delta_raw = winner_summary["raw_mean"] - baseline_summary["raw_mean"]
            delta_std = winner_summary["standard_ant_mean"] - baseline_summary["standard_ant_mean"]
            delta_ood = winner_summary["ood_suite_mean"] - baseline_summary["ood_suite_mean"]
            speed_mult = (
                winner_summary["sec_per_step"] / max(1e-6, baseline_summary["sec_per_step"])
            )
            total_elapsed = time.time() - self.hpo_start_time

            print("\n" + "=" * 88)
            print(f"  FINAL OPTIMIZATION SUMMARY (Completed in {format_time_duration(total_elapsed)})")
            print("=" * 88)
            print(f"  Winning Candidate : {winner_summary['label']}")
            print("----------------------------------------------------------------------------------------")
            print(f"  {'Metric':<25} | {'Baseline (Default)':<20} | {'Optimized Winner':<20} | {'Improvement (Delta)':<18}")
            print(f"  {'-' * 25}-+-{'-' * 20}-+-{'-' * 20}-+-{'-' * 18}")
            print(
                f"  {'Robust LCB Score':<25} | {baseline_summary['lcb_score']:+18.2f} | "
                f"{winner_summary['lcb_score']:+18.2f} | {delta_lcb:+16.2f}"
            )
            print(
                f"  {'Raw Composite Return':<25} | {baseline_summary['raw_mean']:+18.2f} | "
                f"{winner_summary['raw_mean']:+18.2f} | {delta_raw:+16.2f}"
            )
            print(
                f"  {'Standard-Ant-v5 Return':<25} | {baseline_summary['standard_ant_mean']:+18.2f} | "
                f"{winner_summary['standard_ant_mean']:+18.2f} | {delta_std:+16.2f}"
            )
            print(
                f"  {'OOD Suite Mean Return':<25} | {baseline_summary['ood_suite_mean']:+18.2f} | "
                f"{winner_summary['ood_suite_mean']:+18.2f} | {delta_ood:+16.2f}"
            )
            print(
                f"  {'Step Latency (ms)':<25} | {baseline_summary['sec_per_step'] * 1000:18.2f} | "
                f"{winner_summary['sec_per_step'] * 1000:18.2f} | {speed_mult:16.2f}x speed"
            )
            print(
                f"  {'Trainable Parameters':<25} | {baseline_summary['param_count'] / 1e6:17.2f}M | "
                f"{winner_summary['param_count'] / 1e6:17.2f}M | {((winner_summary['param_count'] - baseline_summary['param_count']) / 1e6):+15.2f}M"
            )
            print(
                f"  {'Peak GPU VRAM':<25} | {baseline_summary['peak_gpu_mem_mb'] / 1024.0:17.2f} GB | "
                f"{winner_summary['peak_gpu_mem_mb'] / 1024.0:17.2f} GB | {'OK':>16}"
            )
            print("=" * 88, flush=True)
