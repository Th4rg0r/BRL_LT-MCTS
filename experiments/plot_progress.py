"""Progress & Validation Visualization Tool for LA-MCTS and Balanced RL.

Ingests `progress.jsonl` files (and accompanying `.log` files) to generate
high-resolution publication-quality performance dashboards tracking:
  1. Episode Returns & Dynamic Win Threshold (r_ref)
  2. Meters Traveled & Locomotion Velocity (vx)
  3. Survival & Episode Duration (Steps per Episode)
  4. Representation & Model Dynamics Loss (L_total)
  5. BRL Adversarial Balance (Win Rate & Balancer Budget)
  6. Out-of-Distribution Benchmark Suite Performance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_accompanying_log(log_path: Path) -> Dict[str, Any]:
    """Parses evaluation benchmarks and distance/velocity metrics from a training .log file."""
    if not log_path.exists():
        return {}

    eval_by_ep: Dict[int, Dict[str, Dict[str, float]]] = {}
    eval_scores: Dict[int, float] = {}

    # Regex patterns
    # e.g.: [2026-11-12 00:02:48] >>> [Seed 42 (1/3) | Eval @ Ep  240 (171,479 steps)] Score: -390.12 | Std-Ant: -473.0 | OOD: -307.3
    eval_hdr_re = re.compile(r"Eval @ Ep\s+(\d+)\s+\(\s*([\d,]+)\s+steps\)\s*\]\s+Score:\s*([+-]?\d+(?:\.\d+)?)")
    # e.g.: * Standard-Ant-v5 : Return = -472.98 | Steps = 1000/1000 | Dist =  -0.86m | Vel = -0.017 m/s
    suite_line_re = re.compile(
        r"\*\s+([\w\-]+)\s*:\s+Return\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*Steps\s*=\s*(\d+)/\d+\s*\|\s*Dist\s*=\s*([+-]?\d+(?:\.\d+)?)m\s*\|\s*Vel\s*=\s*([+-]?\d+(?:\.\d+)?)\s*m/s"
    )

    current_ep: Optional[int] = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                hdr_m = eval_hdr_re.search(line)
                if hdr_m:
                    current_ep = int(hdr_m.group(1))
                    eval_scores[current_ep] = float(hdr_m.group(3))
                    if current_ep not in eval_by_ep:
                        eval_by_ep[current_ep] = {}
                    continue

                suite_m = suite_line_re.search(line)
                if suite_m:
                    scenario = suite_m.group(1).strip()
                    ret = float(suite_m.group(2))
                    steps = int(suite_m.group(3))
                    dist = float(suite_m.group(4))
                    vel = float(suite_m.group(5))
                    ep_key = current_ep if current_ep is not None else 0
                    if ep_key not in eval_by_ep:
                        eval_by_ep[ep_key] = {}
                    eval_by_ep[ep_key][scenario] = {
                        "return": ret,
                        "steps": float(steps),
                        "total_distance": dist,
                        "x_velocity": vel,
                    }
    except Exception:
        pass

    return {
        "eval_by_ep": eval_by_ep,
        "eval_scores": eval_scores,
    }


def parse_progress_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """Reads a progress.jsonl file into structured NumPy arrays."""
    records: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return {}

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return {}

    episodes = np.array([r.get("episode", i + 1) for i, r in enumerate(records)], dtype=np.int32)
    total_env_steps = np.array([r.get("total_env_steps", 0) for r in records], dtype=np.int64)
    returns = np.array([r.get("return", 0.0) for r in records], dtype=np.float64)
    steps = np.array([r.get("steps", 0) for r in records], dtype=np.int32)
    losses = np.array([r.get("loss_total", 0.0) for r in records], dtype=np.float64)
    win_rates = np.array([r.get("ema_win_rate", 0.0) for r in records], dtype=np.float64)
    r_refs = np.array([r.get("r_ref", 0.0) for r in records], dtype=np.float64)
    budgets = np.array([r.get("budget", 0.0) for r in records], dtype=np.float64)
    durations = np.array([r.get("ep_duration", 0.0) for r in records], dtype=np.float64)
    wins = np.array([1 if r.get("win", False) else 0 for r in records], dtype=np.int32)

    # Locomotion metrics (direct if present, otherwise default to nan/estimated)
    has_direct_dist = "total_distance" in records[0] or any("total_distance" in r for r in records)
    if has_direct_dist:
        distances = np.array([float(r.get("total_distance", 0.0)) for r in records], dtype=np.float64)
        velocities = np.array([float(r.get("x_velocity", 0.0)) for r in records], dtype=np.float64)
    else:
        distances = np.full(len(records), np.nan, dtype=np.float64)
        velocities = np.full(len(records), np.nan, dtype=np.float64)

    seed_val = records[0].get("seed", 0)

    # Check for accompanying log file
    log_cand1 = jsonl_path.with_name(f"seed_{seed_val}.log")
    log_cand2 = jsonl_path.parent / (jsonl_path.stem.replace("_progress", "") + ".log")
    log_path = log_cand1 if log_cand1.exists() else (log_cand2 if log_cand2.exists() else None)
    log_data = parse_accompanying_log(log_path) if log_path is not None else {}

    return {
        "seed": seed_val,
        "jsonl_path": str(jsonl_path),
        "log_path": str(log_path) if log_path else None,
        "episodes": episodes,
        "total_env_steps": total_env_steps,
        "returns": returns,
        "steps": steps,
        "losses": losses,
        "win_rates": win_rates,
        "r_refs": r_refs,
        "budgets": budgets,
        "durations": durations,
        "wins": wins,
        "distances": distances,
        "velocities": velocities,
        "has_direct_dist": has_direct_dist,
        "eval_by_ep": log_data.get("eval_by_ep", {}),
        "eval_scores": log_data.get("eval_scores", {}),
    }


def compute_rolling_mean(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Computes a causal rolling average with initial edge expansion."""
    if len(arr) == 0:
        return np.array([])
    window = max(1, min(window, len(arr)))
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def plot_single_progress(
    data: Dict[str, Any],
    output_path: Path,
    title_suffix: str = "",
    rolling_window: int = 20,
) -> Path:
    """Renders a comprehensive 6-panel progress dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")

    seed = data.get("seed", "N/A")
    episodes = data["episodes"]
    env_steps = data["total_env_steps"]
    returns = data["returns"]
    steps = data["steps"]
    losses = data["losses"]
    win_rates = data["win_rates"]
    r_refs = data["r_refs"]
    budgets = data["budgets"]
    distances = data["distances"]
    velocities = data["velocities"]
    has_direct_dist = data["has_direct_dist"]
    eval_by_ep = data["eval_by_ep"]

    ret_rolling = compute_rolling_mean(returns, rolling_window)

    # -------------------------------------------------------------
    # Panel 1: Episode Returns & Reference Target
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    ax1.scatter(episodes, returns, color="#4a90e2", alpha=0.35, s=14, label="Raw Return")
    ax1.plot(episodes, ret_rolling, color="#1b4f72", linewidth=2.0, label=f"Rolling Mean (w={rolling_window})")
    if len(r_refs) > 0 and not np.all(r_refs == 0.0):
        ax1.plot(episodes, r_refs, color="#e67e22", linestyle="--", linewidth=1.5, label="Win Threshold (R_ref)")
    ax1.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax1.set_title("1. Cumulative Episode Return", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Locomotion & Meters Traveled (Training or Eval)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    has_plotted_dist = False

    # Check direct training distance
    if has_direct_dist and not np.all(np.isnan(distances)):
        dist_rolling = compute_rolling_mean(distances, rolling_window)
        ax2.scatter(episodes, distances, color="#27ae60", alpha=0.35, s=14, label="Distance (Train)")
        ax2.plot(episodes, dist_rolling, color="#196f3d", linewidth=2.0, label=f"Train Rolling (w={rolling_window})")
        has_plotted_dist = True

    # Plot evaluation distances from log
    if eval_by_ep:
        eval_eps = sorted(eval_by_ep.keys())
        scenarios = set()
        for ep in eval_eps:
            scenarios.update(eval_by_ep[ep].keys())

        colors = {"Standard-Ant-v5": "#2980b9", "Standard-Ant-v4": "#2980b9", "Tripod-Ant": "#d35400",
                  "Spider-Ant": "#8e44ad", "Heavy-Ice-Ant": "#16a085", "Hexapod-Ant": "#c0392b"}
        for sc in sorted(scenarios):
            x_vals = [ep for ep in eval_eps if sc in eval_by_ep[ep]]
            y_vals = [eval_by_ep[ep][sc]["total_distance"] for ep in x_vals]
            if x_vals:
                ax2.plot(x_vals, y_vals, marker="o", markersize=4, linestyle="-",
                         color=colors.get(sc, "#34495e"), label=f"{sc} (Eval Dist)")
                has_plotted_dist = True

    if not has_plotted_dist:
        # Fallback: estimate from step count & zero forward progression
        ax2.text(0.5, 0.5, "Meters traveled logged in upcoming runs\n(Historical JSONL logged steps/return only)",
                 horizontalalignment="center", verticalalignment="center",
                 transform=ax2.transAxes, fontsize=10, color="#7f8c8d")

    ax2.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
    ax2.set_title("2. Locomotion Progress (Meters Traveled)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Distance (m)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    if has_plotted_dist:
        ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 3: Stability & Episode Duration (Steps per Episode)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    ax3.set_facecolor("#ffffff")
    ax3.scatter(episodes, steps, color="#8e44ad", alpha=0.35, s=14, label="Steps Survived")
    steps_rolling = compute_rolling_mean(steps, rolling_window)
    ax3.plot(episodes, steps_rolling, color="#5b2c6f", linewidth=2.0, label=f"Rolling Mean (w={rolling_window})")
    ax3.axhline(1000, color="#e74c3c", linestyle="--", linewidth=1.2, label="Max Cap (1000 steps)")
    ax3.set_title("3. Survival & Episode Duration", fontsize=12, fontweight="bold", pad=8)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps Survived")
    ax3.set_ylim(-20, 1050)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 4: Loss Progression & Latent Model Convergence
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    ax4.set_facecolor("#ffffff")
    valid_loss_mask = losses > 0
    if np.any(valid_loss_mask):
        ep_l = episodes[valid_loss_mask]
        ls_vals = losses[valid_loss_mask]
        ax4.plot(ep_l, ls_vals, color="#e74c3c", alpha=0.4, linewidth=1.0, label="Raw Loss")
        loss_rolling = compute_rolling_mean(ls_vals, rolling_window)
        ax4.plot(ep_l, loss_rolling, color="#922b21", linewidth=2.0, label=f"Rolling Loss (w={rolling_window})")
    ax4.set_title("4. Latent Model & Representation Loss (L_total)", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Total Loss")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 5: BRL Meta-Game Balance (Win Rate & Budget)
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    ax5.set_facecolor("#ffffff")
    color_wr = "#2980b9"
    ax5.plot(episodes, win_rates * 100.0, color=color_wr, linewidth=2.0, label="Agent Win Rate (EMA)")
    ax5.axhline(50.0, color="#27ae60", linestyle="--", linewidth=1.5, label="Target Balance (50%)")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Win Rate (%)", color=color_wr)
    ax5.tick_params(axis="y", labelcolor=color_wr)
    ax5.set_ylim(-2, 102)
    ax5.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for Balancer Budget
    if np.any(budgets > 0):
        ax5_b = ax5.twinx()
        color_b = "#d35400"
        ax5_b.plot(episodes, budgets, color=color_b, linestyle=":", linewidth=1.5, label="Budget (B_e)")
        ax5_b.set_ylabel("Budget Allocation", color=color_b)
        ax5_b.tick_params(axis="y", labelcolor=color_b)

    ax5.set_title("5. BRL Adversarial Triad Balance", fontsize=12, fontweight="bold", pad=8)
    lines5_1, labels5_1 = ax5.get_legend_handles_labels()
    if np.any(budgets > 0):
        lines5_2, labels5_2 = ax5_b.get_legend_handles_labels()
        ax5.legend(lines5_1 + lines5_2, labels5_1 + labels5_2, loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax5.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 6: Validation Benchmark Suite Returns & OOD Robustness
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    ax6.set_facecolor("#ffffff")
    if eval_by_ep:
        eval_eps = sorted(eval_by_ep.keys())
        scenarios = set()
        for ep in eval_eps:
            scenarios.update(eval_by_ep[ep].keys())

        colors = {"Standard-Ant-v5": "#2980b9", "Standard-Ant-v4": "#2980b9", "Tripod-Ant": "#d35400",
                  "Spider-Ant": "#8e44ad", "Heavy-Ice-Ant": "#16a085", "Hexapod-Ant": "#c0392b"}
        for sc in sorted(scenarios):
            x_vals = [ep for ep in eval_eps if sc in eval_by_ep[ep]]
            y_vals = [eval_by_ep[ep][sc]["return"] for ep in x_vals]
            if x_vals:
                ax6.plot(x_vals, y_vals, marker="s", markersize=4, linewidth=1.5,
                         color=colors.get(sc, "#34495e"), label=sc)
        ax6.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.0)
        ax6.legend(loc="lower right", fontsize=8, framealpha=0.9)
    else:
        ax6.text(0.5, 0.5, "Benchmark evaluations recorded every 10 eps\n(Check seed_*.log for details)",
                 horizontalalignment="center", verticalalignment="center",
                 transform=ax6.transAxes, fontsize=10, color="#7f8c8d")

    ax6.set_title("6. Validation & OOD Benchmark Returns", fontsize=12, fontweight="bold", pad=8)
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Evaluation Return")
    ax6.grid(True, linestyle="--", alpha=0.4)

    # Header and Footer
    total_steps_str = f"{env_steps[-1]:,}" if len(env_steps) > 0 else "0"
    main_title = f"LA-MCTS + Balanced RL Performance Dashboard [Seed {seed}] {title_suffix}"
    sub_title = (
        f"Episodes: {len(episodes)} | Total Env Steps: {total_steps_str} | "
        f"Return (Min/Max/Last): {np.min(returns):.1f} / {np.max(returns):.1f} / {returns[-1]:.1f} | "
        f"Loss: {losses[-1]:.2f} (from {losses[0]:.2f})"
    )
    plt.suptitle(f"{main_title}\n{sub_title}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multi_seed_progress(
    datasets: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Renders a combined multi-seed comparison dashboard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.patch.set_facecolor("#fcfcfc")

    palette = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad", "#c0392b", "#16a085"]

    for idx, d in enumerate(datasets):
        color = palette[idx % len(palette)]
        seed = d.get("seed", idx)
        eps = d["episodes"]
        lbl = f"Seed {seed}"

        # 1. Returns
        ret_roll = compute_rolling_mean(d["returns"], window=20)
        axes[0, 0].plot(eps, ret_roll, color=color, linewidth=2.0, label=lbl)

        # 2. Losses
        vmask = d["losses"] > 0
        if np.any(vmask):
            axes[0, 1].plot(eps[vmask], compute_rolling_mean(d["losses"][vmask], window=20),
                            color=color, linewidth=2.0, label=lbl)

        # 3. Win Rates
        axes[1, 0].plot(eps, d["win_rates"] * 100.0, color=color, linewidth=2.0, label=lbl)

        # 4. Steps per ep
        axes[1, 1].plot(eps, compute_rolling_mean(d["steps"], window=20), color=color, linewidth=2.0, label=lbl)

    axes[0, 0].set_title("1. Episode Returns (Rolling Mean 20)", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend()

    axes[0, 1].set_title("2. Representation & Model Dynamics Loss", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss (Rolling 20)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend()

    axes[1, 0].set_title("3. Agent EMA Win Rate (%)", fontweight="bold")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Win Rate (%)")
    axes[1, 0].axhline(50.0, color="#7f8c8d", linestyle="--", alpha=0.7)
    axes[1, 0].set_ylim(-2, 102)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend()

    axes[1, 1].set_title("4. Survival (Steps / Episode)", fontweight="bold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].axhline(1000, color="#e74c3c", linestyle="--", alpha=0.7)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend()

    plt.suptitle("LA-MCTS + Balanced RL Multi-Seed Progression Overview", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_stats_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Exports key numerical statistics into a JSON summary file."""
    stats = {
        "seed": int(data.get("seed", 0)),
        "completed_episodes": int(len(data.get("episodes", []))),
        "total_env_steps": int(data["total_env_steps"][-1]) if len(data.get("total_env_steps", [])) > 0 else 0,
        "return_mean": float(np.mean(data["returns"])) if len(data.get("returns", [])) > 0 else 0.0,
        "return_best": float(np.max(data["returns"])) if len(data.get("returns", [])) > 0 else 0.0,
        "return_min": float(np.min(data["returns"])) if len(data.get("returns", [])) > 0 else 0.0,
        "return_last20_mean": float(np.mean(data["returns"][-20:])) if len(data.get("returns", [])) >= 20 else 0.0,
        "loss_start": float(data["losses"][0]) if len(data.get("losses", [])) > 0 else 0.0,
        "loss_min": float(np.min(data["losses"][data["losses"] > 0])) if np.any(data.get("losses", 0) > 0) else 0.0,
        "loss_final": float(data["losses"][-1]) if len(data.get("losses", [])) > 0 else 0.0,
        "final_win_rate": float(data["win_rates"][-1]) if len(data.get("win_rates", [])) > 0 else 0.0,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and validation charts from progress.jsonl")
    parser.add_argument("--input", "-i", type=str, help="Path to single progress.jsonl file")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing one or more progress.jsonl files")
    parser.add_argument("--output", "-o", type=str, help="Path to output image (.png)")
    parser.add_argument("--rolling", "-r", type=int, default=20, help="Rolling window size (default: 20)")
    args = parser.parse_args()

    if not args.input and not args.dir:
        parser.error("Must provide either --input or --dir")

    input_paths: List[Path] = []
    if args.input:
        input_paths.append(Path(args.input))
    if args.dir:
        d = Path(args.dir)
        input_paths.extend(sorted(d.glob("*progress.jsonl")))

    if not input_paths:
        print(f"Error: No progress.jsonl files found in {args.dir}")
        return

    datasets: List[Dict[str, Any]] = []
    for p in input_paths:
        data = parse_progress_jsonl(p)
        if data and len(data.get("episodes", [])) > 0:
            datasets.append(data)
            out_img = Path(args.output) if (args.output and len(input_paths) == 1) else p.with_name(f"{p.stem}_chart.png")
            plot_single_progress(data, out_img, rolling_window=args.rolling)
            export_stats_json(data, out_img.with_suffix(".json"))
            print(f"[OK] Generated chart: {out_img}")

    if len(datasets) > 1:
        multi_out = Path(args.output) if args.output else Path(args.dir) / "multi_seed_overview.png"
        plot_multi_seed_progress(datasets, multi_out)
        print(f"[OK] Generated multi-seed overview: {multi_out}")


if __name__ == "__main__":
    main()
