"""End-to-End Training and Benchmark Verification Script for LA-MCTS + BRL on Ant-v4."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.*")
import numpy as np
import torch

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
from brl_core.balancer import DDPGBalancer
from brl_core.orchestrator import BRLTrainingOrchestrator
from brl_core.tweaker import StochasticTweakerActorCritic
from config import SystemConfig, load_config
from la_mcts.agent import LatentActionMCTSAgent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_time_duration(seconds: float) -> str:
    """Formats a duration in seconds to a compact, human-readable string."""
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


def evaluate_benchmark_suite(
    agent: LatentActionMCTSAgent,
    env: MorphologicalAntBRLWrapper,
    max_steps: int = 200,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluates the agent deterministically on Standard Ant-v4 and the OOD Creature Suite."""
    suite_specs: Dict[str, Dict[str, Any]] = {
        "Standard-Ant-v4": {
            "tweak": np.zeros(12, dtype=np.float32),
            "num_legs": 4,
        },
        "Tripod-Ant": {
            "tweak": np.array(
                [0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            "num_legs": 3,
        },
        "Spider-Ant": {
            "tweak": np.array(
                [800, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            "num_legs": 4,
        },
        "Heavy-Ice-Ant": {
            "tweak": np.array(
                [0, 0, 0, 0, 600, 700, 0, 0, 0, 300, 0, 0], dtype=np.float32
            ),
            "num_legs": 4,
        },
        "Hexapod-Ant": {
            "tweak": np.array(
                [300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400], dtype=np.float32
            ),
            "num_legs": 6,
        },
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, spec in suite_specs.items():
        obs, info = env.reset(
            options={
                "tweak_vector": spec["tweak"],
                "num_legs": spec["num_legs"],
            }
        )
        morph_mask = info.get(
            "morphology_mask",
            np.ones(agent.cfg.max_action_dim, dtype=np.float32),
        )
        agent.set_episode_context(
            tweak_vector=spec["tweak"], action_mask=morph_mask
        )

        ep_return = 0.0
        steps = 0
        last_info = info
        while steps < max_steps:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_return += float(reward)
            steps += 1
            if terminated or truncated:
                break

        brl_m = last_info.get("brl_metrics", {})
        results[name] = {
            "return": float(ep_return),
            "steps": float(steps),
            "x_velocity": float(brl_m.get("x_velocity", 0.0)),
            "total_distance": float(brl_m.get("total_distance", 0.0)),
        }
        if verbose:
            dist_val = float(brl_m.get("total_distance", 0.0))
            vel_val = float(brl_m.get("x_velocity", 0.0))
            print(
                f"    * {name:<16}: Return = {ep_return:+7.2f} | Steps = {steps:4d}/{max_steps} | "
                f"Dist = {dist_val:+6.2f}m | Vel = {vel_val:+6.3f} m/s",
                flush=True,
            )

    return results


def run_single_seed(
    seed: int,
    cfg: SystemConfig,
    episodes: int,
    max_steps: int,
    eval_interval: int,
    device: str = "cpu",
    seed_idx: int = 0,
    total_seeds: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs the 3-phase BRL + LA-MCTS training pipeline for a single random seed."""
    set_global_seed(seed)
    seed_start_time = time.time()

    if verbose:
        print("\n" + "=" * 80)
        print(f"  >>> SEED {seed_idx + 1}/{total_seeds} (Seed value: {seed}) | {episodes} Episodes on {device}")
        print("=" * 80, flush=True)

    # Use max_action_capacity = 12 so the agent seamlessly handles 3-, 4-, and 6-legged creatures
    la_cfg = replace(
        cfg.la_mcts,
        max_action_dim=cfg.ant_morph.max_action_capacity,
    )

    # Scale R_ref, benchmark_r_ref, and tau_s proportionally to max_steps
    nominal_max_steps = cfg.ant_morph.max_episode_steps
    step_scale = max_steps / float(nominal_max_steps) if nominal_max_steps > 0 else 1.0

    brl_cfg = cfg.brl
    if step_scale != 1.0:
        brl_cfg = replace(
            brl_cfg,
            initial_r_ref=max(10.0, cfg.brl.initial_r_ref * step_scale),
            benchmark_r_ref=max(50.0, cfg.brl.benchmark_r_ref * step_scale),
            tau_s=max(10.0, cfg.brl.tau_s * step_scale),
        )

    # Scale warmup and BRL phases proportionally if episodes is overridden
    if episodes < cfg.brl.total_episodes:
        ep_ratio = episodes / float(cfg.brl.total_episodes)
        scaled_warmup = max(2, int(round(cfg.brl.warmup_episodes * ep_ratio)))
        scaled_brl = max(scaled_warmup + 1, int(round(cfg.brl.brl_episodes * ep_ratio)))
        brl_cfg = replace(
            brl_cfg,
            warmup_episodes=scaled_warmup,
            brl_episodes=scaled_brl,
            total_episodes=episodes,
        )

    sys_cfg = replace(cfg, la_mcts=la_cfg, brl=brl_cfg)

    env = MorphologicalAntBRLWrapper(
        use_contact_forces=sys_cfg.ant_morph.use_contact_forces,
        xml_rebuild_threshold=sys_cfg.ant_morph.xml_rebuild_threshold,
        max_action_capacity=sys_cfg.ant_morph.max_action_capacity,
        fixed_action_space=True,
        default_num_legs=sys_cfg.ant_morph.default_num_legs,
        healthy_z_range=sys_cfg.ant_morph.healthy_z_range,
    )

    agent = LatentActionMCTSAgent(config=sys_cfg.la_mcts, device=device)
    tweaker = StochasticTweakerActorCritic(config=sys_cfg.brl, device=device)
    balancer = DDPGBalancer(config=sys_cfg.brl, device=device)

    orchestrator = BRLTrainingOrchestrator(
        env=env,
        agent=agent,
        tweaker=tweaker,
        balancer=balancer,
        config=sys_cfg,
    )

    episode_logs: List[Dict[str, Any]] = []
    benchmark_evals: Dict[int, Dict[str, Dict[str, float]]] = {}

    for ep in range(1, episodes + 1):
        ep_start_time = time.time()

        def step_callback(cur_step: int, total_step_limit: int, cur_return: float) -> None:
            if (
                verbose
                and sys.stdout.isatty()
                and (cur_step == 1 or cur_step % 25 == 0 or cur_step == total_step_limit)
            ):
                print(
                    f"\r  -> [Seed {seed_idx + 1}/{total_seeds} | Ep {ep:3d}/{episodes}] "
                    f"Step {cur_step:4d}/{total_step_limit} | Return: {cur_return:+7.2f}...",
                    end="",
                    flush=True,
                )

        summary = orchestrator.run_episode(
            episode_idx=ep,
            max_steps=max_steps,
            train_agent_every=max(1, max_steps // 2),
            step_callback=step_callback if verbose else None,
        )
        episode_logs.append(summary)

        ep_duration = time.time() - ep_start_time
        seed_elapsed = time.time() - seed_start_time
        avg_ep_duration = seed_elapsed / ep
        remaining_eps = episodes - ep
        seed_eta = remaining_eps * avg_ep_duration

        if verbose:
            win_tag = "WIN " if summary["win_indicator"] else "LOSS"
            loss_val = summary.get("loss_total", 0.0)
            loss_str = f"{loss_val:5.2f}" if loss_val != 0.0 else "  N/A"
            budget_val = summary.get("budget", 0.0)
            tweak_val = summary.get("mean_tweak", 0.0)
            r_ref_val = summary.get("r_ref", 0.0)
            line = (
                f"\r[Seed {seed_idx + 1}/{total_seeds} | Ep {ep:3d}/{episodes}] "
                f"Steps: {summary['steps']:3d}/{max_steps} | "
                f"Return: {summary['episode_return']:+6.1f} (R_ref: {r_ref_val:4.0f}) | "
                f"WinRate: {summary['ema_win_rate']:4.1%} [{win_tag}] | "
                f"Budget: {budget_val:5.0f} | "
                f"Tweak: {tweak_val:4.1f} | "
                f"Loss: {loss_str} | "
                f"EpTime: {format_time_duration(ep_duration)} | "
                f"SeedETA: {format_time_duration(seed_eta)}"
            )
            print(f"{line:<125}", flush=True)

        if ep % eval_interval == 0 or ep == episodes:
            if verbose:
                print(f"  --- [Benchmark Evaluation @ Episode {ep}/{episodes}] ---", flush=True)
            benchmark_evals[ep] = evaluate_benchmark_suite(
                agent=agent, env=env, max_steps=max_steps, verbose=verbose
            )
            if verbose:
                print(f"  " + "-" * 50, flush=True)

    env.close()
    if verbose:
        total_seed_dur = format_time_duration(time.time() - seed_start_time)
        print(
            f"\n>>> Seed {seed_idx + 1}/{total_seeds} (Seed {seed}) finished in {total_seed_dur}! "
            f"Final EMA Win Rate: {orchestrator.win_evaluator.ema_win_rate:.1%}\n",
            flush=True,
        )

    return {
        "seed": seed,
        "episodes": episode_logs,
        "benchmarks": benchmark_evals,
        "final_ema_win_rate": orchestrator.win_evaluator.ema_win_rate,
    }


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Train and verify LA-MCTS + BRL on Morphological Ant-v4."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes per seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps per episode.",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=None,
        help="Override MCTS simulation count per step.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write checkpoint and summary JSON.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.n_sim is not None:
        cfg = replace(
            cfg,
            la_mcts=replace(cfg.la_mcts, n_sim=args.n_sim, n_sim_eval=args.n_sim),
        )

    num_episodes = (
        args.episodes if args.episodes is not None else cfg.brl.total_episodes
    )
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else cfg.ant_morph.max_episode_steps
    )
    eval_interval = max(1, min(10, num_episodes))

    global_start_time = time.time()
    print("=" * 80)
    print("  LA-MCTS + BRL Training Pipeline")
    print(f"  Device        : {args.device}")
    print(f"  Seeds         : {args.seeds} (42 .. {42 + args.seeds - 1})")
    print(f"  Episodes/Seed : {num_episodes}")
    print(f"  Max Steps/Ep  : {max_steps}")
    print(f"  MCTS Sims/Step: {cfg.la_mcts.n_sim}")
    print(f"  Eval Interval : Every {eval_interval} episodes")
    print(f"  Output Dir    : {args.output_dir}")
    print("=" * 80, flush=True)

    all_seed_results = []
    for s_idx in range(args.seeds):
        seed_val = 42 + s_idx
        res = run_single_seed(
            seed=seed_val,
            cfg=cfg,
            episodes=num_episodes,
            max_steps=max_steps,
            eval_interval=eval_interval,
            device=args.device,
            seed_idx=s_idx,
            total_seeds=args.seeds,
            verbose=True,
        )
        all_seed_results.append(res)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    total_time_str = format_time_duration(time.time() - global_start_time)
    print("=" * 80)
    print(f"  All {args.seeds} seeds completed in {total_time_str}!")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 80, flush=True)

    return {"seeds": all_seed_results, "summary_path": str(summary_path)}


if __name__ == "__main__":
    main()
