"""End-to-End Training and Benchmark Verification Script for LA-MCTS + BRL on Ant-v4."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional
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


def count_agent_parameters(
    agent: LatentActionMCTSAgent,
    tweaker: StochasticTweakerActorCritic,
    balancer: DDPGBalancer,
) -> int:
    """Counts total trainable parameters across Agent, Tweaker, and Balancer."""
    modules = [
        agent.backbone,
        agent.actor_head,
        agent.critic_head,
        agent.rep_trainer,
        tweaker,
        balancer,
    ]
    return int(
        sum(
            p.numel()
            for m in modules
            for p in m.parameters()
            if p.requires_grad
        )
    )


def compute_composite_benchmark_score(
    bench_results: Dict[str, Dict[str, float]],
    objective_mode: str = "mixed",
) -> float:
    """Computes scalar benchmark score from evaluate_benchmark_suite output.

    Args:
        bench_results: Mapping from creature suite name to metrics dict.
        objective_mode: 'mixed' (0.5 * Standard-Ant-v4 + 0.5 * mean OOD),
                        'standard' (Standard-Ant-v4 only), or 'ood' (mean OOD only).
    """
    std_ret = float(bench_results.get("Standard-Ant-v4", {}).get("return", 0.0))
    ood_keys = [k for k in bench_results.keys() if k != "Standard-Ant-v4"]
    ood_ret = (
        float(np.mean([bench_results[k]["return"] for k in ood_keys]))
        if ood_keys
        else std_ret
    )
    if objective_mode == "standard":
        return std_ret
    if objective_mode == "ood":
        return ood_ret
    return 0.5 * std_ret + 0.5 * ood_ret


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
    eval_callback: Optional[
        Callable[[int, Dict[str, Dict[str, float]], Dict[str, Any]], bool]
    ] = None,
    max_gpu_mem_gb: float = 0.0,
) -> Dict[str, Any]:
    """Runs the 3-phase BRL + LA-MCTS training pipeline for a single random seed.

    If eval_callback is provided, it is invoked after each benchmark evaluation as
    `should_prune = eval_callback(ep, benchmark_evals[ep], telemetry)` and stops early
    when it returns True.
    """
    set_global_seed(seed)
    seed_start_time = time.time()

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        if cfg.run.gpu_mem_fraction > 0.0:
            try:
                torch.cuda.set_per_process_memory_fraction(
                    min(0.95, float(cfg.run.gpu_mem_fraction))
                )
            except Exception:
                pass

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
    param_count = count_agent_parameters(agent, tweaker, balancer)

    orchestrator = BRLTrainingOrchestrator(
        env=env,
        agent=agent,
        tweaker=tweaker,
        balancer=balancer,
        config=sys_cfg,
    )

    train_every = (
        int(sys_cfg.la_mcts.train_every_steps)
        if int(sys_cfg.la_mcts.train_every_steps) > 0
        else max(1, max_steps // 2)
    )

    episode_logs: List[Dict[str, Any]] = []
    benchmark_evals: Dict[int, Dict[str, Dict[str, float]]] = {}
    total_env_steps = 0
    pruned = False
    mem_exceeded = False

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
            train_agent_every=train_every,
            step_callback=step_callback if verbose else None,
        )
        episode_logs.append(summary)
        total_env_steps += int(summary.get("steps", 0))

        ep_duration = time.time() - ep_start_time
        seed_elapsed = time.time() - seed_start_time
        avg_ep_duration = seed_elapsed / ep
        remaining_eps = episodes - ep
        seed_eta = remaining_eps * avg_ep_duration

        if use_cuda and max_gpu_mem_gb > 0.0:
            peak_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
            if peak_gb > max_gpu_mem_gb:
                mem_exceeded = True
                pruned = True
                break

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

            if eval_callback is not None:
                elapsed_now = max(1e-6, time.time() - seed_start_time)
                peak_mb_now = (
                    float(torch.cuda.max_memory_allocated() / (1024.0**2))
                    if use_cuda
                    else 0.0
                )
                telemetry = {
                    "episode": ep,
                    "total_env_steps": total_env_steps,
                    "elapsed_sec": elapsed_now,
                    "sec_per_step": elapsed_now / max(1, total_env_steps),
                    "param_count": param_count,
                    "peak_gpu_mem_mb": peak_mb_now,
                    "ema_win_rate": orchestrator.win_evaluator.ema_win_rate,
                }
                if eval_callback(ep, benchmark_evals[ep], telemetry):
                    pruned = True
                    break

    env.close()
    total_wall_sec = max(1e-6, time.time() - seed_start_time)
    peak_gpu_mem_mb = (
        float(torch.cuda.max_memory_allocated() / (1024.0**2))
        if use_cuda
        else 0.0
    )

    if verbose:
        total_seed_dur = format_time_duration(total_wall_sec)
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
        "param_count": param_count,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "total_env_steps": total_env_steps,
        "total_wall_sec": total_wall_sec,
        "sec_per_step": total_wall_sec / max(1, total_env_steps),
        "pruned": pruned,
        "mem_exceeded": mem_exceeded,
    }


def _parallel_seed_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level helper for ProcessPoolExecutor parallel seed execution."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    return run_single_seed(**kwargs)


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
        default=None,
        help="Number of random seeds to run (defaults to config run.seeds or 3).",
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
        default=None,
        help="Compute device (cpu or cuda; defaults to config run.device).",
    )
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=None,
        help="Number of seeds to execute concurrently (defaults to config run.parallel_seeds).",
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

    num_seeds = (
        args.seeds
        if args.seeds is not None
        else (cfg.run.seeds if cfg.run.seeds > 0 else 3)
    )
    seed_offset = cfg.run.seed_offset if cfg.run.seed_offset > 0 else 42
    num_episodes = (
        args.episodes
        if args.episodes is not None
        else (
            cfg.run.episodes
            if cfg.run.episodes > 0
            else cfg.brl.total_episodes
        )
    )
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else (
            cfg.run.max_steps
            if cfg.run.max_steps > 0
            else cfg.ant_morph.max_episode_steps
        )
    )
    device = args.device if args.device is not None else (cfg.run.device or "cpu")
    parallel_seeds = (
        args.parallel_seeds
        if args.parallel_seeds is not None
        else max(1, int(cfg.run.parallel_seeds))
    )
    eval_interval = (
        cfg.run.eval_interval
        if cfg.run.eval_interval > 0 and args.episodes is None
        else max(1, min(10, num_episodes))
    )

    global_start_time = time.time()
    print("=" * 80)
    print("  LA-MCTS + BRL Training Pipeline")
    print(f"  Config        : {args.config}")
    print(f"  Device        : {device}")
    print(f"  Seeds         : {num_seeds} ({seed_offset} .. {seed_offset + num_seeds - 1}) [parallel={parallel_seeds}]")
    print(f"  Episodes/Seed : {num_episodes}")
    print(f"  Max Steps/Ep  : {max_steps}")
    print(f"  MCTS Sims/Step: {cfg.la_mcts.n_sim}")
    print(f"  Eval Interval : Every {eval_interval} episodes")
    print(f"  Output Dir    : {args.output_dir}")
    print("=" * 80, flush=True)

    all_seed_results: List[Dict[str, Any]] = []
    if parallel_seeds <= 1 or num_seeds <= 1:
        for s_idx in range(num_seeds):
            seed_val = seed_offset + s_idx
            res = run_single_seed(
                seed=seed_val,
                cfg=cfg,
                episodes=num_episodes,
                max_steps=max_steps,
                eval_interval=eval_interval,
                device=device,
                seed_idx=s_idx,
                total_seeds=num_seeds,
                verbose=True,
            )
            all_seed_results.append(res)
    else:
        worker_jobs = [
            {
                "seed": seed_offset + s_idx,
                "cfg": cfg,
                "episodes": num_episodes,
                "max_steps": max_steps,
                "eval_interval": eval_interval,
                "device": device,
                "seed_idx": s_idx,
                "total_seeds": num_seeds,
                "verbose": (s_idx == 0),
            }
            for s_idx in range(num_seeds)
        ]
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=min(parallel_seeds, num_seeds)) as pool:
            fut_to_idx = {
                pool.submit(_parallel_seed_worker, job): job["seed_idx"]
                for job in worker_jobs
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results_by_idx[idx] = fut.result()
        all_seed_results = [results_by_idx[i] for i in range(num_seeds)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    total_time_str = format_time_duration(time.time() - global_start_time)
    print("=" * 80)
    print(f"  All {num_seeds} seeds completed in {total_time_str}!")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 80, flush=True)

    return {"seeds": all_seed_results, "summary_path": str(summary_path)}


if __name__ == "__main__":
    main()
