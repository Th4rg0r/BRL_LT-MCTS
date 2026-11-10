"""End-to-End Training and Benchmark Verification Script for LA-MCTS + BRL on Ant-v4."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.*")
import numpy as np
import torch

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
from brl_core.balancer import DDPGBalancer
from brl_core.orchestrator import BRLTrainingOrchestrator
from brl_core.tweaker import StochasticTweakerActorCritic
from config import SystemConfig, config_from_dict, config_to_dict, load_config
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


def save_training_checkpoint(
    path: str | Path,
    seed: int,
    episode: int,
    total_env_steps: int,
    cfg: SystemConfig,
    agent: LatentActionMCTSAgent,
    tweaker: StochasticTweakerActorCritic,
    balancer: DDPGBalancer,
    orchestrator: BRLTrainingOrchestrator,
    benchmark_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    composite_score: Optional[float] = None,
) -> Path:
    """Saves a complete, self-contained PyTorch checkpoint (.pt) for Agent, Tweaker, and Balancer."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "seed": int(seed),
        "episode": int(episode),
        "total_env_steps": int(total_env_steps),
        "config": config_to_dict(cfg),
        "agent_state_dict": agent.state_dict(),
        "tweaker_state_dict": tweaker.state_dict(),
        "tweaker_optimizer_state_dict": tweaker.optimizer.state_dict(),
        "balancer_state_dict": balancer.state_dict(),
        "balancer_actor_optimizer_state_dict": balancer.actor_optimizer.state_dict(),
        "balancer_critic_optimizer_state_dict": balancer.critic_optimizer.state_dict(),
        "balancer_meta": {
            "last_limits": balancer.last_limits.tolist(),
            "last_budget": float(balancer.last_budget),
            "last_costs": balancer.last_costs.tolist(),
            "last_tweak": balancer.last_tweak.tolist(),
        },
        "win_evaluator": {
            "ema_win_rate": float(orchestrator.win_evaluator.ema_win_rate),
            "r_ref": float(orchestrator.win_evaluator.r_ref),
            "episodes_evaluated": int(orchestrator.win_evaluator.episodes_evaluated),
        },
        "benchmark_metrics": benchmark_metrics or {},
        "composite_score": float(composite_score) if composite_score is not None else None,
    }
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(ckpt_path)
    return ckpt_path


def load_training_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[
    LatentActionMCTSAgent,
    StochasticTweakerActorCritic,
    DDPGBalancer,
    SystemConfig,
    Dict[str, Any],
]:
    """Loads a saved `.pt` checkpoint and returns instantiated `(agent, tweaker, balancer, cfg, payload)`."""
    ckpt_path = Path(path)
    payload: Dict[str, Any] = torch.load(
        ckpt_path, map_location=torch.device(device), weights_only=False
    )
    cfg = config_from_dict(payload.get("config", {}))
    la_cfg = replace(
        cfg.la_mcts,
        max_action_dim=cfg.ant_morph.max_action_capacity,
    )
    cfg = replace(cfg, la_mcts=la_cfg)

    agent = LatentActionMCTSAgent(config=cfg.la_mcts, device=device)
    tweaker = StochasticTweakerActorCritic(config=cfg.brl, device=device)
    balancer = DDPGBalancer(config=cfg.brl, device=device)

    if "agent_state_dict" in payload:
        agent.load_state_dict(payload["agent_state_dict"])
    if "tweaker_state_dict" in payload:
        tweaker.load_state_dict(payload["tweaker_state_dict"])
    if "tweaker_optimizer_state_dict" in payload:
        tweaker.optimizer.load_state_dict(payload["tweaker_optimizer_state_dict"])
    if "balancer_state_dict" in payload:
        balancer.load_state_dict(payload["balancer_state_dict"])
    if "balancer_actor_optimizer_state_dict" in payload:
        balancer.actor_optimizer.load_state_dict(
            payload["balancer_actor_optimizer_state_dict"]
        )
    if "balancer_critic_optimizer_state_dict" in payload:
        balancer.critic_optimizer.load_state_dict(
            payload["balancer_critic_optimizer_state_dict"]
        )
    return agent, tweaker, balancer, cfg, payload


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
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_interval: int = 0,
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
    best_composite_score = float("-inf")
    best_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    periodic_checkpoint_paths: List[str] = []

    seed_ckpt_dir: Optional[Path] = None
    if checkpoint_dir is not None:
        seed_ckpt_dir = Path(checkpoint_dir) / f"seed_{seed}"
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            cur_comp_score = compute_composite_benchmark_score(benchmark_evals[ep])
            if seed_ckpt_dir is not None and cur_comp_score >= best_composite_score:
                best_composite_score = cur_comp_score
                best_ckpt = save_training_checkpoint(
                    path=seed_ckpt_dir / "best_model.pt",
                    seed=seed,
                    episode=ep,
                    total_env_steps=total_env_steps,
                    cfg=sys_cfg,
                    agent=agent,
                    tweaker=tweaker,
                    balancer=balancer,
                    orchestrator=orchestrator,
                    benchmark_metrics=benchmark_evals[ep],
                    composite_score=cur_comp_score,
                )
                best_checkpoint_path = str(best_ckpt)
                if verbose:
                    print(
                        f"    [Checkpoint] New best composite score ({cur_comp_score:+.2f}) -> {best_ckpt}",
                        flush=True,
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

        if (
            seed_ckpt_dir is not None
            and checkpoint_interval > 0
            and ep % checkpoint_interval == 0
            and ep < episodes
        ):
            latest_bench = benchmark_evals.get(ep)
            latest_comp = (
                compute_composite_benchmark_score(latest_bench)
                if latest_bench is not None
                else None
            )
            ep_ckpt = save_training_checkpoint(
                path=seed_ckpt_dir / f"checkpoint_ep_{ep:04d}.pt",
                seed=seed,
                episode=ep,
                total_env_steps=total_env_steps,
                cfg=sys_cfg,
                agent=agent,
                tweaker=tweaker,
                balancer=balancer,
                orchestrator=orchestrator,
                benchmark_metrics=latest_bench,
                composite_score=latest_comp,
            )
            periodic_checkpoint_paths.append(str(ep_ckpt))
            if verbose:
                print(f"  [Checkpoint] Saved milestone checkpoint -> {ep_ckpt}", flush=True)

    if seed_ckpt_dir is not None:
        last_ep = len(episode_logs)
        final_bench = benchmark_evals.get(last_ep)
        final_comp = (
            compute_composite_benchmark_score(final_bench)
            if final_bench is not None
            else None
        )
        final_ckpt = save_training_checkpoint(
            path=seed_ckpt_dir / "final_model.pt",
            seed=seed,
            episode=last_ep,
            total_env_steps=total_env_steps,
            cfg=sys_cfg,
            agent=agent,
            tweaker=tweaker,
            balancer=balancer,
            orchestrator=orchestrator,
            benchmark_metrics=final_bench,
            composite_score=final_comp,
        )
        final_checkpoint_path = str(final_ckpt)
        if best_checkpoint_path is None:
            best_ckpt = save_training_checkpoint(
                path=seed_ckpt_dir / "best_model.pt",
                seed=seed,
                episode=last_ep,
                total_env_steps=total_env_steps,
                cfg=sys_cfg,
                agent=agent,
                tweaker=tweaker,
                balancer=balancer,
                orchestrator=orchestrator,
                benchmark_metrics=final_bench,
                composite_score=final_comp,
            )
            best_checkpoint_path = str(best_ckpt)

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
            f"Final EMA Win Rate: {orchestrator.win_evaluator.ema_win_rate:.1%}",
            flush=True,
        )
        if final_checkpoint_path is not None:
            print(
                f"    Final Model : {final_checkpoint_path}\n"
                f"    Best Model  : {best_checkpoint_path}\n",
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
        "best_composite_score": (
            float(best_composite_score) if best_composite_score > float("-inf") else None
        ),
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "periodic_checkpoint_paths": periodic_checkpoint_paths,
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Episode interval for periodic milestone checkpoints (defaults to config run.checkpoint_interval).",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable saving PyTorch model checkpoints (.pt) to disk.",
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
    checkpoint_interval = (
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else int(cfg.run.checkpoint_interval)
    )
    out_dir = Path(args.output_dir)
    ckpt_root_dir: Optional[Path] = (
        None if args.no_save_checkpoints else (out_dir / "checkpoints")
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
    if ckpt_root_dir is not None:
        ckpt_int_str = (
            f"Every {checkpoint_interval} episodes + Best + Final"
            if checkpoint_interval > 0
            else "Best + Final per seed"
        )
        print(f"  Checkpoints   : {ckpt_root_dir} ({ckpt_int_str})")
    else:
        print("  Checkpoints   : Disabled (--no-save-checkpoints)")
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
                checkpoint_dir=ckpt_root_dir,
                checkpoint_interval=checkpoint_interval,
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
                "checkpoint_dir": ckpt_root_dir,
                "checkpoint_interval": checkpoint_interval,
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

    out_dir.mkdir(parents=True, exist_ok=True)

    best_overall_ckpt_path: Optional[str] = None
    final_overall_ckpt_path: Optional[str] = None
    if ckpt_root_dir is not None and all_seed_results:
        scored_seeds = [
            r
            for r in all_seed_results
            if r.get("best_checkpoint_path") is not None
        ]
        if scored_seeds:
            best_seed_res = max(
                scored_seeds,
                key=lambda r: (
                    float(r["best_composite_score"])
                    if r.get("best_composite_score") is not None
                    else float("-inf")
                ),
            )
            src_best = Path(str(best_seed_res["best_checkpoint_path"]))
            if src_best.exists():
                dst_best = out_dir / "best_overall_model.pt"
                shutil.copy2(src_best, dst_best)
                best_overall_ckpt_path = str(dst_best)

            src_final = (
                Path(str(best_seed_res["final_checkpoint_path"]))
                if best_seed_res.get("final_checkpoint_path")
                else None
            )
            if src_final is not None and src_final.exists():
                dst_final = out_dir / "final_model.pt"
                shutil.copy2(src_final, dst_final)
                final_overall_ckpt_path = str(dst_final)

    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    total_time_str = format_time_duration(time.time() - global_start_time)
    print("=" * 80)
    print(f"  All {num_seeds} seeds completed in {total_time_str}!")
    print(f"  Summary saved to     : {summary_path}")
    if best_overall_ckpt_path is not None:
        print(f"  Best Overall Model   : {best_overall_ckpt_path}")
    if final_overall_ckpt_path is not None:
        print(f"  Final Overall Model  : {final_overall_ckpt_path}")
    if ckpt_root_dir is not None:
        print(f"  Per-Seed Checkpoints : {ckpt_root_dir}/seed_<N>/{{best_model.pt, final_model.pt}}")
    print("=" * 80, flush=True)

    return {
        "seeds": all_seed_results,
        "summary_path": str(summary_path),
        "best_overall_checkpoint_path": best_overall_ckpt_path,
        "final_overall_checkpoint_path": final_overall_ckpt_path,
    }


if __name__ == "__main__":
    main()
