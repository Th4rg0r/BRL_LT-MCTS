"""End-to-End Training and Benchmark Verification Script for LA-MCTS + BRL on Unitree Go1 Quadruped Dog."""

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

from brl_core.balancer import DDPGBalancer
from brl_core.orchestrator import BRLTrainingOrchestrator
from brl_core.tweaker import StochasticTweakerActorCritic
from config import SystemConfig, config_from_dict, config_to_dict, load_config
from dog_morph_env.wrapper import (
    MorphologicalDogBRLWrapper,
    evaluate_dog_benchmark_suite,
)
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
    """Computes scalar benchmark score from evaluate_dog_benchmark_suite output."""
    std_key = "Standard-Go1"
    std_ret = float(bench_results.get(std_key, {}).get("return", 0.0))
    ood_keys = [k for k in bench_results.keys() if k != std_key]
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
    """Saves a complete, self-contained PyTorch checkpoint (.pt) for Quadruped Dog."""
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
            "recent_returns": list(balancer.replay_buffer.recent_returns),
            "recent_failures": list(balancer.replay_buffer.recent_failures),
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


def save_progress_metadata(
    path: str | Path,
    metadata: Dict[str, Any],
) -> Path:
    """Saves atomic progress metadata JSON for crash recovery and status tracking."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    tmp_path.replace(target_path)
    return target_path


def load_progress_metadata(path: str | Path) -> Dict[str, Any]:
    """Loads saved progress metadata JSON if present."""
    target_path = Path(path)
    if not target_path.exists():
        return {}
    try:
        with target_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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
    """Loads a saved Dog `.pt` checkpoint and returns instantiated objects."""
    ckpt_path = Path(path)
    payload: Dict[str, Any] = torch.load(
        ckpt_path, map_location=torch.device(device), weights_only=False
    )
    cfg = config_from_dict(payload.get("config", {}))
    la_cfg = replace(
        cfg.la_mcts,
        max_action_dim=cfg.dog_morph.max_action_capacity,
        obs_dim=cfg.la_mcts.obs_dim,
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
    resume: bool = False,
    step_milestones: Optional[List[int]] = None,
    log_file: Optional[str | Path] = None,
    progress_jsonl_file: Optional[str | Path] = None,
    parallel_mode: bool = False,
) -> Dict[str, Any]:
    """Runs the 3-phase BRL + LA-MCTS training pipeline for Dog for a single seed."""
    set_global_seed(seed)
    seed_start_time = time.time()

    log_path: Optional[Path] = Path(log_file) if log_file is not None else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path: Optional[Path] = Path(progress_jsonl_file) if progress_jsonl_file is not None else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def log_line(msg: str, to_console: bool = True) -> None:
        if log_path is not None:
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    f.write(f"[{ts}] {msg}\n")
            except Exception:
                pass
        if to_console and verbose:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()

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

    if verbose and not parallel_mode:
        print("\n" + "=" * 80)
        print(f"  >>> QUADRUPED DOG SEED {seed_idx + 1}/{total_seeds} (Seed: {seed}) | {episodes} Episodes on {device}")
        print("=" * 80, flush=True)

    la_cfg = replace(
        cfg.la_mcts,
        max_action_dim=cfg.dog_morph.max_action_capacity,
    )

    nominal_max_steps = cfg.dog_morph.max_episode_steps
    step_scale = max_steps / float(nominal_max_steps) if nominal_max_steps > 0 else 1.0

    brl_cfg = cfg.brl
    if step_scale != 1.0:
        brl_cfg = replace(
            brl_cfg,
            initial_r_ref=max(10.0, cfg.brl.initial_r_ref * step_scale),
            benchmark_r_ref=max(50.0, cfg.brl.benchmark_r_ref * step_scale),
            tau_s=max(10.0, cfg.brl.tau_s * step_scale),
        )

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

    env = MorphologicalDogBRLWrapper(
        max_episode_steps=max_steps,
        healthy_z_range=sys_cfg.dog_morph.healthy_z_range,
        max_action_capacity=sys_cfg.dog_morph.max_action_capacity,
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

    seed_ckpt_dir: Optional[Path] = None
    if checkpoint_dir is not None:
        seed_ckpt_dir = Path(checkpoint_dir) / f"seed_{seed}"
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_logs: List[Dict[str, Any]] = []
    benchmark_evals: Dict[int, Dict[str, Dict[str, float]]] = {}
    completed_episodes = 0
    total_env_steps = 0
    resume_count = 0
    is_resumed = False
    step_milestones_reached: set[int] = set()
    periodic_checkpoint_paths: List[str] = []
    milestone_checkpoint_paths: Dict[str, str] = {}
    best_composite_score = float("-inf")
    best_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    pruned = False
    mem_exceeded = False

    if resume and seed_ckpt_dir is not None:
        prog_f = seed_ckpt_dir / "progress.json"
        prev_prog = load_progress_metadata(prog_f)

        ckpt_to_resume: Optional[Path] = None
        latest_pt = seed_ckpt_dir / "latest_checkpoint.pt"
        if latest_pt.exists():
            ckpt_to_resume = latest_pt
        else:
            ep_ckpts = sorted(
                seed_ckpt_dir.glob("checkpoint_ep_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
            )
            if ep_ckpts:
                ckpt_to_resume = ep_ckpts[-1]
            elif (seed_ckpt_dir / "final_model.pt").exists():
                ckpt_to_resume = seed_ckpt_dir / "final_model.pt"
            elif (seed_ckpt_dir / "best_model.pt").exists():
                ckpt_to_resume = seed_ckpt_dir / "best_model.pt"

        if ckpt_to_resume is not None:
            is_resumed = True
            resume_count = int(prev_prog.get("resume_count", 0)) + 1
            step_milestones_reached = set(prev_prog.get("step_milestones_reached", []))

            _, _, _, _, payload = load_training_checkpoint(ckpt_to_resume, device=device)
            completed_episodes = int(payload.get("episode", 0))
            total_env_steps = int(payload.get("total_env_steps", 0))
            if payload.get("composite_score") is not None:
                best_composite_score = float(payload["composite_score"])
            if (seed_ckpt_dir / "best_model.pt").exists():
                best_checkpoint_path = str(seed_ckpt_dir / "best_model.pt")

            if "agent_state_dict" in payload:
                agent.load_state_dict(payload["agent_state_dict"])
            if "tweaker_state_dict" in payload:
                tweaker.load_state_dict(payload["tweaker_state_dict"])
            if "tweaker_optimizer_state_dict" in payload:
                tweaker.optimizer.load_state_dict(payload["tweaker_optimizer_state_dict"])
            if "balancer_state_dict" in payload:
                balancer.load_state_dict(payload["balancer_state_dict"])
            if "balancer_actor_optimizer_state_dict" in payload:
                balancer.actor_optimizer.load_state_dict(payload["balancer_actor_optimizer_state_dict"])
            if "balancer_critic_optimizer_state_dict" in payload:
                balancer.critic_optimizer.load_state_dict(payload["balancer_critic_optimizer_state_dict"])

            if "win_evaluator" in payload:
                we_d = payload["win_evaluator"]
                orchestrator.win_evaluator.ema_win_rate = float(we_d.get("ema_win_rate", orchestrator.win_evaluator.ema_win_rate))
                orchestrator.win_evaluator.r_ref = float(we_d.get("r_ref", orchestrator.win_evaluator.r_ref))
                orchestrator.win_evaluator.episodes_evaluated = int(we_d.get("episodes_evaluated", orchestrator.win_evaluator.episodes_evaluated))

            if "balancer_meta" in payload:
                bm = payload["balancer_meta"]
                if "last_limits" in bm:
                    balancer.last_limits = np.array(bm["last_limits"], dtype=np.float32)
                if "last_budget" in bm:
                    balancer.last_budget = float(bm["last_budget"])
                if "last_costs" in bm:
                    balancer.last_costs = np.array(bm["last_costs"], dtype=np.float32)
                if "last_tweak" in bm:
                    balancer.last_tweak = np.array(bm["last_tweak"], dtype=np.float32)
                if "recent_returns" in bm:
                    balancer.replay_buffer.recent_returns.extend(bm["recent_returns"])
                if "recent_failures" in bm:
                    balancer.replay_buffer.recent_failures.extend(bm["recent_failures"])

    start_ep = completed_episodes + 1
    if start_ep > episodes:
        skip_msg = (
            f"[Dog Seed {seed} ({seed_idx + 1}/{total_seeds})] Already completed {completed_episodes}/{episodes} episodes "
            f"({total_env_steps:,} total env steps). Target reached; skipping."
        )
        log_line(skip_msg, to_console=True)
        env.close()
        final_pt = str(seed_ckpt_dir / "final_model.pt") if (seed_ckpt_dir and (seed_ckpt_dir / "final_model.pt").exists()) else None
        return {
            "seed": seed,
            "episodes": episode_logs,
            "benchmarks": benchmark_evals,
            "final_ema_win_rate": orchestrator.win_evaluator.ema_win_rate,
            "param_count": param_count,
            "peak_gpu_mem_mb": 0.0,
            "total_env_steps": total_env_steps,
            "total_wall_sec": 0.0,
            "sec_per_step": 0.0,
            "pruned": False,
            "mem_exceeded": False,
            "best_composite_score": (
                float(best_composite_score) if best_composite_score > float("-inf") else None
            ),
            "best_checkpoint_path": best_checkpoint_path,
            "final_checkpoint_path": final_pt,
            "periodic_checkpoint_paths": periodic_checkpoint_paths,
            "milestone_checkpoint_paths": milestone_checkpoint_paths,
            "completed_episodes": completed_episodes,
            "target_episodes": episodes,
            "is_resumed": True,
            "resume_count": resume_count,
            "skipped": True,
        }

    if is_resumed:
        resume_msg = (
            f">>> [Dog Seed {seed} ({seed_idx + 1}/{total_seeds})] Resuming at Episode {start_ep}/{episodes} "
            f"({episodes - completed_episodes} remaining, {total_env_steps:,} steps already taken) [Resume #{resume_count}]"
        )
        log_line(resume_msg, to_console=True)

    def _save_current_progress(curr_ep: int, latest_path: Optional[str] = None) -> None:
        if seed_ckpt_dir is None:
            return
        meta = {
            "seed": int(seed),
            "completed_episodes": int(curr_ep),
            "target_episodes": int(episodes),
            "remaining_episodes": max(0, int(episodes - curr_ep)),
            "total_env_steps": int(total_env_steps),
            "is_resumed": bool(is_resumed),
            "resume_count": int(resume_count),
            "best_composite_score": (
                float(best_composite_score) if best_composite_score > float("-inf") else None
            ),
            "best_checkpoint_path": best_checkpoint_path,
            "latest_checkpoint_path": latest_path or (str(seed_ckpt_dir / "latest_checkpoint.pt") if (seed_ckpt_dir / "latest_checkpoint.pt").exists() else None),
            "step_milestones_reached": sorted(list(step_milestones_reached)),
            "last_ema_win_rate": float(orchestrator.win_evaluator.ema_win_rate),
            "last_r_ref": float(orchestrator.win_evaluator.r_ref),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_progress_metadata(seed_ckpt_dir / "progress.json", meta)

    train_every = (
        int(sys_cfg.la_mcts.train_every_steps)
        if int(sys_cfg.la_mcts.train_every_steps) > 0
        else max(1, max_steps // 2)
    )

    for ep in range(start_ep, episodes + 1):
        ep_start_time = time.time()

        prev_env_steps = total_env_steps
        summary = orchestrator.run_episode(
            episode_idx=ep,
            max_steps=max_steps,
            train_agent_every=train_every,
        )
        episode_logs.append(summary)
        ep_steps = int(summary.get("steps", 0))
        total_env_steps += ep_steps

        ep_duration = time.time() - ep_start_time
        session_elapsed = max(1e-6, time.time() - seed_start_time)
        session_eps_done = ep - start_ep + 1
        avg_ep_duration = session_elapsed / session_eps_done
        remaining_eps = episodes - ep
        seed_eta = remaining_eps * avg_ep_duration

        if use_cuda and max_gpu_mem_gb > 0.0:
            peak_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
            if peak_gb > max_gpu_mem_gb:
                mem_exceeded = True
                pruned = True
                break

        win_tag = "WIN " if summary["win_indicator"] else "LOSS"
        loss_val = summary.get("loss_total", 0.0)
        loss_str = f"{loss_val:5.2f}" if loss_val != 0.0 else "  N/A"

        line = (
            f"[Dog Seed {seed:2d} ({seed_idx + 1}/{total_seeds}) | Ep {ep:4d}/{episodes} | {total_env_steps:7,d} steps] "
            f"Ret: {summary['episode_return']:+6.1f} | WinRate: {summary['ema_win_rate']:4.1%} [{win_tag}] | "
            f"Loss: {loss_str} | EpTime: {format_time_duration(ep_duration)} | ETA: {format_time_duration(seed_eta)}"
        )
        log_line(line, to_console=True)

        # Check step milestones
        if step_milestones:
            for ms in step_milestones:
                if ms not in step_milestones_reached and prev_env_steps < ms <= total_env_steps:
                    step_milestones_reached.add(ms)
                    ms_label = f"{ms // 1_000_000}M" if ms >= 1_000_000 else f"{ms // 1_000}k" if ms >= 1000 else str(ms)
                    ms_banner = (
                        f"\n{'=' * 80}\n"
                        f">>> [Dog Seed {seed} ({seed_idx + 1}/{total_seeds}) | Ep {ep}] "
                        f"REACHED STEP MILESTONE: {ms:,} Environment Steps ({ms_label})!\n"
                        f"{'=' * 80}"
                    )
                    log_line(ms_banner, to_console=True)
                    ms_bench = evaluate_dog_benchmark_suite(
                        agent=agent,
                        env=env,
                        max_steps=max_steps,
                        verbose=(verbose and not parallel_mode),
                        log_fn=lambda s: log_line(s, to_console=False),
                    )
                    ms_score = compute_composite_benchmark_score(ms_bench)
                    benchmark_evals[ep] = ms_bench
                    if seed_ckpt_dir is not None:
                        ms_ckpt = save_training_checkpoint(
                            path=seed_ckpt_dir / f"checkpoint_step_{ms_label}.pt",
                            seed=seed,
                            episode=ep,
                            total_env_steps=total_env_steps,
                            cfg=sys_cfg,
                            agent=agent,
                            tweaker=tweaker,
                            balancer=balancer,
                            orchestrator=orchestrator,
                            benchmark_metrics=ms_bench,
                            composite_score=ms_score,
                        )
                        milestone_checkpoint_paths[str(ms)] = str(ms_ckpt)
                        log_line(f"    [Checkpoint] Saved Step Milestone ({ms_label}) -> {ms_ckpt}", to_console=True)

        if ep % eval_interval == 0 or ep == episodes:
            eval_suite = evaluate_dog_benchmark_suite(
                agent=agent,
                env=env,
                max_steps=max_steps,
                verbose=(verbose and not parallel_mode),
                log_fn=lambda s: log_line(s, to_console=False),
            )
            benchmark_evals[ep] = eval_suite
            cur_comp_score = compute_composite_benchmark_score(eval_suite)
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
                    benchmark_metrics=eval_suite,
                    composite_score=cur_comp_score,
                )
                best_checkpoint_path = str(best_ckpt)

        latest_path_str = None
        if seed_ckpt_dir is not None:
            latest_ckpt = save_training_checkpoint(
                path=seed_ckpt_dir / "latest_checkpoint.pt",
                seed=seed,
                episode=ep,
                total_env_steps=total_env_steps,
                cfg=sys_cfg,
                agent=agent,
                tweaker=tweaker,
                balancer=balancer,
                orchestrator=orchestrator,
                benchmark_metrics=benchmark_evals.get(ep),
                composite_score=best_composite_score if best_composite_score > float("-inf") else None,
            )
            latest_path_str = str(latest_ckpt)

            if checkpoint_interval > 0 and (ep % checkpoint_interval == 0 or ep == episodes):
                ep_ckpt = save_training_checkpoint(
                    path=seed_ckpt_dir / f"checkpoint_ep_{ep:05d}.pt",
                    seed=seed,
                    episode=ep,
                    total_env_steps=total_env_steps,
                    cfg=sys_cfg,
                    agent=agent,
                    tweaker=tweaker,
                    balancer=balancer,
                    orchestrator=orchestrator,
                    benchmark_metrics=benchmark_evals.get(ep),
                    composite_score=best_composite_score if best_composite_score > float("-inf") else None,
                )
                periodic_checkpoint_paths.append(str(ep_ckpt))

        _save_current_progress(ep, latest_path_str)

    total_wall_sec = time.time() - seed_start_time
    sec_per_step = total_wall_sec / max(1, total_env_steps)
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024.0**2) if use_cuda else 0.0
    )

    if seed_ckpt_dir is not None:
        final_ckpt = save_training_checkpoint(
            path=seed_ckpt_dir / "final_model.pt",
            seed=seed,
            episode=episodes,
            total_env_steps=total_env_steps,
            cfg=sys_cfg,
            agent=agent,
            tweaker=tweaker,
            balancer=balancer,
            orchestrator=orchestrator,
            benchmark_metrics=benchmark_evals.get(episodes),
            composite_score=best_composite_score if best_composite_score > float("-inf") else None,
        )
        final_checkpoint_path = str(final_ckpt)
        _save_current_progress(episodes, final_checkpoint_path)

    env.close()
    return {
        "seed": seed,
        "episodes": episode_logs,
        "benchmarks": benchmark_evals,
        "final_ema_win_rate": orchestrator.win_evaluator.ema_win_rate,
        "param_count": param_count,
        "peak_gpu_mem_mb": peak_gpu_mb,
        "total_env_steps": total_env_steps,
        "total_wall_sec": total_wall_sec,
        "sec_per_step": sec_per_step,
        "pruned": pruned,
        "mem_exceeded": mem_exceeded,
        "best_composite_score": (
            float(best_composite_score) if best_composite_score > float("-inf") else None
        ),
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "periodic_checkpoint_paths": periodic_checkpoint_paths,
        "milestone_checkpoint_paths": milestone_checkpoint_paths,
        "completed_episodes": episodes,
        "target_episodes": episodes,
        "is_resumed": is_resumed,
        "resume_count": resume_count,
    }


def _parallel_seed_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level helper for ProcessPoolExecutor parallel seed execution."""
    import os
    import traceback

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    seed = kwargs.get("seed", "?")
    try:
        return run_single_seed(**kwargs)
    except Exception as exc:
        tb = traceback.format_exc()
        err_msg = f"\n[FATAL ERROR in Worker for Seed {seed}]: {exc}\n{tb}\n"
        sys.stderr.write(err_msg)
        sys.stderr.flush()
        raise


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Train and verify LA-MCTS + BRL on Morphological Quadruped Dog (Unitree Go1)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dog_optimal_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
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
        default=None,
        help="Compute device (cpu or cuda).",
    )
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=None,
        help="Number of seeds to execute concurrently.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dog_validation",
        help="Directory to write checkpoints and summary JSON.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Episode interval for periodic checkpoints.",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Disable saving PyTorch model checkpoints to disk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint.",
    )
    parser.add_argument(
        "--step-milestones",
        type=str,
        default="1000000,3000000,10000000",
        help="Comma-separated cumulative step milestones.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write per-seed log files.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.n_sim is not None:
        cfg = replace(
            cfg,
            la_mcts=replace(cfg.la_mcts, n_sim=args.n_sim, n_sim_eval=args.n_sim),
        )

    num_seeds = args.seeds if args.seeds is not None else (cfg.run.seeds if cfg.run.seeds > 0 else 3)
    seed_offset = cfg.run.seed_offset if cfg.run.seed_offset > 0 else 42
    num_episodes = args.episodes if args.episodes is not None else (cfg.run.episodes if cfg.run.episodes > 0 else cfg.brl.total_episodes)
    max_steps = args.max_steps if args.max_steps is not None else (cfg.run.max_steps if cfg.run.max_steps > 0 else cfg.dog_morph.max_episode_steps)
    device = args.device if args.device is not None else (cfg.run.device or "cpu")
    parallel_seeds = args.parallel_seeds if args.parallel_seeds is not None else max(1, int(cfg.run.parallel_seeds))
    eval_interval = cfg.run.eval_interval if cfg.run.eval_interval > 0 and args.episodes is None else max(1, min(10, num_episodes))
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else int(cfg.run.checkpoint_interval)

    out_dir = Path(args.output_dir)
    ckpt_root_dir: Optional[Path] = None if args.no_save_checkpoints else (out_dir / "checkpoints")
    log_dir = Path(args.log_dir) if args.log_dir is not None else (out_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    step_milestones: List[int] = []
    if args.step_milestones:
        for s in args.step_milestones.split(","):
            s = s.strip()
            if s:
                try:
                    step_milestones.append(int(s))
                except ValueError:
                    pass
    step_milestones.sort()

    print("=" * 80)
    print("  QUADRUPED DOG (Unitree Go1) BRL + LA-MCTS Training Pipeline")
    print(f"  Config        : {args.config}")
    print(f"  Device        : {device}")
    print(f"  Seeds         : {num_seeds} ({seed_offset} .. {seed_offset + num_seeds - 1}) [parallel={parallel_seeds}]")
    print(f"  Episodes/Seed : {num_episodes}")
    print(f"  Max Steps/Ep  : {max_steps}")
    print(f"  MCTS Sims/Step: {cfg.la_mcts.n_sim}")
    print(f"  Eval Interval : Every {eval_interval} episodes")
    print(f"  Resume Mode   : {'Enabled (--resume)' if args.resume else 'Disabled (Fresh Run)'}")
    print(f"  Output Dir    : {args.output_dir}")
    print(f"  Logs Dir      : {log_dir}")
    print("=" * 80, flush=True)

    seed_list = [seed_offset + i for i in range(num_seeds)]
    tasks = []
    for idx, s in enumerate(seed_list):
        tasks.append({
            "seed": s,
            "cfg": cfg,
            "episodes": num_episodes,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "device": device,
            "seed_idx": idx,
            "total_seeds": num_seeds,
            "verbose": True,
            "checkpoint_dir": ckpt_root_dir,
            "checkpoint_interval": checkpoint_interval,
            "resume": args.resume,
            "step_milestones": step_milestones,
            "log_file": log_dir / f"seed_{s}.log",
            "progress_jsonl_file": log_dir / f"progress_seed_{s}.jsonl",
            "parallel_mode": parallel_seeds > 1,
        })

    all_results: List[Dict[str, Any]] = []
    if parallel_seeds > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=parallel_seeds) as executor:
            future_to_seed = {executor.submit(_parallel_seed_worker, t): t["seed"] for t in tasks}
            for fut in as_completed(future_to_seed):
                sd = future_to_seed[fut]
                try:
                    res = fut.result()
                    all_results.append(res)
                except Exception as exc:
                    print(f"Seed {sd} failed with exception: {exc}")
    else:
        for t in tasks:
            res = run_single_seed(**t)
            all_results.append(res)

    summary_file = out_dir / "dog_training_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({"seeds": [r["seed"] for r in all_results], "results": all_results}, f, indent=2)

    print(f"\n[DONE] Quadruped Dog training complete! Results saved to {summary_file}")
    return {"all_results": all_results}


if __name__ == "__main__":
    main()
