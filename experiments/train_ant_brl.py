"""End-to-End Training and Benchmark Verification Script for LA-MCTS + BRL on Ant-v4."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any, Dict, List
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


def evaluate_benchmark_suite(
    agent: LatentActionMCTSAgent,
    env: MorphologicalAntBRLWrapper,
    max_steps: int = 200,
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

    return results


def run_single_seed(
    seed: int,
    cfg: SystemConfig,
    episodes: int,
    max_steps: int,
    eval_interval: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs the 3-phase BRL + LA-MCTS training pipeline for a single random seed."""
    set_global_seed(seed)

    # Use max_action_capacity = 12 so the agent seamlessly handles 3-, 4-, and 6-legged creatures
    la_cfg = replace(
        cfg.la_mcts,
        max_action_dim=cfg.ant_morph.max_action_capacity,
    )
    sys_cfg = replace(cfg, la_mcts=la_cfg)

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
        summary = orchestrator.run_episode(
            episode_idx=ep,
            max_steps=max_steps,
            train_agent_every=max(1, max_steps // 2),
        )
        episode_logs.append(summary)

        if ep % eval_interval == 0 or ep == episodes:
            benchmark_evals[ep] = evaluate_benchmark_suite(
                agent=agent, env=env, max_steps=max_steps
            )

    env.close()
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
        )
        all_seed_results.append(res)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    return {"seeds": all_seed_results, "summary_path": str(summary_path)}


if __name__ == "__main__":
    main()
