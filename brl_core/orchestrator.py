"""Co-Evolutionary BRL Training Orchestrator coupling Agent, Tweaker, and Balancer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from brl_core.balancer import DDPGBalancer
from brl_core.interfaces import AbstractBRLSolverAgent, WinEvaluator
from brl_core.tweaker import StochasticTweakerActorCritic
from config import SystemConfig


class BRLTrainingOrchestrator:
    """Coordinates the 3-phase co-evolutionary training loop across Agent, Tweaker, and Balancer."""

    def __init__(
        self,
        env: Any,
        agent: AbstractBRLSolverAgent,
        tweaker: StochasticTweakerActorCritic,
        balancer: DDPGBalancer,
        config: Optional[SystemConfig] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.tweaker = tweaker
        self.balancer = balancer
        self.cfg = config or SystemConfig()

        self.win_evaluator = WinEvaluator(
            initial_r_ref=self.cfg.brl.initial_r_ref,
            benchmark_r_ref=self.cfg.brl.benchmark_r_ref,
            tau_s=self.cfg.brl.tau_s,
            beta_w=self.cfg.brl.ema_beta_w,
        )

        self.last_win: int = 0
        self.last_dominance: float = 0.5
        self.mean_x_vel: float = 0.0
        self.termination_rate: float = 0.0
        self.energy_cost: float = 0.0
        self.episode_history: List[Dict[str, Any]] = []

    def run_episode(
        self,
        episode_idx: int,
        max_steps: Optional[int] = None,
        train_agent_every: int = 10,
    ) -> Dict[str, Any]:
        """Executes a single co-evolutionary BRL episode (Warmup / BRL / Consolidation)."""
        brl_cfg = self.cfg.brl
        is_warmup = episode_idx <= brl_cfg.warmup_episodes
        is_consolidation = episode_idx > brl_cfg.brl_episodes

        # 1. Balancer emits constraints (L_e, B_e, C_e)
        bal_state = self.balancer.build_state(
            ema_win_rate=self.win_evaluator.ema_win_rate,
            last_win=self.last_win,
            dominance_score=self.last_dominance,
        )
        limits, budget, costs, bal_raw_act = self.balancer.select_constraints(
            state=bal_state,
            warmup=is_warmup,
            deterministic=is_consolidation,
        )

        # 2. Tweaker samples valid parameter vector w_e under Knapsack constraints
        rho_fail = self.balancer.replay_buffer.get_failure_correlations()
        twk_state = self.tweaker.build_state(
            limits=limits,
            budget=budget,
            costs=costs,
            param_sensitivity=rho_fail,
            mean_x_vel=self.mean_x_vel,
            termination_rate=self.termination_rate,
            energy_cost=self.energy_cost,
            ema_win_rate=self.win_evaluator.ema_win_rate,
        )
        w_proj, w_raw, u_raw, log_p, twk_val = self.tweaker.sample_tweak(
            state=twk_state,
            limits=limits,
            budget=budget,
            costs=costs,
            deterministic=is_consolidation,
        )

        # 3. Reset environment with tweak_vector w_e
        obs, reset_info = self.env.reset(options={"tweak_vector": w_proj})
        morph_mask = reset_info.get(
            "morphology_mask",
            np.ones(self.cfg.la_mcts.max_action_dim, dtype=np.float32),
        )
        if hasattr(self.agent, "set_episode_context"):
            self.agent.set_episode_context(
                tweak_vector=w_proj, action_mask=morph_mask
            )

        # 4. Agent interacts with environment E(w_e)
        step_limit = max_steps or self.cfg.ant_morph.max_episode_steps
        ep_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = reset_info
        agent_metrics: Dict[str, float] = {}

        while steps < step_limit:
            action = self.agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.agent.observe_transition(
                obs, action, float(reward), next_obs, terminated, truncated, info
            )
            ep_return += float(reward)
            steps += 1
            obs = next_obs
            last_info = info

            if steps % train_agent_every == 0:
                agent_metrics = self.agent.train_step()

            if terminated or truncated:
                break

        # Final episode agent update & Reanalyze
        agent_metrics = self.agent.train_step()
        if hasattr(self.agent, "reanalyze_batch") and (
            is_consolidation or episode_idx % 2 == 0
        ):
            self.agent.reanalyze_batch(num_states=4)

        # 5. Evaluate duel outcome via WinEvaluator
        brl_info = last_info.get("brl_metrics", {})
        unhealthy_term = bool(terminated and steps < step_limit)
        x_vel = float(brl_info.get("x_velocity", ep_return / max(steps, 1)))
        win_ind, dom_score, ema_win = self.win_evaluator.evaluate_episode(
            episode_return=ep_return,
            terminated_unhealthy=unhealthy_term,
            forward_velocity=x_vel,
        )

        # Update rolling stats
        self.mean_x_vel = 0.9 * self.mean_x_vel + 0.1 * x_vel
        self.termination_rate = 0.9 * self.termination_rate + 0.1 * float(unhealthy_term)
        self.energy_cost = float(brl_info.get("energy_cost", 0.0))
        self.last_win = win_ind
        self.last_dominance = dom_score
        self.balancer.last_tweak = w_proj.copy()
        self.balancer.replay_buffer.record_outcome(
            episode_return=ep_return, tweak_vector=w_proj, failed=unhealthy_term
        )

        # 6. Update Tweaker and Balancer
        twk_reward = self.tweaker.record_episode(
            state=twk_state,
            u_raw=u_raw,
            w_raw=w_raw,
            w_proj=w_proj,
            limits=limits,
            budget=budget,
            costs=costs,
            win_indicator=win_ind,
            dominance_score=dom_score,
            log_prob=log_p,
            value=twk_val,
        )
        twk_metrics = self.tweaker.update(ppo_epochs=2)

        bal_reward = self.balancer.compute_reward(
            ema_win_rate=ema_win,
            dominance_score=dom_score,
            w_e=w_proj,
            costs_e=costs,
        )
        next_bal_state = self.balancer.build_state(
            ema_win_rate=ema_win,
            last_win=win_ind,
            dominance_score=dom_score,
        )
        self.balancer.replay_buffer.add_transition(
            state=bal_state,
            raw_action=bal_raw_act,
            reward=bal_reward,
            next_state=next_bal_state,
        )
        bal_metrics = self.balancer.update(batch_size=16)

        summary = {
            "episode": episode_idx,
            "episode_return": ep_return,
            "steps": steps,
            "win_indicator": win_ind,
            "dominance_score": dom_score,
            "ema_win_rate": ema_win,
            "r_ref": self.win_evaluator.r_ref,
            "budget": budget,
            "mean_limit": float(np.mean(limits)),
            "mean_cost": float(np.mean(costs)),
            "mean_tweak": float(np.mean(w_proj)),
            "tweaker_reward": twk_reward,
            "balancer_reward": bal_reward,
            **agent_metrics,
            **twk_metrics,
            **bal_metrics,
        }
        self.episode_history.append(summary)
        return summary
