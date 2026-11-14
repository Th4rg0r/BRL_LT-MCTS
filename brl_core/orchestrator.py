"""Co-Evolutionary BRL Training Orchestrator coupling Agent, Tweaker, and Balancer."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
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
        self.last_perception_correctness: float = 1.0
        self.last_leg_length_exactness: float = 1.0
        self.last_obs_rmse: float = 0.0
        self.episode_history: List[Dict[str, Any]] = []
        self._warmup_returns: List[float] = []
        self._warmup_calibrated: bool = False

    def run_episode(
        self,
        episode_idx: int,
        max_steps: Optional[int] = None,
        train_agent_every: int = 10,
        step_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, Any]:
        """Executes a single co-evolutionary BRL episode (Warmup / BRL / Consolidation)."""
        brl_cfg = self.cfg.brl
        is_warmup = episode_idx <= brl_cfg.warmup_episodes
        is_consolidation = episode_idx > brl_cfg.brl_episodes

        # On transition out of warmup, reset EMA win rate to 50% equilibrium
        # and calibrate r_ref to the median of the last 20 warmup episode returns.
        if not is_warmup and not self._warmup_calibrated:
            self.win_evaluator.ema_win_rate = 0.50
            if self._warmup_returns:
                median_ret = float(np.median(self._warmup_returns[-20:]))
                self.win_evaluator.r_ref = float(
                    np.clip(
                        median_ret,
                        self.win_evaluator.min_floor_ref,
                        self.win_evaluator.benchmark_r_ref,
                    )
                )
            self._warmup_calibrated = True

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
        perceived_tweak = reset_info.get("perceived_tweak_vector", w_proj)
        limb_kinematics = reset_info.get("limb_kinematics", None)
        if hasattr(self.agent, "set_episode_context"):
            self.agent.set_episode_context(
                tweak_vector=perceived_tweak,
                action_mask=morph_mask,
                limb_kinematics=limb_kinematics,
            )

        # 4. Agent interacts with environment E(w_e)
        step_limit = max_steps or self.cfg.ant_morph.max_episode_steps
        ep_return = 0.0
        agent_ep_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_info: Dict[str, Any] = reset_info
        agent_metrics: Dict[str, float] = {}

        while steps < step_limit:
            action = self.agent.select_action(
                obs, deterministic=False, limb_kinematics=last_info.get("limb_kinematics")
            )
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.agent.observe_transition(
                obs, action, float(reward), next_obs, terminated, truncated, info
            )
            raw_step_reward = float(info.get("raw_env_reward", reward))
            ep_return += raw_step_reward
            agent_ep_return += float(reward)
            steps += 1
            obs = next_obs
            last_info = info

            if step_callback is not None:
                step_callback(steps, step_limit, ep_return)

            if steps % train_agent_every == 0:
                agent_metrics = self.agent.train_step()

            if terminated or truncated:
                break

        # Final episode agent update & Reanalyze (frequency/size configurable; 0 = disabled)
        agent_metrics = self.agent.train_step()
        reanalyze_every = int(getattr(self.cfg.la_mcts, "reanalyze_every", 2))
        reanalyze_states = int(getattr(self.cfg.la_mcts, "reanalyze_states", 4))
        if (
            hasattr(self.agent, "reanalyze_batch")
            and reanalyze_every > 0
            and reanalyze_states > 0
            and (is_consolidation or episode_idx % reanalyze_every == 0)
        ):
            self.agent.reanalyze_batch(num_states=reanalyze_states)

        # 5. Evaluate duel outcome via WinEvaluator
        brl_info = last_info.get("brl_metrics", {})
        perc_metrics = last_info.get("perception_metrics", {})
        if not perc_metrics and "perception_metrics" in brl_info:
            perc_metrics = brl_info.get("perception_metrics", {})

        p_corr = float(perc_metrics.get("perception_correctness", 1.0))
        leg_exact = float(perc_metrics.get("leg_length_exactness", 1.0))
        obs_exact = float(perc_metrics.get("obs_exactness", 1.0))
        obs_rmse = float(perc_metrics.get("obs_rmse", 0.0))

        unhealthy_term = bool(terminated and steps < step_limit)
        x_vel = float(brl_info.get("x_velocity", ep_return / max(steps, 1)))
        if hasattr(self.env, "update_velocity_curriculum"):
            self.env.update_velocity_curriculum(
                x_vel,
                ep_steps=steps,
                required_steps=step_limit,
                terminated_unhealthy=unhealthy_term,
            )
        if is_warmup:
            self._warmup_returns.append(float(ep_return))
            if len(self._warmup_returns) > 20:
                self._warmup_returns = self._warmup_returns[-20:]
        win_ind, dom_score, ema_win = self.win_evaluator.evaluate_episode(
            episode_return=ep_return,
            terminated_unhealthy=unhealthy_term,
            forward_velocity=x_vel,
        )

        # Update rolling stats
        self.mean_x_vel = 0.9 * self.mean_x_vel + 0.1 * x_vel
        self.termination_rate = 0.9 * self.termination_rate + 0.1 * float(unhealthy_term)
        self.energy_cost = float(brl_info.get("energy_cost", 0.0))
        self.last_perception_correctness = 0.9 * self.last_perception_correctness + 0.1 * p_corr
        self.last_leg_length_exactness = 0.9 * self.last_leg_length_exactness + 0.1 * leg_exact
        self.last_obs_rmse = 0.9 * self.last_obs_rmse + 0.1 * obs_rmse
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

        if len(w_proj) >= 18:
            mean_phys_tweak = float(np.mean(w_proj[:12]))
            mean_perc_tweak = float(np.mean(w_proj[12:18]))
        else:
            mean_phys_tweak = float(np.mean(w_proj))
            mean_perc_tweak = 0.0

        summary = {
            "episode": episode_idx,
            "episode_return": ep_return,
            "agent_episode_return": agent_ep_return,
            "steps": steps,
            "total_distance": float(brl_info.get("total_distance", 0.0)),
            "x_velocity": float(brl_info.get("x_velocity", 0.0)),
            "target_vel": float(getattr(self.env, "target_vel", brl_info.get("target_vel", 0.0))),
            "current_vel": float(getattr(self.env, "current_vel", brl_info.get("current_vel", 0.0))),
            "reward_blend_alpha": float(brl_info.get("reward_blend_alpha", 0.0)),
            "win_indicator": win_ind,
            "dominance_score": dom_score,
            "ema_win_rate": ema_win,
            "r_ref": self.win_evaluator.r_ref,
            "budget": budget,
            "mean_limit": float(np.mean(limits)),
            "mean_cost": float(np.mean(costs)),
            "mean_tweak": float(np.mean(w_proj)),
            "mean_physics_tweak": mean_phys_tweak,
            "mean_perception_tweak": mean_perc_tweak,
            "perception_correctness": p_corr,
            "leg_length_exactness": leg_exact,
            "obs_exactness": obs_exact,
            "obs_rmse": obs_rmse,
            "tweaker_reward": twk_reward,
            "balancer_reward": bal_reward,
            **agent_metrics,
            **twk_metrics,
            **bal_metrics,
        }
        self.episode_history.append(summary)
        return summary
