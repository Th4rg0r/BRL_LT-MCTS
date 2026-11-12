"""Universal Agent Interface and Win Evaluation Module for Balanced RL (BRL)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Tuple
import numpy as np


class AbstractBRLSolverAgent(ABC):
    """Universal pluggable solver agent interface for the BRL framework."""

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Selects a continuous control action in [-1, 1]^N for the current step."""

    @abstractmethod
    def observe_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Stores an environment transition in the agent's internal replay buffer."""

    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """Executes a single gradient update step and returns training metrics."""

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, float]:
        """Returns internal value estimates V(s_0) and TD errors for Tweaker/Balancer."""


class WinEvaluator:
    """Evaluates episode duels between Agent and Environment/Tweaker.

    Computes binary win indicator W_e in {0, 1}, smooth dominance score S_e in (0, 1),
    and exponential moving average win rate W_bar_e.
    """

    def __init__(
        self,
        initial_r_ref: float = 1000.0,
        benchmark_r_ref: float = 6169.54,
        tau_s: float = 500.0,
        beta_w: float = 0.05,
        adaptive_ref_rate: float = 0.02,
        min_floor_ref: Optional[float] = None,
    ) -> None:
        self.initial_r_ref = float(initial_r_ref)
        self.benchmark_r_ref = float(benchmark_r_ref)
        self.r_ref = float(initial_r_ref)
        self.tau_s = float(tau_s)
        self.beta_w = float(beta_w)
        self.adaptive_ref_rate = float(adaptive_ref_rate)
        if min_floor_ref is not None:
            self.min_floor_ref = float(min_floor_ref)
        else:
            self.min_floor_ref = -max(2500.0, abs(self.initial_r_ref) * 2.5)
        self.ema_win_rate: float = 0.50
        self.episodes_evaluated: int = 0

    def evaluate_episode(
        self,
        episode_return: float,
        terminated_unhealthy: bool,
        forward_velocity: float = 0.0,
        adapt_reference: bool = True,
    ) -> Tuple[int, float, float]:
        """Evaluates whether the agent won the episode.

        Args:
            episode_return: Cumulative episode return R_e^agent.
            terminated_unhealthy: True if episode ended prematurely due to unhealthy state.
            forward_velocity: Mean forward x-velocity achieved in the episode.
            adapt_reference: Whether to adapt R_ref toward benchmark_r_ref when agent excels.

        Returns:
            win_indicator: W_e in {0, 1}.
            dominance_score: S_e in (0, 1).
            ema_win_rate: Updated EMA win rate W_bar_e in [0, 1].
        """
        # Forward locomotion requirement: agent must actively locomote forward (mean velocity >= 0.15 m/s)
        # to qualify for a win. Crouching or standing still (velocity < 0.15 m/s) is strictly a LOSS (0).
        is_moving = forward_velocity >= 0.15
        win_indicator = int(
            (not terminated_unhealthy)
            and is_moving
            and (episode_return >= self.r_ref)
        )

        # Smooth logistic dominance score S_e = sigma((R_e - R_ref) / tau_s)
        z = max(-30.0, min(30.0, (episode_return - self.r_ref) / max(self.tau_s, 1e-6)))
        dominance_score = 1.0 / (1.0 + math.exp(-z))
        if terminated_unhealthy or not is_moving:
            dominance_score *= 0.5

        self.ema_win_rate = (
            1.0 - self.beta_w
        ) * self.ema_win_rate + self.beta_w * float(win_indicator)
        self.episodes_evaluated += 1

        # Adapt reference threshold upward when agent consistently wins or achieves high velocity
        if adapt_reference and win_indicator == 1 and self.ema_win_rate > 0.52:
            target_ref = min(
                self.benchmark_r_ref,
                max(self.r_ref, episode_return * 0.90),
            )
            self.r_ref = (
                1.0 - self.adaptive_ref_rate
            ) * self.r_ref + self.adaptive_ref_rate * target_ref
        elif adapt_reference and win_indicator == 0 and self.ema_win_rate < 0.35:
            # Gently relax R_ref during early curriculum learning so duel stays informative
            target_floor = episode_return * 0.90 if episode_return < 0 else episode_return * 1.05
            floor_ref = min(self.initial_r_ref, max(self.min_floor_ref, target_floor))
            self.r_ref = (
                1.0 - self.adaptive_ref_rate
            ) * self.r_ref + self.adaptive_ref_rate * floor_ref

        return win_indicator, float(dominance_score), float(self.ema_win_rate)
