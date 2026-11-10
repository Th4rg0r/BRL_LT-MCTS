"""Environment Test Tweaker: Stochastic Actor-Critic with Beta Policy and Knapsack Projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim

from brl_core.knapsack_proj import DifferentiableKnapsackProjection
from config import BRLConfig


@dataclass
class TweakerTransition:
    """Stores an episode-level Tweaker transition."""

    state: np.ndarray
    u_raw: np.ndarray
    w_raw: np.ndarray
    w_proj: np.ndarray
    limits: np.ndarray
    budget: float
    costs: np.ndarray
    reward: float
    log_prob: float
    value: float


class StochasticTweakerActorCritic(nn.Module):
    """Stochastic Actor-Critic (PPO/SAC-Hybrid with Beta Policy) generating w_e in [0, 1000]^P."""

    def __init__(
        self,
        config: Optional[BRLConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.cfg = config or BRLConfig()
        self.device = torch.device(device)
        self.num_params = self.cfg.num_params
        # State dimension: 3P + 5 (normalized L_e, B_e, C_e + 4 agent metrics)
        # Specifically: L_e (P), B_e (1), C_e (P), sensitivity (P), plus 4 scalar metrics = 3P + 5
        self.state_dim = 3 * self.num_params + 5

        hidden_dim = 256
        self.actor_backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # Beta concentration parameters alpha > 1, beta > 1
        self.alpha_head = nn.Linear(hidden_dim, self.num_params)
        self.beta_head = nn.Linear(hidden_dim, self.num_params)

        self.critic_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.tweaker_lr)
        self.memory: List[TweakerTransition] = []

    def build_state(
        self,
        limits: np.ndarray,
        budget: float,
        costs: np.ndarray,
        param_sensitivity: np.ndarray,
        mean_x_vel: float = 0.0,
        termination_rate: float = 0.0,
        energy_cost: float = 0.0,
        ema_win_rate: float = 0.5,
    ) -> np.ndarray:
        """Constructs the Tweaker state vector s_e^twk in R^(3P + 5)."""
        norm_l = np.asarray(limits, dtype=np.float32) / self.cfg.param_max
        norm_b = np.array([float(budget) / self.cfg.budget_max], dtype=np.float32)
        norm_c = np.asarray(costs, dtype=np.float32) / self.cfg.cost_max
        sens = np.asarray(param_sensitivity, dtype=np.float32)
        metrics = np.array(
            [
                float(mean_x_vel) / 5.0,
                float(termination_rate),
                float(energy_cost) / 100.0,
                float(ema_win_rate),
            ],
            dtype=np.float32,
        )
        return np.concatenate([norm_l, norm_b, norm_c, sens, metrics], axis=0)

    def get_distribution(self, state_t: torch.Tensor) -> Tuple[D.Independent, torch.Tensor]:
        """Returns Independent Beta(alpha, beta) over (0, 1)^P and value estimate V(s)."""
        feat = self.actor_backbone(state_t)
        # Ensure alpha, beta > 1.05 for unimodal/smooth bounded densities
        alpha = F.softplus(self.alpha_head(feat)) + 1.05
        beta = F.softplus(self.beta_head(feat)) + 1.05
        dist = D.Independent(D.Beta(alpha, beta), 1)
        val = self.critic_net(state_t)
        return dist, val

    def sample_tweak(
        self,
        state: np.ndarray,
        limits: np.ndarray,
        budget: float,
        costs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Samples raw tweak u_e in (0, 1)^P, scales by L_e, and projects via Knapsack KKT.

        Returns:
            w_proj: Valid tweak vector w_e in [0, L_e] satisfying sum_i c_i w_i <= B_e.
            w_raw: Unprojected limit-scaled tweak vector L_e * u_e.
            u_raw: Unit hypercube sample in (0, 1)^P.
            log_prob: Log probability density under the Beta distribution.
            value: Critic value estimate V(s_e^twk).
        """
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            l_t = torch.as_tensor(limits, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_t = torch.as_tensor(costs, dtype=torch.float32, device=self.device).unsqueeze(0)
            b_t = torch.tensor([[float(budget)]], dtype=torch.float32, device=self.device)

            dist, val_t = self.get_distribution(s_t)
            if deterministic:
                u_t = dist.mean
            else:
                u_t = dist.sample()
            u_t = torch.clamp(u_t, 1e-4, 1.0 - 1e-4)
            log_p = float(dist.log_prob(u_t).item())
            val = float(val_t.squeeze().item())

            w_raw_t = l_t * u_t
            w_proj_t = DifferentiableKnapsackProjection.apply(w_raw_t, c_t, b_t, l_t)

        return (
            w_proj_t.squeeze(0).cpu().numpy().astype(np.float32),
            w_raw_t.squeeze(0).cpu().numpy().astype(np.float32),
            u_t.squeeze(0).cpu().numpy().astype(np.float32),
            log_p,
            val,
        )

    def compute_reward(
        self,
        win_indicator: int,
        dominance_score: float,
        w_raw: np.ndarray,
        costs: np.ndarray,
        budget: float,
        entropy: float = 0.0,
    ) -> float:
        """Computes Tweaker reward r_e^twk penalizing agent wins and pre-projection budget excess."""
        raw_spend = float(np.sum(w_raw * costs))
        excess_ratio = max(0.0, (raw_spend - float(budget)) / (float(budget) + 1e-6))
        r_twk = (
            (1.0 - 2.0 * float(win_indicator))
            - self.cfg.beta_score * float(dominance_score)
            - self.cfg.eta_viol * excess_ratio
            + self.cfg.beta_ent * float(entropy)
        )
        return float(r_twk)

    def record_episode(
        self,
        state: np.ndarray,
        u_raw: np.ndarray,
        w_raw: np.ndarray,
        w_proj: np.ndarray,
        limits: np.ndarray,
        budget: float,
        costs: np.ndarray,
        win_indicator: int,
        dominance_score: float,
        log_prob: float,
        value: float,
    ) -> float:
        reward = self.compute_reward(
            win_indicator=win_indicator,
            dominance_score=dominance_score,
            w_raw=w_raw,
            costs=costs,
            budget=budget,
        )
        self.memory.append(
            TweakerTransition(
                state=state.copy(),
                u_raw=u_raw.copy(),
                w_raw=w_raw.copy(),
                w_proj=w_proj.copy(),
                limits=limits.copy(),
                budget=float(budget),
                costs=costs.copy(),
                reward=reward,
                log_prob=float(log_prob),
                value=float(value),
            )
        )
        return reward

    def update(self, ppo_epochs: int = 4, clip_eps: float = 0.2) -> Dict[str, float]:
        """Updates Tweaker Actor-Critic via PPO + differentiable Knapsack regularizer."""
        if len(self.memory) == 0:
            return {"tweaker_loss": 0.0, "tweaker_reward": 0.0}

        states = torch.as_tensor(
            np.stack([m.state for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        u_actions = torch.as_tensor(
            np.stack([m.u_raw for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        limits_t = torch.as_tensor(
            np.stack([m.limits for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        costs_t = torch.as_tensor(
            np.stack([m.costs for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        budgets_t = torch.as_tensor(
            np.array([[m.budget] for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            np.array([m.reward for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.array([m.log_prob for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            np.array([m.value for m in self.memory]),
            dtype=torch.float32,
            device=self.device,
        )

        advantages = rewards - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        total_loss_val = 0.0
        for _ in range(ppo_epochs):
            dist, values = self.get_distribution(states)
            new_log_probs = dist.log_prob(u_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), rewards)

            # Differentiable pathway through Knapsack projection on mean action
            u_mean = dist.mean
            w_raw_mean = limits_t * u_mean
            w_proj_mean = DifferentiableKnapsackProjection.apply(
                w_raw_mean, costs_t, budgets_t, limits_t
            )
            proj_discrepancy = F.mse_loss(
                w_raw_mean / self.cfg.param_max,
                w_proj_mean.detach() / self.cfg.param_max,
            )

            loss = (
                policy_loss
                + 0.5 * value_loss
                + 0.1 * proj_discrepancy
                - self.cfg.beta_ent * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            self.optimizer.step()
            total_loss_val = float(loss.detach().item())

        mean_rew = float(rewards.mean().item())
        self.memory.clear()
        return {"tweaker_loss": total_loss_val, "tweaker_reward": mean_rew}
