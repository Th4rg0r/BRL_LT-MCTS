"""DDPG Meta-Controller Balancer regulating Limits L_e, Budget B_e, and Costs C_e."""

from __future__ import annotations

from collections import deque
import copy
from typing import Deque, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import BRLConfig


class BalancerReplayBuffer:
    """Replay Buffer and Rolling Statistics Tracker for the DDPG Balancer."""

    def __init__(
        self, capacity: int = 5000, num_params: int = 18, window_k_w: int = 20
    ) -> None:
        self.capacity = capacity
        self.num_params = num_params
        self.window_k_w = window_k_w

        self.transitions: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = (
            deque(maxlen=capacity)
        )
        self.recent_returns: Deque[float] = deque(maxlen=window_k_w)
        self.recent_tweaks: Deque[np.ndarray] = deque(maxlen=max(window_k_w, 50))
        self.recent_failures: Deque[float] = deque(maxlen=max(window_k_w, 50))

    def record_outcome(
        self, episode_return: float, tweak_vector: np.ndarray, failed: bool
    ) -> None:
        self.recent_returns.append(float(episode_return))
        self.recent_tweaks.append(np.asarray(tweak_vector, dtype=np.float32).copy())
        self.recent_failures.append(1.0 if failed else 0.0)

    def add_transition(
        self,
        state: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        self.transitions.append(
            (
                state.copy(),
                raw_action.copy(),
                float(reward),
                next_state.copy(),
            )
        )

    def get_return_stats(self) -> Tuple[float, float, float]:
        """Returns normalized (mean_return, std_return, trend_gradient) over K_w episodes."""
        if len(self.recent_returns) == 0:
            return 0.0, 0.0, 0.0
        arr = np.array(self.recent_returns, dtype=np.float64)
        mean_r = float(np.mean(arr)) / 6000.0
        std_r = float(np.std(arr)) / 6000.0
        if len(arr) >= 2:
            x = np.arange(len(arr), dtype=np.float64)
            x_mean = np.mean(x)
            cov = np.sum((x - x_mean) * (arr - np.mean(arr)))
            var = np.sum((x - x_mean) ** 2) + 1e-8
            trend = float(cov / var) / 1000.0
        else:
            trend = 0.0
        return mean_r, std_r, trend

    def get_failure_correlations(self) -> np.ndarray:
        """Computes empirical Pearson correlation rho_fail in [-1, 1]^P between w_i and failure."""
        if len(self.recent_tweaks) < 3:
            return np.zeros(self.num_params, dtype=np.float32)

        w_mat = np.stack(list(self.recent_tweaks), axis=0).astype(np.float64)  # (N, P)
        f_vec = np.array(self.recent_failures, dtype=np.float64)  # (N,)
        f_std = np.std(f_vec)
        if f_std < 1e-8:
            return np.zeros(self.num_params, dtype=np.float32)

        f_centered = f_vec - np.mean(f_vec)
        w_centered = w_mat - np.mean(w_mat, axis=0, keepdims=True)
        w_std = np.std(w_mat, axis=0)

        corr = np.zeros(self.num_params, dtype=np.float32)
        valid = w_std > 1e-8
        if np.any(valid):
            cov = np.mean(w_centered[:, valid] * f_centered[:, np.newaxis], axis=0)
            corr_valid = cov / (w_std[valid] * f_std + 1e-8)
            corr[valid] = np.clip(corr_valid, -1.0, 1.0).astype(np.float32)
        return corr

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, len(self.transitions), size=batch_size)
        s_b, a_b, r_b, ns_b = [], [], [], []
        for i in idxs:
            s, a, r, ns = self.transitions[int(i)]
            s_b.append(s)
            a_b.append(a)
            r_b.append(r)
            ns_b.append(ns)
        return (
            torch.as_tensor(np.stack(s_b), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(a_b), dtype=torch.float32, device=device),
            torch.as_tensor(
                np.array(r_b, dtype=np.float32), dtype=torch.float32, device=device
            ).unsqueeze(-1),
            torch.as_tensor(np.stack(ns_b), dtype=torch.float32, device=device),
        )


class BalancerActorNet(nn.Module):
    """Deterministic 3-layer MLP with LayerNorm, ReLU, and Sigmoid head."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TwinQCriticNet(nn.Module):
    """Twin Q-Critic network for TD3/DDPG meta-gradient stability."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class DDPGBalancer(nn.Module):
    """Deterministic DDPG/TD3 Meta-Controller regulating (L_e, B_e, C_e)."""

    def __init__(
        self,
        config: Optional[BRLConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.cfg = config or BRLConfig()
        self.device = torch.device(device)
        self.P = self.cfg.num_params
        self.state_dim = 4 * self.P + 6
        self.action_dim = 2 * self.P + 1

        self.actor = BalancerActorNet(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = TwinQCriticNet(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.cfg.balancer_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.cfg.balancer_lr * 2.0
        )

        self.replay_buffer = BalancerReplayBuffer(
            capacity=5000, num_params=self.P, window_k_w=self.cfg.window_k_w
        )

        # Previous Balancer outputs initialized to warmup defaults
        self.last_limits = np.full(
            self.P, self.cfg.warmup_limit, dtype=np.float32
        )
        self.last_budget = float(self.cfg.warmup_budget)
        self.last_costs = np.ones(self.P, dtype=np.float32)
        self.last_tweak = np.zeros(self.P, dtype=np.float32)
        self.last_raw_action = self._encode_raw_action(
            self.last_limits, self.last_budget, self.last_costs
        )
        self._update_step = 0

    def _encode_raw_action(
        self, limits: np.ndarray, budget: float, costs: np.ndarray
    ) -> np.ndarray:
        a_l = np.clip(limits / self.cfg.param_max, 0.0, 1.0)
        a_b = np.array([np.clip(budget / self.cfg.budget_max, 0.0, 1.0)], dtype=np.float32)
        a_c = np.clip(
            (costs - self.cfg.cost_min) / (self.cfg.cost_max - self.cfg.cost_min),
            0.0,
            1.0,
        )
        return np.concatenate([a_l, a_b, a_c], axis=0).astype(np.float32)

    def _decode_raw_action(
        self, raw_action: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        P = self.P
        limits = (self.cfg.param_max * raw_action[:P]).astype(np.float32)
        budget = float(self.cfg.budget_max * raw_action[P])
        costs = (
            self.cfg.cost_min
            + (self.cfg.cost_max - self.cfg.cost_min) * raw_action[P + 1 : 2 * P + 1]
        ).astype(np.float32)
        return limits, budget, costs

    def build_state(
        self,
        ema_win_rate: float,
        last_win: int,
        dominance_score: float,
    ) -> np.ndarray:
        """Constructs Balancer state s_e^bal in R^(4P + 6) across the 5 information groups."""
        mean_r, std_r, trend_r = self.replay_buffer.get_return_stats()
        rho_fail = self.replay_buffer.get_failure_correlations()

        group1 = np.array(
            [float(ema_win_rate), float(last_win), float(dominance_score)],
            dtype=np.float32,
        )
        # Group 2: risk-adjusted normalized return (mean - 0.25*std) and trend gradient (2 dims -> total 4P+6)
        group2 = np.array([mean_r - 0.25 * std_r, trend_r], dtype=np.float32)
        group3 = (self.last_tweak / self.cfg.param_max).astype(np.float32)
        group4 = rho_fail.astype(np.float32)
        group5 = np.concatenate(
            [
                (self.last_limits / self.cfg.param_max).astype(np.float32),
                np.array([self.last_budget / self.cfg.budget_max], dtype=np.float32),
                (self.last_costs / self.cfg.cost_max).astype(np.float32),
            ],
            axis=0,
        )
        state = np.concatenate([group1, group2, group3, group4, group5], axis=0)
        assert state.shape[0] == self.state_dim
        return state

    def select_constraints(
        self,
        state: np.ndarray,
        warmup: bool = False,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Emits (L_e, B_e, C_e, raw_action) for episode e."""
        if warmup:
            limits = np.full(self.P, self.cfg.warmup_limit, dtype=np.float32)
            budget = float(self.cfg.warmup_budget)
            costs = np.ones(self.P, dtype=np.float32)
            raw_act = self._encode_raw_action(limits, budget, costs)
            self.last_limits = limits
            self.last_budget = budget
            self.last_costs = costs
            self.last_raw_action = raw_act
            return limits, budget, costs, raw_act

        with torch.no_grad():
            s_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            raw_act = self.actor(s_t).squeeze(0).cpu().numpy()

        if not deterministic and self.cfg.balancer_noise_std > 0.0:
            noise = np.random.normal(
                0.0, self.cfg.balancer_noise_std, size=raw_act.shape
            ).astype(np.float32)
            raw_act = np.clip(raw_act + noise, 0.01, 0.99)

        limits, budget, costs = self._decode_raw_action(raw_act)
        self.last_limits = limits
        self.last_budget = budget
        self.last_costs = costs
        self.last_raw_action = raw_act
        return limits, budget, costs, raw_act

    def compute_reward(
        self,
        ema_win_rate: float,
        dominance_score: float,
        w_e: np.ndarray,
        costs_e: np.ndarray,
    ) -> float:
        """Computes Balancer reward r_e^bal balancing equilibrium (W_bar -> 0.5) and progression."""
        rho_star = self.cfg.target_win_rate
        eq_term = -self.cfg.alpha_eq * ((ema_win_rate - rho_star) ** 2)
        inst_term = -self.cfg.alpha_inst * ((dominance_score - rho_star) ** 2)

        # Gaussian Zone of Proximal Development (ZPD) gate
        zpd_gate = float(
            np.exp(
                -((ema_win_rate - rho_star) ** 2)
                / (2.0 * (self.cfg.sigma_zpd**2))
            )
        )

        # Normalized difficulty and budget allocation Shannon entropy H(p_spend)
        mean_diff = float(np.mean(w_e / self.cfg.param_max))
        spend = costs_e * w_e
        total_spend = float(np.sum(spend)) + 1e-8
        p_spend = spend / total_spend
        entropy = float(
            -np.sum(p_spend * np.log(p_spend + 1e-8)) / np.log(float(self.P))
        )
        entropy = float(np.clip(entropy, 0.0, 1.0))

        prog_term = (
            self.cfg.alpha_prog
            * zpd_gate
            * (mean_diff + self.cfg.lambda_entropy * entropy)
        )
        return float(eq_term + inst_term + prog_term)

    def update(self, batch_size: int = 32, gamma: float = 0.95) -> Dict[str, float]:
        """Executes a DDPG/TD3 gradient step with soft target updates."""
        if len(self.replay_buffer.transitions) < 4:
            return {"balancer_q_loss": 0.0, "balancer_actor_loss": 0.0}

        bsz = min(batch_size, len(self.replay_buffer.transitions))
        states, actions, rewards, next_states = self.replay_buffer.sample(
            bsz, self.device
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_target = rewards + gamma * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        self._update_step += 1
        if self._update_step % 2 == 0 or len(self.replay_buffer.transitions) < 16:
            pred_actions = self.actor(states)
            q1_pred, _ = self.critic(states, pred_actions)

            # Cybernetic self-regulation guidance auxiliary loss ensuring immediate response
            # to win-rate imbalance even with small replay buffer sizes:
            # states[:, 0] is ema_win_rate; states[:, 5+P : 5+2*P] is rho_fail
            ema_w = states[:, 0:1]
            imbalance = ema_w - self.cfg.target_win_rate  # >0 if agent wins too much
            rho_fail = states[:, 5 + self.P : 5 + 2 * self.P]

            # When imbalance < 0 (env wins too much), reduce limits/budget and raise costs on failing params
            target_scale = torch.clamp(0.5 + 1.5 * imbalance, 0.05, 0.95)
            target_costs = torch.clamp(0.2 + 0.6 * F.relu(rho_fail), 0.05, 0.95)
            cyber_reg = (
                F.mse_loss(pred_actions[:, : self.P + 1], target_scale.expand(-1, self.P + 1))
                + F.mse_loss(pred_actions[:, self.P + 1 :], target_costs)
            )

            actor_loss = -q1_pred.mean() + 0.25 * cyber_reg
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_optimizer.step()
            actor_loss_val = float(actor_loss.detach().item())

            # Soft update target networks
            tau = self.cfg.balancer_tau
            with torch.no_grad():
                for p, p_targ in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    p_targ.data.mul_(1.0 - tau).add_(tau * p.data)
                for p, p_targ in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

        return {
            "balancer_q_loss": float(q_loss.detach().item()),
            "balancer_actor_loss": actor_loss_val,
        }
