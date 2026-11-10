"""Trainable Latent-Action MCTS Agent with Trajectory Replay Buffer and Reanalyze."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim

from brl_core.interfaces import AbstractBRLSolverAgent
from config import LAMCTSConfig
from la_mcts.backbone import SelfAttentionBackbone
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead
from la_mcts.rep_trainer import ModelRepresentationTrainer
from la_mcts.sampled_mcts import SampledContinuousMCTS


@dataclass
class EpisodeTrajectory:
    """Stores a single episode's transitions and MCTS targets for K-step unroll training."""

    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    mcts_values: List[float] = field(default_factory=list)
    mcts_policy_samples: List[np.ndarray] = field(default_factory=list)
    mcts_policy_weights: List[np.ndarray] = field(default_factory=list)
    context: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.float32)
    )
    action_mask: np.ndarray = field(
        default_factory=lambda: np.ones(8, dtype=np.float32)
    )

    def __len__(self) -> int:
        return len(self.rewards)


class TrajectoryReplayBuffer:
    """Prioritized Trajectory Replay Buffer supporting K-step unrolls and MuZero Reanalyze."""

    def __init__(
        self,
        capacity_steps: int = 50000,
        unroll_steps: int = 5,
        td_steps: int = 5,
        gamma: float = 0.99,
    ) -> None:
        self.capacity_steps = capacity_steps
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.gamma = gamma
        self.trajectories: Deque[EpisodeTrajectory] = deque()
        self.total_steps: int = 0

    def add_trajectory(self, traj: EpisodeTrajectory) -> None:
        if len(traj) == 0:
            return
        self.trajectories.append(traj)
        self.total_steps += len(traj)
        while self.total_steps > self.capacity_steps and len(self.trajectories) > 1:
            removed = self.trajectories.popleft()
            self.total_steps -= len(removed)

    def compute_td_target(self, traj: EpisodeTrajectory, idx: int) -> float:
        """Computes n_td-step bootstrapped target value V_{t+k}^target."""
        T = len(traj)
        value_target = 0.0
        discount = 1.0
        for j in range(self.td_steps):
            step = idx + j
            if step < T:
                value_target += discount * float(traj.rewards[step])
                discount *= self.gamma
            else:
                break
        boot_idx = idx + self.td_steps
        if boot_idx < len(traj.mcts_values):
            value_target += discount * float(traj.mcts_values[boot_idx])
        return value_target

    def sample_unroll_batch(
        self, batch_size: int, device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[Tuple[EpisodeTrajectory, int]],
    ]:
        """Samples a batch of K-step unroll windows from stored trajectories."""
        valid_trajs = [t for t in self.trajectories if len(t) >= 1]
        if not valid_trajs:
            raise ValueError("Replay buffer is empty.")

        K = self.unroll_steps
        obs_batch = []
        act_batch = []
        rew_batch = []
        val_batch = []
        samples_batch = []
        weights_batch = []
        ctx_batch = []
        mask_batch = []
        refs: List[Tuple[EpisodeTrajectory, int]] = []

        traj_indices = np.random.randint(0, len(valid_trajs), size=batch_size)
        for t_idx in traj_indices:
            traj = valid_trajs[int(t_idx)]
            T = len(traj)
            start = int(np.random.randint(0, T))
            refs.append((traj, start))

            obs_seq = []
            act_seq = []
            rew_seq = []
            val_seq = []
            samp_seq = []
            w_seq = []

            for k in range(K + 1):
                step = min(start + k, T - 1)
                obs_seq.append(traj.obs[step])
                val_seq.append(self.compute_td_target(traj, step))
                samp_seq.append(traj.mcts_policy_samples[step])
                w_seq.append(traj.mcts_policy_weights[step])
                if k < K:
                    if start + k < T:
                        act_seq.append(traj.actions[start + k])
                        rew_seq.append(traj.rewards[start + k])
                    else:
                        act_seq.append(np.zeros_like(traj.actions[-1]))
                        rew_seq.append(0.0)

            obs_batch.append(np.stack(obs_seq, axis=0))
            act_batch.append(np.stack(act_seq, axis=0))
            rew_batch.append(np.array(rew_seq, dtype=np.float32))
            val_batch.append(np.array(val_seq, dtype=np.float32))
            samples_batch.append(np.stack(samp_seq, axis=0))
            weights_batch.append(np.stack(w_seq, axis=0))
            ctx_batch.append(traj.context)
            mask_batch.append(traj.action_mask)

        return (
            torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(act_batch), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(rew_batch), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(val_batch), dtype=torch.float32, device=device),
            torch.as_tensor(
                np.stack(samples_batch), dtype=torch.float32, device=device
            ),
            torch.as_tensor(
                np.stack(weights_batch), dtype=torch.float32, device=device
            ),
            torch.as_tensor(np.stack(ctx_batch), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(mask_batch), dtype=torch.float32, device=device),
            refs,
        )


class LatentActionMCTSAgent(AbstractBRLSolverAgent):
    """Complete Framework 1 Agent combining Backbone, GMM Heads, Rep Trainer, and MCTS."""

    def __init__(
        self,
        config: Optional[LAMCTSConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = config or LAMCTSConfig()
        self.device = torch.device(device)

        self.backbone = SelfAttentionBackbone(
            obs_dim=self.cfg.obs_dim,
            k_tok=self.cfg.k_tok,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            l_enc=self.cfg.l_enc,
            latent_dim=self.cfg.latent_dim,
            context_dim=self.cfg.context_dim,
            history_len=self.cfg.history_len,
            max_action_dim=self.cfg.max_action_dim,
        ).to(self.device)

        self.actor_head = MultiPeakGMMActorHead(
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.max_action_dim,
            num_peaks=self.cfg.num_peaks,
            log_std_bounds=self.cfg.log_std_bounds,
        ).to(self.device)

        self.critic_head = SupportCriticHead(
            latent_dim=self.cfg.latent_dim,
            use_categorical=self.cfg.use_categorical_critic,
            support_min=self.cfg.support_min,
            support_max=self.cfg.support_max,
            num_bins=self.cfg.num_bins,
        ).to(self.device)

        self.rep_trainer = ModelRepresentationTrainer(
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.max_action_dim,
            num_blocks=self.cfg.l_dyn,
            proj_dim=self.cfg.proj_dim,
            pred_dim=self.cfg.pred_dim,
            beta_mse=self.cfg.beta_mse,
        ).to(self.device)

        self.mcts = SampledContinuousMCTS(
            actor_head=self.actor_head,
            critic_head=self.critic_head,
            rep_trainer=self.rep_trainer,
            n_sim=self.cfg.n_sim,
            k_act=self.cfg.k_act,
            gamma=self.cfg.gamma,
            c_1=self.cfg.c_1,
            c_2=self.cfg.c_2,
            root_tau_beta=self.cfg.root_tau_beta,
            inner_tau_beta=self.cfg.inner_tau_beta,
            tau_act=self.cfg.tau_act_train,
        )

        params = (
            list(self.backbone.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
            + list(self.rep_trainer.parameters())
        )
        self.optimizer = optim.AdamW(params, lr=self.cfg.lr, weight_decay=1e-5)

        self.replay_buffer = TrajectoryReplayBuffer(
            capacity_steps=self.cfg.buffer_capacity,
            unroll_steps=self.cfg.unroll_steps,
            td_steps=self.cfg.td_steps,
            gamma=self.cfg.gamma,
        )

        # Active episode state
        self.current_context = np.zeros(self.cfg.context_dim, dtype=np.float32)
        self.current_action_mask = np.ones(self.cfg.max_action_dim, dtype=np.float32)
        self.current_trajectory = EpisodeTrajectory(
            context=self.current_context.copy(),
            action_mask=self.current_action_mask.copy(),
        )
        self._last_mcts_info: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self._diagnostics: Dict[str, float] = {
            "v_s0": 0.0,
            "td_error": 0.0,
            "loss_total": 0.0,
            "loss_rep": 0.0,
        }

    def set_episode_context(
        self,
        tweak_vector: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Sets the context token (normalized tweak vector w_e / 1000) and morphology mask."""
        if tweak_vector is not None:
            norm_w = np.asarray(tweak_vector, dtype=np.float32) / 1000.0
            if norm_w.shape[0] < self.cfg.context_dim:
                norm_w = np.pad(norm_w, (0, self.cfg.context_dim - norm_w.shape[0]))
            self.current_context = norm_w[: self.cfg.context_dim].copy()
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=np.float32)
            if mask.shape[0] < self.cfg.max_action_dim:
                mask = np.pad(mask, (0, self.cfg.max_action_dim - mask.shape[0]))
            self.current_action_mask = mask[: self.cfg.max_action_dim].copy()

        self.current_trajectory.context = self.current_context.copy()
        self.current_trajectory.action_mask = self.current_action_mask.copy()

    def _align_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        if obs_arr.shape[0] == self.cfg.obs_dim:
            return obs_arr
        if obs_arr.shape[0] < self.cfg.obs_dim:
            return np.pad(obs_arr, (0, self.cfg.obs_dim - obs_arr.shape[0]))
        return obs_arr[: self.cfg.obs_dim]

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Runs Sampled Continuous MCTS from z_t = backbone(obs) to choose an action."""
        obs_aligned = self._align_obs(obs)
        with torch.inference_mode():
            obs_t = torch.as_tensor(
                obs_aligned, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            ctx_t = torch.as_tensor(
                self.current_context, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                self.current_action_mask, dtype=torch.float32, device=self.device
            )

            z_root = self.backbone(obs_t, context=ctx_t).squeeze(0)
            n_sim = self.cfg.n_sim_eval if deterministic else self.cfg.n_sim
            tau_act = self.cfg.tau_act_eval if deterministic else self.cfg.tau_act_train

            chosen_action, policy_weights, root_samples, root_value, _ = (
                self.mcts.search(
                    z_root,
                    deterministic=deterministic,
                    tau_act=tau_act,
                    n_sim=n_sim,
                    action_mask=mask_t,
                )
            )

        self._last_mcts_info = (root_samples, policy_weights, root_value)
        if len(self.current_trajectory) == 0:
            self._diagnostics["v_s0"] = float(root_value)

        return chosen_action * self.current_action_mask

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
        obs_aligned = self._align_obs(obs)
        act_arr = np.asarray(action, dtype=np.float32).flatten()
        if act_arr.shape[0] < self.cfg.max_action_dim:
            act_arr = np.pad(act_arr, (0, self.cfg.max_action_dim - act_arr.shape[0]))
        else:
            act_arr = act_arr[: self.cfg.max_action_dim]

        if self._last_mcts_info is not None:
            root_samples, policy_weights, root_value = self._last_mcts_info
        else:
            root_samples = np.tile(act_arr, (self.cfg.k_act, 1))
            policy_weights = np.full(
                self.cfg.k_act, 1.0 / self.cfg.k_act, dtype=np.float32
            )
            root_value = float(reward)

        self.current_trajectory.obs.append(obs_aligned)
        self.current_trajectory.actions.append(act_arr)
        self.current_trajectory.rewards.append(float(reward))
        self.current_trajectory.mcts_values.append(float(root_value))
        self.current_trajectory.mcts_policy_samples.append(root_samples)
        self.current_trajectory.mcts_policy_weights.append(policy_weights)
        self._last_mcts_info = None

        if terminated or truncated:
            self.replay_buffer.add_trajectory(self.current_trajectory)
            self.current_trajectory = EpisodeTrajectory(
                context=self.current_context.copy(),
                action_mask=self.current_action_mask.copy(),
            )

    def reanalyze_batch(self, num_states: int = 8) -> None:
        """MuZero Reanalyze: refreshes stored MCTS policy/value targets using latest networks."""
        valid_trajs = [t for t in self.replay_buffer.trajectories if len(t) >= 1]
        if not valid_trajs:
            return

        with torch.inference_mode():
            for _ in range(num_states):
                traj = valid_trajs[int(np.random.randint(0, len(valid_trajs)))]
                step = int(np.random.randint(0, len(traj)))
                obs_t = torch.as_tensor(
                    traj.obs[step], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                ctx_t = torch.as_tensor(
                    traj.context, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                mask_t = torch.as_tensor(
                    traj.action_mask, dtype=torch.float32, device=self.device
                )
                z_root = self.backbone(obs_t, context=ctx_t).squeeze(0)
                _, policy_weights, root_samples, root_value, _ = self.mcts.search(
                    z_root,
                    deterministic=False,
                    tau_act=self.cfg.tau_act_train,
                    n_sim=max(16, self.cfg.n_sim // 2),
                    action_mask=mask_t,
                )
                traj.mcts_policy_samples[step] = root_samples
                traj.mcts_policy_weights[step] = policy_weights
                traj.mcts_values[step] = float(root_value)

    def train_step(self) -> Dict[str, float]:
        """Executes a K-step unrolled gradient update on the replay buffer."""
        if len(self.replay_buffer.trajectories) == 0:
            return dict(self._diagnostics)

        bsz = min(self.cfg.batch_size, max(4, self.replay_buffer.total_steps))
        (
            obs_seq,
            act_seq,
            rew_seq,
            val_seq,
            samples_seq,
            weights_seq,
            ctx_batch,
            mask_batch,
            _,
        ) = self.replay_buffer.sample_unroll_batch(bsz, self.device)

        self.optimizer.zero_grad()
        total_loss, metrics = self.rep_trainer.compute_unrolled_loss(
            obs_seq=obs_seq,
            action_seq=act_seq,
            reward_seq=rew_seq,
            target_value_seq=val_seq,
            mcts_policy_samples=samples_seq,
            mcts_policy_weights=weights_seq,
            backbone=self.backbone,
            actor_head=self.actor_head,
            critic_head=self.critic_head,
            context=ctx_batch,
            action_mask=mask_batch,
            lambda_v=self.cfg.lambda_v,
            lambda_p=self.cfg.lambda_p,
            lambda_r=self.cfg.lambda_r,
            lambda_rep=self.cfg.lambda_rep,
            c_reg=self.cfg.c_reg,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters())
            + list(self.rep_trainer.parameters()),
            max_norm=5.0,
        )
        self.optimizer.step()

        self._diagnostics.update(metrics)
        self._diagnostics["td_error"] = float(metrics.get("loss_v", 0.0))
        return dict(self._diagnostics)

    def get_diagnostics(self) -> Dict[str, float]:
        return dict(self._diagnostics)
