"""Model Representation Trainer and SimSiam Consistency Head for LA-MCTS."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from la_mcts.backbone import min_max_normalize, SelfAttentionBackbone
from la_mcts.gmm_heads import MultiPeakGMMActorHead, SupportCriticHead


class SimSiamConsistencyHead(nn.Module):
    """SimSiam-style Projector and Predictor MLPs for self-supervised representation consistency."""

    def __init__(
        self,
        latent_dim: int = 256,
        proj_dim: int = 512,
        pred_dim: int = 256,
        beta_mse: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta_mse = beta_mse
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.LayerNorm(pred_dim),
            nn.SiLU(),
            nn.Linear(pred_dim, proj_dim),
        )

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def predict(self, p: torch.Tensor) -> torch.Tensor:
        return self.predictor(p)

    def forward(self, z_hat: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        """Computes hybrid SimSiam negative cosine similarity + stop-gradient MSE loss."""
        p_hat = self.projector(z_hat)
        q_hat = self.predictor(p_hat)
        with torch.no_grad():
            p_true = self.projector(z_true.detach()).detach()

        cos_sim = F.cosine_similarity(q_hat, p_true, dim=-1, eps=1e-8)
        cos_loss = (1.0 - cos_sim).mean()
        mse_loss = F.mse_loss(z_hat, z_true.detach())
        return cos_loss + self.beta_mse * mse_loss


class FiLMResidualBlock(nn.Module):
    """Action-Conditioned FiLM Residual Block: h + W2 * SiLU(gamma * LN(W1 * h) + beta)."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        mod = gamma * self.ln(self.fc1(h)) + beta
        return h + self.fc2(self.act(mod))


class ModelRepresentationTrainer(nn.Module):
    """Latent dynamics and reward model g_psi(z_t, a_t) -> (z_hat_{t+1}, r_hat_t)."""

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 8,
        num_blocks: int = 3,
        proj_dim: int = 512,
        pred_dim: int = 256,
        beta_mse: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_blocks = num_blocks
        self.proj_dim = proj_dim
        self.pred_dim = pred_dim

        # Action encoder producing FiLM modulation parameters (gamma_a, beta_a)
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )

        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(latent_dim=latent_dim) for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(latent_dim)

        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, 1),
        )

        self.consistency_head = SimSiamConsistencyHead(
            latent_dim=latent_dim,
            proj_dim=proj_dim,
            pred_dim=pred_dim,
            beta_mse=beta_mse,
        )

    def _align_action_dim(self, action: torch.Tensor) -> torch.Tensor:
        cur_dim = action.shape[-1]
        if cur_dim == self.action_dim:
            return action
        if cur_dim < self.action_dim:
            pad = torch.zeros(
                *action.shape[:-1],
                self.action_dim - cur_dim,
                device=action.device,
                dtype=action.dtype,
            )
            return torch.cat([action, pad], dim=-1)
        return action[..., : self.action_dim]

    def forward(
        self, z_t: torch.Tensor, a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts next penultimate representation z_hat_{t+1} in [0, 1]^d and reward r_hat_t."""
        squeeze_out = False
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
            squeeze_out = True
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)

        a_t = self._align_action_dim(a_t)
        film_params = self.action_emb(a_t)
        gamma_a, beta_a = torch.chunk(film_params, 2, dim=-1)
        # Center gamma around 1.0 for stable identity initialization
        gamma_a = 1.0 + gamma_a

        h = z_t
        for block in self.blocks:
            h = block(h, gamma_a, beta_a)

        z_next = min_max_normalize(self.out_ln(h), eps=1e-6)
        r_hat = self.reward_head(h)

        if squeeze_out:
            return z_next.squeeze(0), r_hat.squeeze(0)
        return z_next, r_hat

    def compute_unrolled_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        target_value_seq: torch.Tensor,
        mcts_policy_samples: torch.Tensor,
        mcts_policy_weights: torch.Tensor,
        backbone: SelfAttentionBackbone,
        actor_head: MultiPeakGMMActorHead,
        critic_head: SupportCriticHead,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        limb_kinematics: Optional[torch.Tensor] = None,
        lambda_v: float = 0.5,
        lambda_p: float = 1.0,
        lambda_r: float = 1.0,
        lambda_rep: float = 4.0,
        c_reg: float = 1e-4,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes K-step unrolled total loss with SimSiam representation consistency.

        Args:
            obs_seq: (B, K+1, D_s)
            action_seq: (B, K, N)
            reward_seq: (B, K)
            target_value_seq: (B, K+1)
            mcts_policy_samples: (B, K+1, K_act, N)
            mcts_policy_weights: (B, K+1, K_act)
        """
        bsz, k_plus_1, obs_dim = obs_seq.shape
        unroll_k = k_plus_1 - 1

        # Compute ground-truth representations z_{t+k}^0 for all k = 0, ..., K
        flat_obs = obs_seq.reshape(bsz * k_plus_1, obs_dim)
        flat_ctx = None
        if context is not None:
            flat_ctx = (
                context.unsqueeze(1)
                .expand(bsz, k_plus_1, -1)
                .reshape(bsz * k_plus_1, -1)
            )
        flat_hist = None
        if history is not None:
            if history.dim() == 3:
                flat_hist = history.reshape(bsz * k_plus_1, -1)
            elif history.dim() == 2:
                flat_hist = (
                    history.unsqueeze(1)
                    .expand(bsz, k_plus_1, -1)
                    .reshape(bsz * k_plus_1, -1)
                )
        z_true_all = backbone(flat_obs, context=flat_ctx, history=flat_hist).view(bsz, k_plus_1, -1)

        z_hat = z_true_all[:, 0, :]

        # Step k = 0 value and policy loss
        kin_0 = (
            limb_kinematics[:, 0]
            if (limb_kinematics is not None and limb_kinematics.dim() == 4)
            else limb_kinematics
        )
        loss_v = critic_head.compute_loss(z_hat, target_value_seq[:, 0])
        log_probs_0 = actor_head.log_prob(
            z_hat, mcts_policy_samples[:, 0], action_mask=action_mask, limb_kinematics=kin_0
        )  # (B, K_act)
        loss_p = -torch.sum(mcts_policy_weights[:, 0] * log_probs_0, dim=-1).mean()

        loss_r = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
        loss_rep = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)

        for k in range(1, unroll_k + 1):
            z_hat, r_hat = self.forward(z_hat, action_seq[:, k - 1])
            # MuZero / EfficientZero 0.5 gradient scaling across unroll steps
            z_hat = 0.5 * z_hat + 0.5 * z_hat.detach()

            z_target_k = z_true_all[:, k, :]
            loss_rep = loss_rep + self.consistency_head(z_hat, z_target_k)
            loss_r = loss_r + F.mse_loss(r_hat.squeeze(-1), reward_seq[:, k - 1])
            loss_v = loss_v + critic_head.compute_loss(z_hat, target_value_seq[:, k])

            kin_k = (
                limb_kinematics[:, k]
                if (limb_kinematics is not None and limb_kinematics.dim() == 4)
                else limb_kinematics
            )
            log_probs_k = actor_head.log_prob(
                z_hat, mcts_policy_samples[:, k], action_mask=action_mask, limb_kinematics=kin_k
            )
            loss_p = (
                loss_p
                - torch.sum(mcts_policy_weights[:, k] * log_probs_k, dim=-1).mean()
            )

        # Normalize over unroll horizon
        loss_v = loss_v / float(k_plus_1)
        loss_p = loss_p / float(k_plus_1)
        if unroll_k > 0:
            loss_r = loss_r / float(unroll_k)
            loss_rep = loss_rep / float(unroll_k)

        # L2 weight regularization
        reg_loss = torch.tensor(0.0, device=obs_seq.device, dtype=obs_seq.dtype)
        if c_reg > 0.0:
            for p in self.parameters():
                reg_loss = reg_loss + p.pow(2).sum()
            for p in backbone.parameters():
                reg_loss = reg_loss + p.pow(2).sum()

        total_loss = (
            lambda_v * loss_v
            + lambda_p * loss_p
            + lambda_r * loss_r
            + lambda_rep * loss_rep
            + c_reg * reg_loss
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_v": float(loss_v.detach().item()),
            "loss_p": float(loss_p.detach().item()),
            "loss_r": float(loss_r.detach().item()),
            "loss_rep": float(loss_rep.detach().item()),
        }
        return total_loss, metrics
