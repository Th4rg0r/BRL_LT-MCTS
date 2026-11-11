"""Multi-Peak GMM Actor Head and Categorical Support Critic Head for LA-MCTS."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class TanhMixtureSameFamily:
    """Bijective tanh-squashed Gaussian Mixture Model distribution in [-1, 1]^N.

    Wraps torch.distributions.MixtureSameFamily with exact Jacobian-corrected log_prob
    and optional action-dimension masking for variable morphologies.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        self.logits = logits
        self.mu = mu
        self.std = std
        self.eps = eps

        self.mix = D.Categorical(logits=logits, validate_args=False)
        self.comp = D.Independent(
            D.Normal(loc=mu, scale=std, validate_args=False),
            reinterpreted_batch_ndims=1,
            validate_args=False,
        )
        self.gmm = D.MixtureSameFamily(
            mixture_distribution=self.mix,
            component_distribution=self.comp,
            validate_args=False,
        )

    @property
    def batch_shape(self) -> torch.Size:
        return self.gmm.batch_shape

    @property
    def event_shape(self) -> torch.Size:
        return self.gmm.event_shape

    def sample(
        self,
        sample_shape: torch.Size | Tuple[int, ...] = torch.Size(),
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draws samples u ~ GMM and applies a = tanh(u) in (-1, 1)^N."""
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        u = self.gmm.sample(torch.Size(sample_shape))
        a = torch.tanh(u)
        if action_mask is not None:
            a = a * action_mask
        return a

    def rsample_mode(self, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Deterministic evaluation action selecting the highest-weight mixture mode."""
        best_peak = torch.argmax(self.logits, dim=-1)  # (...,)
        idx = best_peak.unsqueeze(-1).unsqueeze(-1).expand(*best_peak.shape, 1, self.mu.shape[-1])
        chosen_mu = torch.gather(self.mu, dim=-2, index=idx).squeeze(-2)
        a = torch.tanh(chosen_mu)
        if action_mask is not None:
            a = a * action_mask
        return a

    def log_prob(
        self, a: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes exact Jacobian-corrected log probability density log pi(a | z_t)."""
        a_safe = torch.clamp(a, -0.999999, 0.999999)
        u = torch.atanh(a_safe)

        if action_mask is None:
            log_p_u = self.gmm.log_prob(u)
            jacobian = torch.sum(torch.log(1.0 - a_safe.pow(2) + self.eps), dim=-1)
            return log_p_u - jacobian

        # Masked evaluation across active joint dimensions
        # u shape: (..., N), mu/std shape: (..., M, N)
        u_exp = u.unsqueeze(-2)  # (..., 1, N)
        mask_exp = action_mask.unsqueeze(-2)  # (..., 1, N)
        var = self.std.pow(2)
        log_comp_per_dim = -0.5 * (
            ((u_exp - self.mu) / self.std).pow(2)
            + 2.0 * torch.log(self.std)
            + math.log(2.0 * math.pi)
        )
        log_comp = torch.sum(log_comp_per_dim * mask_exp, dim=-1)  # (..., M)
        log_mix = F.log_softmax(self.logits, dim=-1)  # (..., M)
        log_p_u = torch.logsumexp(log_mix + log_comp, dim=-1)
        jacobian = torch.sum(
            action_mask * torch.log(1.0 - a_safe.pow(2) + self.eps), dim=-1
        )
        return log_p_u - jacobian


class MultiPeakGMMActorHead(nn.Module):
    """Multi-Peak Gaussian Mixture Model Actor Head operating directly on z_t."""

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 8,
        num_peaks: int = 5,
        log_std_bounds: Tuple[float, float] = (-3.0, 2.0),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_peaks = num_peaks
        self.log_std_bounds = log_std_bounds

        # Strictly three direct linear layers from z_t without intermediate hidden layers
        self.logits_net = nn.Linear(latent_dim, num_peaks)
        self.mu_net = nn.Linear(latent_dim, num_peaks * action_dim)
        self.log_std_net = nn.Linear(latent_dim, num_peaks * action_dim)

    def get_distribution_params(
        self, z_t: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects z_t -> (logits, mu, std) with soft-clamped means and bounded log_std."""
        batch_shape = z_t.shape[:-1]
        logits = self.logits_net(z_t)
        mu_raw = self.mu_net(z_t).view(*batch_shape, self.num_peaks, self.action_dim)
        mu = 5.0 * torch.tanh(mu_raw / 5.0)

        log_std = self.log_std_net(z_t).view(*batch_shape, self.num_peaks, self.action_dim)
        log_std = torch.clamp(
            log_std, min=self.log_std_bounds[0], max=self.log_std_bounds[1]
        )
        std = torch.exp(log_std)

        if temperature != 1.0 and temperature > 0.0:
            logits = logits / temperature
            std = torch.clamp(
                std * math.sqrt(temperature),
                min=math.exp(self.log_std_bounds[0]),
                max=math.exp(self.log_std_bounds[1]) * 2.0,
            )

        return logits, mu, std

    def forward(
        self, z_t: torch.Tensor, temperature: float = 1.0
    ) -> TanhMixtureSameFamily:
        """Returns the TanhMixtureSameFamily distribution for latent state z_t."""
        logits, mu, std = self.get_distribution_params(z_t, temperature=temperature)
        return TanhMixtureSameFamily(logits=logits, mu=mu, std=std)

    def _fast_log_prob_u(
        self,
        u: torch.Tensor,
        logits: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized GMM log-density in pre-tanh space without Distribution object overhead.

        Args:
            u: (B, K_act, N)
            logits: (B, M)
            mu: (B, M, N)
            std: (B, M, N)
        Returns:
            log_p_u: (B, K_act)
        """
        # u_exp: (B, K_act, 1, N), mu_exp/std_exp: (B, 1, M, N)
        u_exp = u.unsqueeze(-2)
        mu_exp = mu.unsqueeze(1)
        std_exp = std.unsqueeze(1)
        log_comp_dim = -0.5 * (
            ((u_exp - mu_exp) / std_exp).pow(2)
            + 2.0 * torch.log(std_exp)
            + math.log(2.0 * math.pi)
        )
        if action_mask is not None:
            mask_exp = action_mask.view(1, 1, 1, -1) if action_mask.dim() == 1 else action_mask.unsqueeze(1).unsqueeze(1)
            log_comp = torch.sum(log_comp_dim * mask_exp, dim=-1)
        else:
            log_comp = torch.sum(log_comp_dim, dim=-1)  # (B, K_act, M)
        log_mix = F.log_softmax(logits, dim=-1).unsqueeze(1)  # (B, 1, M)
        return torch.logsumexp(log_mix + log_comp, dim=-1)  # (B, K_act)

    def sample_actions(
        self,
        z_t: torch.Tensor,
        k_act: int = 16,
        temperature: float = 1.0,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples K_act actions from beta(. | z_t) and returns (actions, prior_weights).

        Uses direct tensorized categorical + Gaussian sampling per Section 2.2 for low latency.
        """
        single = z_t.dim() == 1
        if single:
            z_t = z_t.unsqueeze(0)

        bsz = z_t.shape[0]
        logits_prop, mu_prop, std_prop = self.get_distribution_params(
            z_t, temperature=temperature
        )
        probs = F.softmax(logits_prop, dim=-1)  # (B, M)
        # Draw K_act mixture indices per batch item: (B, K_act)
        comp_indices = torch.multinomial(probs, num_samples=k_act, replacement=True)
        idx_exp = comp_indices.unsqueeze(-1).expand(bsz, k_act, self.action_dim)

        chosen_mu = torch.gather(mu_prop, dim=1, index=idx_exp)  # (B, K_act, N)
        chosen_std = torch.gather(std_prop, dim=1, index=idx_exp)  # (B, K_act, N)
        u = chosen_mu + chosen_std * torch.randn_like(chosen_mu)
        sampled = torch.tanh(u)
        if action_mask is not None:
            sampled = sampled * action_mask

        if abs(temperature - 1.0) < 1e-6:
            prior_weights = torch.full(
                (bsz, k_act),
                1.0 / float(k_act),
                device=z_t.device,
                dtype=z_t.dtype,
            )
        else:
            logits_targ, mu_targ, std_targ = self.get_distribution_params(
                z_t, temperature=1.0
            )
            log_pi_u = self._fast_log_prob_u(
                u, logits_targ, mu_targ, std_targ, action_mask=action_mask
            )
            log_beta_u = self._fast_log_prob_u(
                u, logits_prop, mu_prop, std_prop, action_mask=action_mask
            )
            # Note: tanh Jacobian cancels out identically in the importance ratio pi(a)/beta(a)!
            log_ratio = torch.clamp(log_pi_u - log_beta_u, min=-20.0, max=20.0)
            prior_weights = F.softmax(log_ratio, dim=-1)

        if single:
            return sampled.squeeze(0), prior_weights.squeeze(0)
        return sampled, prior_weights

    def log_prob(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Evaluates exact log_prob(actions | z_t) supporting both (B, N) and (B, K_act, N)."""
        dist = self.forward(z_t, temperature=temperature)
        if actions.dim() == z_t.dim() + 1:
            # actions is (B, K_act, N), dist batch is (B,)
            a_trans = actions.transpose(0, 1)  # (K_act, B, N)
            lp = dist.log_prob(a_trans, action_mask=action_mask)  # (K_act, B)
            return lp.transpose(0, 1)  # (B, K_act)
        return dist.log_prob(actions, action_mask=action_mask)


def symlog_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Invertible symmetric logarithmic scaling h(x) = sign(x)(sqrt(|x|+1) - 1 + eps*x)."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0 + eps * torch.abs(x))


def symexp_inverse(y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Exact inverse of symlog_transform h^{-1}(y)."""
    abs_y = torch.abs(y)
    inner = (torch.sqrt(1.0 + 4.0 * eps * (abs_y + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return torch.sign(y) * (inner.pow(2) - 1.0)


class SupportCriticHead(nn.Module):
    """Critic Head operating directly on z_t via Categorical Support (601 bins) or Linear."""

    def __init__(
        self,
        latent_dim: int = 256,
        use_categorical: bool = True,
        support_min: int = -300,
        support_max: int = 300,
        num_bins: int = 601,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_categorical = use_categorical
        self.support_min = float(support_min)
        self.support_max = float(support_max)
        self.num_bins = num_bins

        if use_categorical:
            self.value_net = nn.Linear(latent_dim, num_bins)
            support = torch.linspace(self.support_min, self.support_max, num_bins)
            self.register_buffer("support", support)
        else:
            self.value_net = nn.Linear(latent_dim, 1)

    def forward_logits(self, z_t: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (..., num_bins) or scalar (..., 1)."""
        return self.value_net(z_t)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Computes scalar state value estimation v_t in R^(... x 1)."""
        out = self.value_net(z_t)
        if not self.use_categorical:
            return out
        probs = F.softmax(out, dim=-1)
        expected_scaled = torch.sum(probs * self.support, dim=-1, keepdim=True)
        return symexp_inverse(expected_scaled)

    def scalar_to_two_hot(self, target_scalars: torch.Tensor) -> torch.Tensor:
        """Projects scalar targets into two-hot categorical distributions over the support."""
        target = target_scalars.squeeze(-1) if target_scalars.dim() > 1 else target_scalars
        scaled = symlog_transform(target)
        scaled = torch.clamp(scaled, min=self.support_min, max=self.support_max)

        bin_width = (self.support_max - self.support_min) / float(self.num_bins - 1)
        pos = (scaled - self.support_min) / bin_width
        low_idx = torch.floor(pos).long().clamp(0, self.num_bins - 1)
        high_idx = (low_idx + 1).clamp(0, self.num_bins - 1)
        upper_weight = (pos - low_idx.float()).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight

        two_hot = torch.zeros(
            *target.shape, self.num_bins, device=target.device, dtype=target.dtype
        )
        two_hot.scatter_add_(-1, low_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        two_hot.scatter_add_(-1, high_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return two_hot

    def compute_loss(self, z_t: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Computes categorical cross-entropy or MSE loss against target_values."""
        if not self.use_categorical:
            pred = self.forward(z_t).view_as(target_values)
            return F.mse_loss(pred, target_values)

        logits = self.forward_logits(z_t)
        target_dist = self.scalar_to_two_hot(target_values)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(target_dist * log_probs, dim=-1).mean()
