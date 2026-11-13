"""Multi-Peak GMM Actor Head and Categorical Support Critic Head for LA-MCTS."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple
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


class PeakPositionalEncoding(nn.Module):
    """Positional encoder for GMM peak indices [0, 1, ..., num_peaks - 1]."""

    def __init__(self, num_peaks: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_peaks = num_peaks
        self.hidden_dim = hidden_dim
        pe = torch.zeros(num_peaks, hidden_dim)
        position = torch.arange(0, num_peaks, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, hidden_dim))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: hidden_dim // 2])
        self.register_buffer("sinusoidal_pe", pe)
        self.learnable_pe = nn.Embedding(num_peaks, hidden_dim)
        nn.init.normal_(self.learnable_pe.weight, mean=0.0, std=0.02)

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Returns peak positional encodings of shape (1, num_peaks, 1, hidden_dim)."""
        idx = torch.arange(self.num_peaks, device=device)
        pe = self.sinusoidal_pe.to(device=device, dtype=dtype) + self.learnable_pe(idx).to(dtype=dtype)
        return pe.view(1, self.num_peaks, 1, self.hidden_dim)


class MorphologicalLimbHead(nn.Module):
    """Universal Morphological Limb Network f_phi(z_t, k_i(t)) with Transformer Encoder-Decoder.

    Projects the central overall latent representation z_t and all limbs' 8D harmonic
    kinematic embeddings (3D anchor + yaw/pitch direction) into a shared hidden dimension d_limb.
    A Pre-LN Transformer Encoder lets every limb attend to the global state and all other limbs
    for inter-limb coordination, and a Pre-LN Transformer Decoder cross-attends each limb query
    against the coordinated memory before predicting per-limb GMM parameters (mu, log_std).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_peaks: int = 5,
        max_joints_per_limb: int = 4,
        hidden_dim: int = 128,
        log_std_bounds: Tuple[float, float] = (-3.0, 2.0),
        n_heads: int = 4,
        n_enc_layers: int = 1,
        n_dec_layers: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_peaks = num_peaks
        self.max_joints_per_limb = max_joints_per_limb
        self.hidden_dim = hidden_dim
        self.log_std_bounds = log_std_bounds
        self.n_heads = n_heads

        # 8D geometric query embedding:
        # [x, y, z, yaw, cos(2*pi*yaw), sin(2*pi*yaw), pitch, sin(pi*pitch)]
        self.geom_dim = 8

        # Project global latent z_t and per-limb kinematics into shared token dimension
        self.global_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.kin_proj = nn.Sequential(
            nn.Linear(self.geom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Role embeddings (global state vs. limb token) - permutation-invariant across morphologies
        self.global_type_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.limb_type_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Peak positional encoding [0, 1, ..., num_peaks - 1] + limb query fusion for the Decoder
        self.peak_pos_encoder = PeakPositionalEncoding(num_peaks=num_peaks, hidden_dim=hidden_dim)
        self.dec_query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc_layers, enable_nested_tensor=False
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.out_ln = nn.LayerNorm(hidden_dim)

        # Per-(peak, limb) composition output heads: weight (1), mean (J), and log_std (J)
        self.weight_head = nn.Linear(hidden_dim, 1)
        self.mu_head = nn.Linear(hidden_dim, max_joints_per_limb)
        self.log_std_head = nn.Linear(hidden_dim, max_joints_per_limb)
        self.last_limb_weights: Optional[torch.Tensor] = None
        self.last_peak_logits: Optional[torch.Tensor] = None

    @staticmethod
    def encode_kinematics(kinematics: torch.Tensor) -> torch.Tensor:
        """Projects (..., 5) [x, y, z, yaw, pitch] into (..., 8) harmonic kinematic representation."""
        x = kinematics[..., 0:1]
        y = kinematics[..., 1:2]
        z = kinematics[..., 2:3]
        yaw = kinematics[..., 3:4]
        pitch = kinematics[..., 4:5]

        two_pi_yaw = 2.0 * math.pi * yaw
        pi_pitch = math.pi * pitch

        cos_yaw = torch.cos(two_pi_yaw)
        sin_yaw = torch.sin(two_pi_yaw)
        sin_pitch = torch.sin(pi_pitch)

        return torch.cat([x, y, z, yaw, cos_yaw, sin_yaw, pitch, sin_pitch], dim=-1)

    def _encode_and_decode(
        self,
        z_flat: torch.Tensor,
        query_geom: torch.Tensor,
        context_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs Transformer Encoder over [global, context_limbs] and Decoder over query_limbs.

        Args:
            z_flat: (B, latent_dim)
            query_geom: (B, N_query, 8)
            context_geom: (B, N_ctx, 8); defaults to query_geom when None.
        Returns:
            h: (B, N_query, hidden_dim)
        """
        e_global = self.global_proj(z_flat).unsqueeze(1) + self.global_type_emb  # (B, 1, D)
        e_query = self.kin_proj(query_geom) + self.limb_type_emb  # (B, N_q, D)
        if context_geom is None:
            e_ctx = e_query
        else:
            e_ctx = self.kin_proj(context_geom) + self.limb_type_emb  # (B, N_ctx, D)

        enc_seq = torch.cat([e_global, e_ctx], dim=1)  # (B, 1 + N_ctx, D)
        memory = self.encoder(enc_seq)
        bsz, n_query, _ = e_query.shape
        peak_pe = self.peak_pos_encoder(z_flat.device, z_flat.dtype)
        peak_exp = peak_pe.expand(bsz, self.num_peaks, n_query, self.hidden_dim)
        limb_exp = e_query.unsqueeze(1).expand(bsz, self.num_peaks, n_query, self.hidden_dim)
        dec_queries = (
            self.dec_query_proj(torch.cat([limb_exp, peak_exp], dim=-1))
            + limb_exp
            + peak_exp
        ).reshape(bsz, self.num_peaks * n_query, self.hidden_dim)
        dec_out = self.decoder(tgt=dec_queries, memory=memory)
        return self.out_ln(dec_out).view(bsz, self.num_peaks, n_query, self.hidden_dim)

    def forward_single_limb(
        self,
        z_t: torch.Tensor,
        anchor: Tuple[float, float, float] | Sequence[float] | torch.Tensor,
        yaw: float | torch.Tensor,
        pitch: float | torch.Tensor,
        num_joints: Optional[int] = None,
        all_limbs_kinematics: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Computes per-peak (mu, log_std) [and weight when return_weights=True] for a single limb."""
        device = z_t.device
        dtype = z_t.dtype
        if not torch.is_tensor(anchor):
            anchor_t = torch.tensor(anchor, dtype=dtype, device=device)
        else:
            anchor_t = anchor.to(device=device, dtype=dtype)
        if not torch.is_tensor(yaw):
            yaw_t = torch.tensor([yaw], dtype=dtype, device=device)
        else:
            yaw_t = yaw.to(device=device, dtype=dtype).view(-1)
        if not torch.is_tensor(pitch):
            pitch_t = torch.tensor([pitch], dtype=dtype, device=device)
        else:
            pitch_t = pitch.to(device=device, dtype=dtype).view(-1)

        kin_5d = torch.cat([anchor_t.view(-1, 3), yaw_t.view(-1, 1), pitch_t.view(-1, 1)], dim=-1)
        geom = self.encode_kinematics(kin_5d).unsqueeze(1)  # (B_or_1, 1, 8)

        if z_t.dim() == 1:
            z_t_in = z_t.unsqueeze(0)
        else:
            z_t_in = z_t

        bsz = z_t_in.shape[0]
        if geom.shape[0] == 1 and bsz > 1:
            geom = geom.expand(bsz, -1, -1)

        ctx_geom = None
        if all_limbs_kinematics is not None:
            ctx_t = all_limbs_kinematics.to(device=device, dtype=dtype)
            if ctx_t.dim() == 2:
                ctx_t = ctx_t.unsqueeze(0).expand(bsz, -1, -1)
            ctx_geom = self.encode_kinematics(ctx_t)

        h = self._encode_and_decode(z_t_in, query_geom=geom, context_geom=ctx_geom).squeeze(2)
        weight_raw = self.weight_head(h).squeeze(-1)
        mu_raw = self.mu_head(h).view(bsz, self.num_peaks, self.max_joints_per_limb)
        log_std_raw = self.log_std_head(h).view(bsz, self.num_peaks, self.max_joints_per_limb)
        n_j = num_joints if num_joints is not None else self.max_joints_per_limb
        mu_raw = mu_raw[..., :n_j]
        log_std_raw = log_std_raw[..., :n_j]
        self.last_limb_weights = weight_raw.unsqueeze(-1).detach()
        self.last_peak_logits = weight_raw.detach()
        if z_t.dim() == 1:
            if return_weights:
                return weight_raw.squeeze(0), mu_raw.squeeze(0), log_std_raw.squeeze(0)
            return mu_raw.squeeze(0), log_std_raw.squeeze(0)
        if return_weights:
            return weight_raw, mu_raw, log_std_raw
        return mu_raw, log_std_raw

    def forward_all_limbs(
        self,
        z_t: torch.Tensor,
        limb_kinematics: torch.Tensor,
        limb_joint_counts: Optional[Sequence[int]] = None,
        target_action_dim: Optional[int] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Vectorized forward pass over all creature limbs and all GMM peaks."""
        orig_shape = z_t.shape[:-1]
        z_flat = z_t.view(-1, self.latent_dim)
        bsz = z_flat.shape[0]

        if limb_kinematics.dim() == 2:
            n_limbs = limb_kinematics.shape[0]
            kin_flat = limb_kinematics.unsqueeze(0).expand(bsz, n_limbs, 5)
        elif limb_kinematics.dim() >= 3:
            n_limbs = limb_kinematics.shape[-2]
            kin_flat = limb_kinematics.reshape(-1, n_limbs, 5)
            if kin_flat.shape[0] == 1 and bsz > 1:
                kin_flat = kin_flat.expand(bsz, n_limbs, 5)
        else:
            raise ValueError(f"Unsupported limb_kinematics shape: {limb_kinematics.shape}")

        geom_emb = self.encode_kinematics(kin_flat)  # (bsz, n_limbs, 8)
        h = self._encode_and_decode(z_flat, query_geom=geom_emb)  # (bsz, num_peaks, n_limbs, hidden_dim)
        limb_weights = self.weight_head(h).squeeze(-1)  # (bsz, num_peaks, n_limbs)
        mu_limbs = self.mu_head(h)  # (bsz, num_peaks, n_limbs, max_joints_per_limb)
        log_std_limbs = self.log_std_head(h)  # (bsz, num_peaks, n_limbs, max_joints_per_limb)
        if limb_joint_counts is not None:
            pieces_mu = []
            pieces_std = []
            active_limb_indices = []
            for i, j_count in enumerate(limb_joint_counts):
                if i < n_limbs and j_count > 0:
                    pieces_mu.append(mu_limbs[:, :, i, :j_count])
                    pieces_std.append(log_std_limbs[:, :, i, :j_count])
                    active_limb_indices.append(i)
            mu_assembled = torch.cat(pieces_mu, dim=-1)
            log_std_assembled = torch.cat(pieces_std, dim=-1)
            peak_logits = (
                limb_weights[:, :, active_limb_indices].mean(dim=-1)
                if active_limb_indices
                else limb_weights.mean(dim=-1)
            )
        else:
            mu_assembled = mu_limbs.reshape(bsz, self.num_peaks, n_limbs * self.max_joints_per_limb)
            log_std_assembled = log_std_limbs.reshape(bsz, self.num_peaks, n_limbs * self.max_joints_per_limb)
            peak_logits = limb_weights.mean(dim=-1)

        if target_action_dim is not None:
            cur_dim = mu_assembled.shape[-1]
            if cur_dim < target_action_dim:
                pad_size = target_action_dim - cur_dim
                mu_pad = torch.zeros(bsz, self.num_peaks, pad_size, device=z_t.device, dtype=z_t.dtype)
                std_pad = torch.zeros(bsz, self.num_peaks, pad_size, device=z_t.device, dtype=z_t.dtype)
                mu_assembled = torch.cat([mu_assembled, mu_pad], dim=-1)
                log_std_assembled = torch.cat([log_std_assembled, std_pad], dim=-1)
            elif cur_dim > target_action_dim:
                mu_assembled = mu_assembled[..., :target_action_dim]
                log_std_assembled = log_std_assembled[..., :target_action_dim]

        if orig_shape:
            mu_assembled = mu_assembled.view(*orig_shape, self.num_peaks, mu_assembled.shape[-1])
            log_std_assembled = log_std_assembled.view(*orig_shape, self.num_peaks, log_std_assembled.shape[-1])
            peak_logits = peak_logits.view(*orig_shape, self.num_peaks)
            limb_weights = limb_weights.view(*orig_shape, self.num_peaks, n_limbs)
        self.last_limb_weights = limb_weights.detach()
        self.last_peak_logits = peak_logits.detach()
        if return_weights:
            return peak_logits, mu_assembled, log_std_assembled, limb_weights
        return mu_assembled, log_std_assembled


class MultiPeakGMMActorHead(nn.Module):
    """Multi-Peak Gaussian Mixture Model Actor Head operating directly on z_t.

    Supports universal parameter sharing across limbs via MorphologicalLimbHead,
    evaluating limbs in parallel using their 3D connection anchor (x, y, z) and
    instantaneous pointing direction (yaw, pitch).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 8,
        num_peaks: int = 5,
        log_std_bounds: Tuple[float, float] = (-3.0, 2.0),
        use_morphological_limbs: bool = True,
        joints_per_limb: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_peaks = num_peaks
        self.log_std_bounds = log_std_bounds
        self.use_morphological_limbs = use_morphological_limbs
        self.joints_per_limb = joints_per_limb

        # Mixture mode selection logits net
        self.logits_net = nn.Linear(latent_dim, num_peaks)

        # Baseline direct linear layers (for legacy fallback)
        self.mu_net = nn.Linear(latent_dim, num_peaks * action_dim)
        self.log_std_net = nn.Linear(latent_dim, num_peaks * action_dim)

        # Configure default kinematics and joint partitioning based on action_dim
        if action_dim == 8:
            # Ant (4 legs, 2 joints each)
            self.default_joint_counts = [2, 2, 2, 2]
            max_j = 2
            def_kin = torch.tensor([
                [ 0.2,  0.2, 0.0, 0.625, 0.50],
                [-0.2,  0.2, 0.0, 0.875, 0.50],
                [-0.2, -0.2, 0.0, 0.125, 0.50],
                [ 0.2, -0.2, 0.0, 0.375, 0.50],
            ], dtype=torch.float32)
        elif action_dim == 12:
            # Dog / Hexapod (4 legs * 3 joints, or 6 legs * 2 joints)
            j_lim = joints_per_limb if joints_per_limb is not None else 3
            self.default_joint_counts = [j_lim] * 4
            max_j = 3
            def_kin = torch.tensor([
                [ 0.188, -0.08, 0.0, 0.064, 0.286],
                [ 0.188,  0.08, 0.0, 0.938, 0.260],
                [-0.188, -0.08, 0.0, 0.060, 0.256],
                [-0.188,  0.08, 0.0, 0.946, 0.261],
            ], dtype=torch.float32)
        elif action_dim == 17:
            # Humanoid (waist: 3, r_leg: 4, l_leg: 4, r_arm: 3, l_arm: 3)
            self.default_joint_counts = [3, 4, 4, 3, 3]
            max_j = 4
            def_kin = torch.tensor([
                [ 0.000,  0.000,  0.000, 0.500, 0.500],
                [-0.010, -0.100, -0.465, 0.741, 0.008],
                [-0.008,  0.100, -0.465, 0.364, 0.004],
                [ 0.000, -0.170,  0.060, 0.376, 0.303],
                [ 0.000,  0.170,  0.060, 0.623, 0.302],
            ], dtype=torch.float32)
        else:
            j_per = max(1, action_dim // 4)
            rem = action_dim - j_per * 3
            self.default_joint_counts = [j_per, j_per, j_per, rem]
            max_j = max(j_per, rem)
            def_kin = torch.tensor([
                [ 0.2,  0.2, 0.0, 0.125, 0.50],
                [-0.2,  0.2, 0.0, 0.875, 0.50],
                [-0.2, -0.2, 0.0, 0.375, 0.50],
                [ 0.2, -0.2, 0.0, 0.625, 0.50],
            ], dtype=torch.float32)

        self.register_buffer("default_limb_kinematics", def_kin)

        effective_max_j = joints_per_limb if joints_per_limb is not None else max_j
        self.limb_head = MorphologicalLimbHead(
            latent_dim=latent_dim,
            num_peaks=num_peaks,
            max_joints_per_limb=effective_max_j,
            hidden_dim=128,
            log_std_bounds=log_std_bounds,
        )
        self.last_limb_weights: Optional[torch.Tensor] = None
        nn.init.zeros_(self.limb_head.weight_head.weight)
        nn.init.zeros_(self.limb_head.weight_head.bias)

    def get_distribution_params(
        self,
        z_t: torch.Tensor,
        temperature: float = 1.0,
        limb_kinematics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects z_t -> (logits, mu, std) with soft-clamped means and bounded log_std."""
        batch_shape = z_t.shape[:-1]
        logits = self.logits_net(z_t)

        if self.use_morphological_limbs:
            kin = limb_kinematics
            if kin is None:
                kin = self.default_limb_kinematics.to(z_t.device)
            elif not torch.is_tensor(kin):
                kin = torch.as_tensor(kin, dtype=torch.float32, device=z_t.device)

            n_limbs = kin.shape[-2]
            if self.joints_per_limb is not None:
                counts = [self.joints_per_limb] * n_limbs
            elif len(self.default_joint_counts) == n_limbs:
                counts = self.default_joint_counts
            else:
                counts = None
            peak_logits_limb, mu_raw, log_std_raw, limb_weights = (
                self.limb_head.forward_all_limbs(
                    z_t,
                    kin,
                    limb_joint_counts=counts,
                    target_action_dim=self.action_dim,
                    return_weights=True,
                )
            )
            logits = logits + peak_logits_limb
            self.last_limb_weights = limb_weights.detach()
            mu = 5.0 * torch.tanh(mu_raw / 5.0)
            log_std = torch.clamp(
                log_std_raw, min=self.log_std_bounds[0], max=self.log_std_bounds[1]
            )
            std = torch.exp(log_std)
        else:
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

    def select_composition(
        self,
        z_t: torch.Tensor,
        limb_kinematics: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stochastically selects a whole-body position COMPOSITION (peak index k*)
        and samples all limbs jointly from that selected peak composition.
        """
        single = z_t.dim() == 1
        if single:
            z_t = z_t.unsqueeze(0)
        logits, mu, std = self.get_distribution_params(
            z_t, temperature=temperature, limb_kinematics=limb_kinematics
        )
        peak_probs = F.softmax(logits, dim=-1)
        if deterministic:
            selected_peak = torch.argmax(logits, dim=-1)
        else:
            selected_peak = D.Categorical(probs=peak_probs).sample()
        idx = selected_peak.unsqueeze(-1).unsqueeze(-1).expand(*selected_peak.shape, 1, self.action_dim)
        chosen_mu = torch.gather(mu, dim=-2, index=idx).squeeze(-2)
        chosen_std = torch.gather(std, dim=-2, index=idx).squeeze(-2)
        u = chosen_mu if deterministic else (chosen_mu + chosen_std * torch.randn_like(chosen_mu))
        action = torch.tanh(u)
        if action_mask is not None:
            action = action * action_mask
        if single:
            return action.squeeze(0), selected_peak.squeeze(0), peak_probs.squeeze(0)
        return action, selected_peak, peak_probs

    def get_limb_action(
        self,
        z_t: torch.Tensor,
        anchor: Tuple[float, float, float] | Sequence[float],
        yaw: float,
        pitch: float,
        num_joints: Optional[int] = None,
        deterministic: bool = False,
        peak_idx: Optional[int | torch.Tensor] = None,
        all_limbs_kinematics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Queries movements for a specific limb given anchor (x,y,z) and angles (yaw, pitch)."""
        limb_weights, mu_raw, log_std_raw = self.limb_head.forward_single_limb(
            z_t,
            anchor=anchor,
            yaw=yaw,
            pitch=pitch,
            num_joints=num_joints,
            all_limbs_kinematics=all_limbs_kinematics,
            return_weights=True,
        )
        logits = self.logits_net(z_t) + limb_weights
        mu = 5.0 * torch.tanh(mu_raw / 5.0)
        std = torch.exp(
            torch.clamp(log_std_raw, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        )
        if peak_idx is not None:
            mode_idx = (
                peak_idx.to(device=z_t.device, dtype=torch.long)
                if torch.is_tensor(peak_idx)
                else torch.tensor(peak_idx, dtype=torch.long, device=z_t.device)
            )
        elif deterministic:
            mode_idx = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            mode_idx = D.Categorical(probs=probs).sample()
        if deterministic:
            if z_t.dim() == 1:
                u = mu[mode_idx]
            else:
                idx = mode_idx.unsqueeze(-1).unsqueeze(-1).expand(*mode_idx.shape, 1, mu.shape[-1])
                u = torch.gather(mu, dim=-2, index=idx).squeeze(-2)
        else:
            if z_t.dim() == 1:
                chosen_mu = mu[mode_idx]
                chosen_std = std[mode_idx]
            else:
                idx = mode_idx.unsqueeze(-1).unsqueeze(-1).expand(*mode_idx.shape, 1, mu.shape[-1])
                chosen_mu = torch.gather(mu, dim=-2, index=idx).squeeze(-2)
                chosen_std = torch.gather(std, dim=-2, index=idx).squeeze(-2)
            u = chosen_mu + chosen_std * torch.randn_like(chosen_mu)

        return torch.tanh(u)

    def forward(
        self,
        z_t: torch.Tensor,
        temperature: float = 1.0,
        limb_kinematics: Optional[torch.Tensor] = None,
    ) -> TanhMixtureSameFamily:
        """Returns the TanhMixtureSameFamily distribution for latent state z_t."""
        logits, mu, std = self.get_distribution_params(
            z_t, temperature=temperature, limb_kinematics=limb_kinematics
        )
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
        limb_kinematics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples K_act actions from beta(. | z_t) and returns (actions, prior_weights).

        Uses direct tensorized categorical + Gaussian sampling per Section 2.2 for low latency.
        """
        single = z_t.dim() == 1
        if single:
            z_t = z_t.unsqueeze(0)

        bsz = z_t.shape[0]
        logits_prop, mu_prop, std_prop = self.get_distribution_params(
            z_t, temperature=temperature, limb_kinematics=limb_kinematics
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
                z_t, temperature=1.0, limb_kinematics=limb_kinematics
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
        limb_kinematics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates exact log_prob(actions | z_t) supporting both (B, N) and (B, K_act, N)."""
        dist = self.forward(z_t, temperature=temperature, limb_kinematics=limb_kinematics)
        if actions.dim() == z_t.dim() + 1:
            # actions is (B, K_act, N), dist batch is (B,)
            a_trans = actions.transpose(0, 1)  # (K_act, B, N)
            lp = dist.log_prob(a_trans, action_mask=action_mask)  # (K_act, B)
            return lp.transpose(0, 1)  # (B, K_act)
        return dist.log_prob(actions, action_mask=action_mask)

    def log_prob_pre_tanh(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        limb_kinematics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates pre-tanh GMM log_prob(atanh(actions) | z_t), omitting the constant tanh Jacobian."""
        logits, mu, std = self.get_distribution_params(
            z_t, temperature=temperature, limb_kinematics=limb_kinematics
        )
        a_clamped = torch.clamp(actions, -1.0 + 1e-5, 1.0 - 1e-5)
        u = torch.atanh(a_clamped)
        if actions.dim() == z_t.dim():
            u = u.unsqueeze(-2)
            lp = self._fast_log_prob_u(u, logits, mu, std, action_mask=action_mask)
            return lp.squeeze(-1)
        return self._fast_log_prob_u(u, logits, mu, std, action_mask=action_mask)

    def entropy_proxy(
        self,
        z_t: torch.Tensor,
        limb_kinematics: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable entropy proxy combining peak mixture entropy and per-joint Gaussian entropy.

        Returns:
            total_entropy: (B,) peak_entropy + mean_per_joint_gaussian_entropy
            peak_entropy: (B,) Categorical entropy over GMM peaks
            mean_std: (B,) mixture-weighted mean action std across active joints
        """
        logits, _, std = self.get_distribution_params(
            z_t, temperature=1.0, limb_kinematics=limb_kinematics
        )
        peak_probs = F.softmax(logits, dim=-1)
        peak_log_probs = F.log_softmax(logits, dim=-1)
        peak_entropy = -(peak_probs * peak_log_probs).sum(dim=-1)

        comp_gauss_ent = 0.5 * math.log(2.0 * math.pi * math.e) + torch.log(std)
        if action_mask is not None:
            mask = (
                action_mask.view(*([1] * (std.dim() - 1)), -1)
                if action_mask.dim() == 1
                else action_mask.unsqueeze(-2)
            )
            denom = mask.sum(dim=-1, keepdim=False).clamp_min(1.0)
            if denom.dim() > peak_probs.dim() - 1:
                denom = denom.squeeze(-1)
            gauss_entropy = (peak_probs.unsqueeze(-1) * comp_gauss_ent * mask).sum(dim=(-2, -1)) / denom
            mean_std = (peak_probs.unsqueeze(-1) * std * mask).sum(dim=(-2, -1)) / denom
        else:
            gauss_entropy = (peak_probs.unsqueeze(-1) * comp_gauss_ent).sum(dim=-2).mean(dim=-1)
            mean_std = (peak_probs.unsqueeze(-1) * std).sum(dim=-2).mean(dim=-1)

        return peak_entropy + gauss_entropy, peak_entropy, mean_std


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
