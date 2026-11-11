"""Sub-State Tokenizer and Self-Attention Backbone for Latent-Action MCTS."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def min_max_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise Min-Max normalization mapping activations strictly into [0, 1]^d."""
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    normed = (x - x_min) / (x_max - x_min + eps)
    return torch.clamp(normed, 0.0, 1.0)


class SubStateTokenizer(nn.Module):
    """Tokenizes continuous observations into K_tok feature tokens with optional context."""

    def __init__(
        self,
        obs_dim: int = 27,
        k_tok: int = 8,
        d_model: int = 256,
        context_dim: int = 18,
        history_len: int = 8,
        max_action_dim: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.k_tok = k_tok
        self.d_model = d_model
        self.context_dim = context_dim
        self.history_len = history_len
        self.max_action_dim = max_action_dim

        self.token_proj = nn.Linear(obs_dim, k_tok * d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, k_tok, d_model) * 0.02)

        # Context conditioning: either explicit normalized tweak vector w_e/1000 or history adapter
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.history_adapter = nn.Sequential(
            nn.Linear(history_len * (obs_dim + max_action_dim), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.context_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _align_obs_dim(self, obs: torch.Tensor) -> torch.Tensor:
        """Pads or truncates the last dimension of obs to match self.obs_dim."""
        cur_dim = obs.shape[-1]
        if cur_dim == self.obs_dim:
            return obs
        if cur_dim < self.obs_dim:
            pad = torch.zeros(
                *obs.shape[:-1], self.obs_dim - cur_dim, device=obs.device, dtype=obs.dtype
            )
            return torch.cat([obs, pad], dim=-1)
        return obs[..., : self.obs_dim]

    def _align_context_dim(self, ctx: torch.Tensor) -> torch.Tensor:
        """Pads or truncates the last dimension of ctx to match self.context_dim."""
        cur_dim = ctx.shape[-1]
        if cur_dim == self.context_dim:
            return ctx
        if cur_dim < self.context_dim:
            pad = torch.zeros(
                *ctx.shape[:-1], self.context_dim - cur_dim, device=ctx.device, dtype=ctx.dtype
            )
            return torch.cat([ctx, pad], dim=-1)
        return ctx[..., : self.context_dim]

    def _align_history_dim(self, hist: torch.Tensor) -> torch.Tensor:
        """Pads or truncates the flattened dimension of history to match expected input."""
        expected_dim = self.history_len * (self.obs_dim + self.max_action_dim)
        cur_dim = hist.shape[-1]
        if cur_dim == expected_dim:
            return hist
        if cur_dim < expected_dim:
            pad = torch.zeros(
                *hist.shape[:-1], expected_dim - cur_dim, device=hist.device, dtype=hist.dtype
            )
            return torch.cat([hist, pad], dim=-1)
        return hist[..., : expected_dim]

    def forward(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Maps observation (B, D_s) -> token matrix X^{(0)} of shape (B, K_tok [+1/2], d_model)."""
        obs = self._align_obs_dim(obs)
        bsz = obs.shape[0]
        tokens = self.token_proj(obs).view(bsz, self.k_tok, self.d_model)
        tokens = tokens + self.pos_embedding

        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0).expand(bsz, -1)
            context = self._align_context_dim(context)
            ctx_tok = self.context_proj(context).unsqueeze(1) + self.context_pos_embedding
            tokens = torch.cat([tokens, ctx_tok], dim=1)

        if history is not None:
            if history.dim() == 3:
                history = history.reshape(bsz, -1)
            history = self._align_history_dim(history)
            hist_tok = self.history_adapter(history).unsqueeze(1) + self.context_pos_embedding
            tokens = torch.cat([tokens, hist_tok], dim=1)

        return tokens


class PreLNMHSABlock(nn.Module):
    """Single Pre-LayerNorm Multi-Head Self-Attention Encoder Block with 4x GELU FFN."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.mhsa(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SelfAttentionBackbone(nn.Module):
    """Transformer Self-Attention Backbone producing penultimate representation z_t in [0, 1]^d."""

    def __init__(
        self,
        obs_dim: int = 27,
        k_tok: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        l_enc: int = 3,
        latent_dim: int = 256,
        context_dim: int = 18,
        history_len: int = 8,
        max_action_dim: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.k_tok = k_tok
        self.d_model = d_model
        self.n_heads = n_heads
        self.l_enc = l_enc
        self.latent_dim = latent_dim

        self.tokenizer = SubStateTokenizer(
            obs_dim=obs_dim,
            k_tok=k_tok,
            d_model=d_model,
            context_dim=context_dim,
            history_len=history_len,
            max_action_dim=max_action_dim,
        )
        self.blocks = nn.ModuleList(
            [PreLNMHSABlock(d_model=d_model, n_heads=n_heads) for _ in range(l_enc)]
        )
        self.pool_ln = nn.LayerNorm(d_model)
        self.attn_pool_Scorer = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, latent_dim)
        self.out_ln = nn.LayerNorm(latent_dim)

    def forward(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes normalized penultimate representation z_t in [0, 1]^(B x d)."""
        squeeze_out = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_out = True

        x = self.tokenizer(obs, context=context, history=history)
        for block in self.blocks:
            x = block(x, key_padding_mask=token_mask)

        x_norm = self.pool_ln(x)
        attn_logits = self.attn_pool_Scorer(x_norm)
        if token_mask is not None:
            attn_logits = attn_logits.masked_fill(token_mask.unsqueeze(-1), float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn_weights * x_norm, dim=1)

        z_tilde = self.out_ln(self.proj(pooled))
        z_t = min_max_normalize(z_tilde, eps=1e-6)

        if squeeze_out:
            return z_t.squeeze(0)
        return z_t
