"""Differentiable Bounded Knapsack Projection Layer via KKT Bisection and Implicit Grad."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class DifferentiableKnapsackProjection(torch.autograd.Function):
    """Solves min_w 0.5 * ||w - w_raw||_2^2 s.t. 0 <= w_i <= L_i and sum_i c_i w_i <= B.

    Uses vectorized bisection on the KKT Lagrange multiplier lambda* in forward()
    and exact implicit differentiation along the active hyperplane in backward().
    """

    @staticmethod
    def forward(
        ctx,
        w_raw: torch.Tensor,
        costs: torch.Tensor,
        budget: torch.Tensor,
        limits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze_batch = False
        if w_raw.dim() == 1:
            w_raw = w_raw.unsqueeze(0)
            squeeze_batch = True
        if costs.dim() == 1:
            costs = costs.unsqueeze(0).expand_as(w_raw)
        if budget.dim() == 0:
            budget = budget.view(1, 1).expand(w_raw.shape[0], 1)
        elif budget.dim() == 1:
            budget = budget.unsqueeze(-1)

        if limits is not None:
            if limits.dim() == 1:
                limits = limits.unsqueeze(0).expand_as(w_raw)
            zero_t = torch.zeros_like(limits)
            w_box = torch.clamp(w_raw, min=zero_t, max=limits)
        else:
            zero_t = torch.zeros_like(w_raw)
            w_box = torch.clamp(w_raw, min=zero_t)

        total_cost = (w_box * costs).sum(dim=-1, keepdim=True)
        exceeded = total_cost > budget

        # Bisection bounds for Lagrange multiplier lambda* >= 0
        eps = 1e-12 if w_raw.dtype == torch.float64 else 1e-7
        lam_low = torch.zeros_like(budget)
        lam_high = (w_box / (costs + eps)).max(dim=-1, keepdim=True)[0] + 1e-6

        # 25 bisection iterations locate the exact KKT active set I_act
        lam_mid = 0.5 * (lam_low + lam_high)
        for _ in range(25):
            lam_mid = 0.5 * (lam_low + lam_high)
            if limits is not None:
                w_cand = torch.clamp(w_raw - lam_mid * costs, min=zero_t, max=limits)
            else:
                w_cand = torch.clamp(w_box - lam_mid * costs, min=zero_t)
            cand_cost = (w_cand * costs).sum(dim=-1, keepdim=True)
            cond = cand_cost > budget
            lam_low = torch.where(cond, lam_mid, lam_low)
            lam_high = torch.where(cond, lam_high, lam_mid)

        lam_bisect = 0.5 * (lam_low + lam_high)
        if limits is not None:
            w_bisect = torch.clamp(w_raw - lam_bisect * costs, min=zero_t, max=limits)
            active_mask = ((w_bisect > 0.0) & (w_bisect < limits)).to(w_raw.dtype)
            upper_mask = (w_bisect >= limits).to(w_raw.dtype)
            upper_spend = (upper_mask * limits * costs).sum(dim=-1, keepdim=True)
            active_raw_spend = (active_mask * w_raw * costs).sum(dim=-1, keepdim=True)
        else:
            w_bisect = torch.clamp(w_box - lam_bisect * costs, min=zero_t)
            active_mask = (w_bisect > 0.0).to(w_raw.dtype)
            upper_spend = torch.zeros_like(budget)
            active_raw_spend = (active_mask * w_box * costs).sum(dim=-1, keepdim=True)

        # Exact closed-form KKT multiplier on the identified active hyperplane piece
        active_c2 = (active_mask * costs.pow(2)).sum(dim=-1, keepdim=True)
        lam_exact = (active_raw_spend + upper_spend - budget) / (active_c2 + eps)
        lam_star = torch.where(active_c2 > eps, torch.clamp(lam_exact, min=0.0), lam_bisect)

        if limits is not None:
            w_proj_ex = torch.clamp(w_raw - lam_star * costs, min=zero_t, max=limits)
        else:
            w_proj_ex = torch.clamp(w_box - lam_star * costs, min=zero_t)

        w_out = torch.where(exceeded, w_proj_ex, w_box)

        ctx.save_for_backward(
            w_raw,
            w_out,
            costs,
            budget,
            lam_star,
            exceeded,
            limits if limits is not None else torch.empty(0, device=w_raw.device),
        )
        ctx.has_limits = limits is not None
        ctx.squeeze_batch = squeeze_batch

        if squeeze_batch:
            return w_out.squeeze(0)
        return w_out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        w_raw, w_out, costs, budget, lam_star, exceeded, limits = ctx.saved_tensors
        squeeze_batch = ctx.squeeze_batch
        if squeeze_batch:
            grad_output = grad_output.unsqueeze(0)

        eps = 1e-12 if w_raw.dtype == torch.float64 else 1e-6

        # Active set I_act = {i : 0 < w_out_i < L_i}
        if ctx.has_limits:
            active_ex = ((w_out > 0.0) & (w_out < limits)).to(grad_output.dtype)
            upper_ex = (w_out >= limits).to(grad_output.dtype)
            active_non_ex = ((w_raw >= 0.0) & (w_raw <= limits)).to(grad_output.dtype)
        else:
            active_ex = (w_out > 0.0).to(grad_output.dtype)
            upper_ex = torch.zeros_like(active_ex)
            active_non_ex = (w_raw >= 0.0).to(grad_output.dtype)

        # Hyperplane orthogonal projection for budget-exceeding batch items:
        # (J^T g)_i = active_i * g_i - active_i * c_i * (sum_k active_k c_k g_k) / (sum_k active_k c_k^2)
        denom = (active_ex * costs.pow(2)).sum(dim=-1, keepdim=True) + eps
        proj_scalar = (active_ex * costs * grad_output).sum(dim=-1, keepdim=True) / denom
        grad_w_ex = active_ex * grad_output - active_ex * costs * proj_scalar
        grad_w_non_ex = active_non_ex * grad_output
        grad_w_raw = torch.where(exceeded, grad_w_ex, grad_w_non_ex)

        grad_costs = None
        if ctx.needs_input_grad[1]:
            # Analytical gradient w.r.t. costs
            d_lam_dc = (active_ex * (w_out - lam_star * costs) + upper_ex * w_out) / denom
            grad_c_ex = -lam_star * active_ex * grad_output - d_lam_dc * (
                active_ex * costs * grad_output
            ).sum(dim=-1, keepdim=True)
            grad_costs = torch.where(exceeded, grad_c_ex, torch.zeros_like(costs))

        grad_budget = None
        if ctx.needs_input_grad[2]:
            # Analytical gradient w.r.t. budget
            grad_b_ex = proj_scalar
            grad_budget = torch.where(exceeded, grad_b_ex, torch.zeros_like(budget))

        grad_limits = None
        if ctx.has_limits and ctx.needs_input_grad[3]:
            grad_l_ex = upper_ex * grad_output - upper_ex * costs * proj_scalar
            grad_l_non_ex = (w_raw > limits).to(grad_output.dtype) * grad_output
            grad_limits = torch.where(exceeded, grad_l_ex, grad_l_non_ex)

        if squeeze_batch:
            grad_w_raw = grad_w_raw.squeeze(0)
            if grad_costs is not None:
                grad_costs = grad_costs.squeeze(0)
            if grad_budget is not None:
                grad_budget = grad_budget.squeeze(0)
            if grad_limits is not None:
                grad_limits = grad_limits.squeeze(0)

        return grad_w_raw, grad_costs, grad_budget, grad_limits


class KnapsackProjectionLayer(nn.Module):
    """Module wrapper around DifferentiableKnapsackProjection."""

    def forward(
        self,
        w_raw: torch.Tensor,
        costs: torch.Tensor,
        budget: torch.Tensor,
        limits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if limits is not None:
            w_raw_clamped = torch.clamp(w_raw, min=0.0, max=limits)
        else:
            w_raw_clamped = torch.clamp(w_raw, min=0.0)
        return DifferentiableKnapsackProjection.apply(
            w_raw_clamped, costs, budget, limits
        )
