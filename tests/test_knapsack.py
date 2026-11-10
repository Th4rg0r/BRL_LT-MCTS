"""Verification Gate 2: Differentiable Knapsack Projection Tests (brl_core/knapsack_proj.py)."""

from __future__ import annotations

import numpy as np
import torch

from brl_core.knapsack_proj import DifferentiableKnapsackProjection


def test_gate2_knapsack_constraints_and_100_gradchecks() -> None:
    """Executes torch.autograd.gradcheck across 100 random (L, B, C, w_raw) combinations."""
    rng = np.random.default_rng(12345)
    torch.manual_seed(12345)

    num_combinations = 100
    P = 12

    for i in range(num_combinations):
        # Generate random limits L in [50, 1000]^12 and costs C in [0.1, 10.0]^12
        limits_np = rng.uniform(50.0, 1000.0, size=(P,))
        costs_np = rng.uniform(0.1, 10.0, size=(P,))

        # Generate raw tweaks strictly inside box (0, L_i) to avoid non-differentiable kinks
        # at 0 or L_i, and ensure lam_mid * c_i != w_raw_i kinks
        u_np = rng.uniform(0.15, 0.85, size=(P,))
        w_raw_np = limits_np * u_np
        raw_spend = float(np.sum(w_raw_np * costs_np))

        # Alternate between budget-exceeding (active KKT hyperplane) and non-exceeding regimes
        if i % 4 == 0:
            budget_val = raw_spend * rng.uniform(1.15, 1.50)
        else:
            # Ensure budget does not land exactly on a kink point w_raw_k / c_k
            for _ in range(20):
                budget_val = raw_spend * rng.uniform(0.25, 0.85)
                l_t = torch.tensor(limits_np, dtype=torch.float64)
                c_t = torch.tensor(costs_np, dtype=torch.float64)
                b_t = torch.tensor([budget_val], dtype=torch.float64)
                w_t = torch.tensor(w_raw_np, dtype=torch.float64)
                w_p = DifferentiableKnapsackProjection.apply(w_t, c_t, b_t, l_t)
                # Verify away from kink (no component in (0, 1e-3))
                near_zero = (w_p > 0.0) & (w_p < 1e-2)
                if not torch.any(near_zero) and torch.any(w_p > 1e-2):
                    break

        w_raw = torch.tensor(w_raw_np, dtype=torch.float64, requires_grad=True)
        costs = torch.tensor(costs_np, dtype=torch.float64, requires_grad=False)
        budget = torch.tensor([budget_val], dtype=torch.float64, requires_grad=False)
        limits = torch.tensor(limits_np, dtype=torch.float64, requires_grad=False)

        # 1. Verify primal feasibility constraints
        w_proj = DifferentiableKnapsackProjection.apply(w_raw, costs, budget, limits)
        max_limit_viol = float((w_proj - limits).max().item())
        min_val = float(w_proj.min().item())
        total_projected_cost = float((w_proj * costs).sum().item())

        assert max_limit_viol <= 1e-5, f"Limit violated at iter {i}: {max_limit_viol}"
        assert min_val >= 0.0, f"Negativity violation at iter {i}: {min_val}"
        assert (
            total_projected_cost <= budget_val + 1e-4
        ), f"Budget violated at iter {i}: {total_projected_cost} > {budget_val}"

        # 2. Verify analytical gradient against finite differences via torch.autograd.gradcheck
        passed = torch.autograd.gradcheck(
            lambda w: DifferentiableKnapsackProjection.apply(w, costs, budget, limits),
            (w_raw,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
        assert passed, f"gradcheck failed on random configuration {i}"
