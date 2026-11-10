"""Hyperparameter Importance Evaluation (PED-ANOVA + fANOVA + Prior Blending)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import optuna
from optuna.importance import FanovaImportanceEvaluator, PedAnovaImportanceEvaluator
from optuna.trial import TrialState

from experiments.hpo.search_space import HYPERPARAM_SPECS, PRIOR_TIER_WEIGHTS, SPECS_BY_NAME


@dataclass
class ImportanceResult:
    """Blended importance record for a single hyperparameter."""

    name: str
    tier: str
    prior_weight: float
    ped_anova_score: float
    fanova_score: float
    empirical_score: float
    blended_score: float
    promoted_or_active: bool


def _rank_corr_fallback(study: optuna.Study, param_names: Sequence[str]) -> Dict[str, float]:
    """Computes absolute Spearman rank-correlation importance when < 8 completed trials exist."""
    completed = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if len(completed) < 3:
        return {name: 1.0 / max(1, len(param_names)) for name in param_names}

    y = np.array([float(t.value) for t in completed], dtype=np.float64)
    y_ranks = np.argsort(np.argsort(y)).astype(np.float64)
    y_std = float(np.std(y_ranks))
    if y_std < 1e-8:
        return {name: 1.0 / max(1, len(param_names)) for name in param_names}

    scores: Dict[str, float] = {}
    for name in param_names:
        xs = []
        ys = []
        for idx, t in enumerate(completed):
            if name in t.params:
                v = t.params[name]
                xs.append(float(v) if isinstance(v, (int, float)) else float(hash(v) % 997))
                ys.append(y_ranks[idx])
        if len(xs) < 3:
            scores[name] = 0.0
            continue
        x_arr = np.asarray(xs, dtype=np.float64)
        x_ranks = np.argsort(np.argsort(x_arr)).astype(np.float64)
        x_std = float(np.std(x_ranks))
        if x_std < 1e-8:
            scores[name] = 0.0
            continue
        y_sub = np.asarray(ys, dtype=np.float64)
        corr = abs(
            float(
                np.mean((x_ranks - np.mean(x_ranks)) * (y_sub - np.mean(y_sub)))
                / (x_std * float(np.std(y_sub)) + 1e-8)
            )
        )
        scores[name] = corr

    total = sum(scores.values())
    if total <= 1e-9:
        return {name: 1.0 / max(1, len(param_names)) for name in param_names}
    return {k: v / total for k, v in scores.items()}


def compute_blended_importances(
    study: optuna.Study,
    prior_blend: float = 0.50,
) -> List[ImportanceResult]:
    """Computes blended hyperparameter importances combining prior tiers with PED-ANOVA and fANOVA.

    PED-ANOVA (Watanabe et al., 2023) measures importance specifically within the top-performing
    quantile of trials, while fANOVA (Hutter et al., 2014) measures global variance contribution
    via a random forest surrogate. Blending with prior domain knowledge guarantees stability even
    at moderate trial counts.
    """
    all_names = [s.name for s in HYPERPARAM_SPECS]
    completed = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]

    ped_scores: Dict[str, float] = {}
    fanova_scores: Dict[str, float] = {}

    if len(completed) >= 8:
        try:
            ped_eval = PedAnovaImportanceEvaluator(baseline_quantile=0.25)
            ped_raw = optuna.importance.get_param_importances(
                study, evaluator=ped_eval, normalize=True
            )
            ped_scores = {k: float(v) for k, v in ped_raw.items()}
        except Exception:
            ped_scores = _rank_corr_fallback(study, all_names)

        try:
            fanova_eval = FanovaImportanceEvaluator(n_trees=32, max_depth=8, seed=42)
            fanova_raw = optuna.importance.get_param_importances(
                study, evaluator=fanova_eval, normalize=True
            )
            fanova_scores = {k: float(v) for k, v in fanova_raw.items()}
        except Exception:
            fanova_scores = dict(ped_scores)
    else:
        fallback = _rank_corr_fallback(study, all_names)
        ped_scores = dict(fallback)
        fanova_scores = dict(fallback)

    # Normalize prior weights across all parameters so prior and empirical sums match
    raw_priors = {
        s.name: PRIOR_TIER_WEIGHTS.get(s.tier, 0.2) for s in HYPERPARAM_SPECS
    }
    prior_sum = sum(raw_priors.values()) or 1.0
    norm_priors = {k: v / prior_sum for k, v in raw_priors.items()}

    results: List[ImportanceResult] = []
    for spec in HYPERPARAM_SPECS:
        name = spec.name
        p_w = norm_priors[name]
        ped_s = float(ped_scores.get(name, 0.0))
        fan_s = float(fanova_scores.get(name, 0.0))
        emp_s = 0.60 * ped_s + 0.40 * fan_s
        blended = prior_blend * p_w + (1.0 - prior_blend) * emp_s
        results.append(
            ImportanceResult(
                name=name,
                tier=spec.tier,
                prior_weight=PRIOR_TIER_WEIGHTS.get(spec.tier, 0.2),
                ped_anova_score=ped_s,
                fanova_score=fan_s,
                empirical_score=emp_s,
                blended_score=blended,
                promoted_or_active=(spec.tier == "A"),
            )
        )

    # Normalize blended_score to sum to 1.0 and sort descending
    total_blended = sum(r.blended_score for r in results) or 1.0
    for r in results:
        r.blended_score = r.blended_score / total_blended

    results.sort(key=lambda r: r.blended_score, reverse=True)
    return results


def partition_parameters_by_importance(
    importances: Sequence[ImportanceResult],
    max_focused_params: int = 16,
    max_cma_params: int = 10,
) -> Tuple[List[str], List[str], List[str]]:
    """Partitions hyperparameters into:
    1. `focused_names`: Tier A parameters + highest-importance promoted Tier B/C parameters (for Stage 2).
    2. `cma_continuous_names`: Top continuous/integer parameters for Stage 3a CMA-ES fine-tuning.
    3. `minor_refinement_names`: Remaining secondary/Tier-C parameters for Stage 3b local refinement.
    """
    focused: List[str] = []
    # Always retain Tier A parameters unless their empirical + blended score is near the very bottom
    for r in importances:
        if r.tier == "A":
            focused.append(r.name)
            r.promoted_or_active = True

    # Promote top non-A parameters by blended importance up to max_focused_params
    for r in importances:
        if len(focused) >= max_focused_params:
            break
        if r.name not in focused:
            focused.append(r.name)
            r.promoted_or_active = True

    # Top continuous/int parameters for CMA-ES local fine-tuning
    cma_names: List[str] = []
    for r in importances:
        spec = SPECS_BY_NAME[r.name]
        if spec.kind in ("float", "int") and len(cma_names) < max_cma_params:
            cma_names.append(r.name)

    focused_set = set(focused)
    minor_names = [r.name for r in importances if r.name not in focused_set]
    if not minor_names:
        minor_names = [s.name for s in HYPERPARAM_SPECS if s.tier == "C"]

    return focused, cma_names, minor_names
