"""Tiered Search Space, Range Narrowing, and Model Cost Estimator for HPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import optuna

from config import BRLConfig, LAMCTSConfig, SystemConfig


PRIOR_TIER_WEIGHTS: Dict[str, float] = {
    "A": 1.00,  # Critical parameters: emphasized across all stages
    "B": 0.55,  # Important parameters: screened in Stage 1-2, promoted if impactful
    "C": 0.20,  # Minor parameters: narrow ranges, frozen early, refined in Stage 3b
}


@dataclass(frozen=True)
class HyperparamSpec:
    """Specification of a single tunable hyperparameter."""

    name: str
    section: str  # "la_mcts", "brl", or "coupled"
    field: str
    tier: str  # "A", "B", or "C"
    kind: str  # "float", "int", or "categorical"
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    log: bool = False
    cost_sensitive: bool = False
    description: str = ""


HYPERPARAM_SPECS: Tuple[HyperparamSpec, ...] = (
    # =========================================================================
    # TIER A: Critical Parameters (Highest impact on learning & planning)
    # =========================================================================
    HyperparamSpec(
        name="lr",
        section="la_mcts",
        field="lr",
        tier="A",
        kind="float",
        default=3e-4,
        low=5e-5,
        high=2e-3,
        log=True,
        description="Agent AdamW learning rate",
    ),
    HyperparamSpec(
        name="train_every_steps",
        section="la_mcts",
        field="train_every_steps",
        tier="A",
        kind="int",
        default=25,
        low=5,
        high=125,
        log=True,
        cost_sensitive=True,
        description="Environment steps between agent gradient updates",
    ),
    HyperparamSpec(
        name="batch_size",
        section="la_mcts",
        field="batch_size",
        tier="A",
        kind="categorical",
        default=64,
        choices=(32, 64, 128, 256),
        cost_sensitive=True,
        description="Unrolled training mini-batch size",
    ),
    HyperparamSpec(
        name="n_sim",
        section="la_mcts",
        field="n_sim",
        tier="A",
        kind="categorical",
        default=20,
        choices=(8, 12, 16, 20, 32, 50),
        cost_sensitive=True,
        description="MCTS simulations per control step",
    ),
    HyperparamSpec(
        name="k_act",
        section="la_mcts",
        field="k_act",
        tier="A",
        kind="categorical",
        default=16,
        choices=(4, 8, 12, 16, 24),
        cost_sensitive=True,
        description="Sampled continuous candidate actions per MCTS node",
    ),
    HyperparamSpec(
        name="unroll_steps",
        section="la_mcts",
        field="unroll_steps",
        tier="A",
        kind="int",
        default=5,
        low=3,
        high=7,
        cost_sensitive=True,
        description="Latent dynamics unroll horizon K",
    ),
    HyperparamSpec(
        name="td_steps",
        section="la_mcts",
        field="td_steps",
        tier="A",
        kind="int",
        default=5,
        low=3,
        high=10,
        description="n_td-step value bootstrap horizon",
    ),
    HyperparamSpec(
        name="gamma",
        section="la_mcts",
        field="gamma",
        tier="A",
        kind="categorical",
        default=0.99,
        choices=(0.97, 0.98, 0.99, 0.995),
        description="Discount factor gamma",
    ),
    HyperparamSpec(
        name="lambda_rep",
        section="la_mcts",
        field="lambda_rep",
        tier="A",
        kind="float",
        default=4.0,
        low=0.5,
        high=8.0,
        log=True,
        description="SimSiam representation consistency loss coefficient",
    ),
    # =========================================================================
    # TIER B: Important Architectural, MCTS Exploration & Curriculum Parameters
    # =========================================================================
    HyperparamSpec(
        name="model_dim",
        section="coupled",
        field="model_dim",
        tier="B",
        kind="categorical",
        default=256,
        choices=(64, 128, 256, 384),
        cost_sensitive=True,
        description="Coupled Transformer d_model and penultimate latent_dim",
    ),
    HyperparamSpec(
        name="n_heads",
        section="la_mcts",
        field="n_heads",
        tier="B",
        kind="categorical",
        default=4,
        choices=(2, 4, 8),
        cost_sensitive=True,
        description="Multi-Head Self-Attention head count",
    ),
    HyperparamSpec(
        name="l_enc",
        section="la_mcts",
        field="l_enc",
        tier="B",
        kind="int",
        default=3,
        low=1,
        high=4,
        cost_sensitive=True,
        description="Number of Self-Attention encoder layers",
    ),
    HyperparamSpec(
        name="k_tok",
        section="la_mcts",
        field="k_tok",
        tier="B",
        kind="categorical",
        default=8,
        choices=(4, 8, 12),
        cost_sensitive=True,
        description="Number of sub-state feature tokens",
    ),
    HyperparamSpec(
        name="l_dyn",
        section="la_mcts",
        field="l_dyn",
        tier="B",
        kind="int",
        default=3,
        low=1,
        high=4,
        cost_sensitive=True,
        description="Number of FiLM residual blocks in representation trainer",
    ),
    HyperparamSpec(
        name="num_peaks",
        section="la_mcts",
        field="num_peaks",
        tier="B",
        kind="int",
        default=5,
        low=2,
        high=8,
        description="Number of Gaussian Mixture components M",
    ),
    HyperparamSpec(
        name="lambda_v",
        section="la_mcts",
        field="lambda_v",
        tier="B",
        kind="float",
        default=0.5,
        low=0.25,
        high=2.0,
        log=True,
        description="Critic value loss weight",
    ),
    HyperparamSpec(
        name="lambda_p",
        section="la_mcts",
        field="lambda_p",
        tier="B",
        kind="float",
        default=1.0,
        low=0.25,
        high=2.0,
        log=True,
        description="GMM policy KL loss weight",
    ),
    HyperparamSpec(
        name="lambda_r",
        section="la_mcts",
        field="lambda_r",
        tier="B",
        kind="float",
        default=1.0,
        low=0.5,
        high=2.0,
        description="Latent dynamics reward prediction loss weight",
    ),
    HyperparamSpec(
        name="tau_act_train",
        section="la_mcts",
        field="tau_act_train",
        tier="B",
        kind="float",
        default=1.0,
        low=0.5,
        high=1.5,
        description="Root visit count temperature during training",
    ),
    HyperparamSpec(
        name="root_tau_beta",
        section="la_mcts",
        field="root_tau_beta",
        tier="B",
        kind="float",
        default=1.25,
        low=1.0,
        high=2.0,
        description="Proposal distribution temperature at MCTS root",
    ),
    HyperparamSpec(
        name="c_1",
        section="la_mcts",
        field="c_1",
        tier="B",
        kind="float",
        default=1.25,
        low=0.75,
        high=2.5,
        description="Sampled PUCT exploration constant c_1",
    ),
    HyperparamSpec(
        name="tau_s",
        section="brl",
        field="tau_s",
        tier="B",
        kind="float",
        default=500.0,
        low=200.0,
        high=1000.0,
        log=True,
        description="Dominance score sigmoid temperature tau_s",
    ),
    HyperparamSpec(
        name="balancer_noise_std",
        section="brl",
        field="balancer_noise_std",
        tier="B",
        kind="float",
        default=0.05,
        low=0.01,
        high=0.15,
        log=True,
        description="DDPG Balancer exploration noise standard deviation",
    ),
    HyperparamSpec(
        name="ema_beta_w",
        section="brl",
        field="ema_beta_w",
        tier="B",
        kind="float",
        default=0.05,
        low=0.02,
        high=0.15,
        log=True,
        description="Exponential moving average rate for win-rate tracking",
    ),
    # =========================================================================
    # TIER C: Minor / Fine-Tuning Parameters (Narrow priors, refined later)
    # =========================================================================
    HyperparamSpec(
        name="beta_mse",
        section="la_mcts",
        field="beta_mse",
        tier="C",
        kind="float",
        default=1.0,
        low=0.25,
        high=2.5,
        description="L2 MSE weight inside SimSiam representation loss",
    ),
    HyperparamSpec(
        name="proj_dim",
        section="la_mcts",
        field="proj_dim",
        tier="C",
        kind="categorical",
        default=512,
        choices=(256, 512),
        cost_sensitive=True,
        description="SimSiam projector hidden/output dimension",
    ),
    HyperparamSpec(
        name="pred_dim",
        section="la_mcts",
        field="pred_dim",
        tier="C",
        kind="categorical",
        default=256,
        choices=(128, 256),
        cost_sensitive=True,
        description="SimSiam predictor hidden dimension",
    ),
    HyperparamSpec(
        name="c_reg",
        section="la_mcts",
        field="c_reg",
        tier="C",
        kind="float",
        default=1e-4,
        low=1e-5,
        high=5e-4,
        log=True,
        description="L2 weight decay regularization coefficient",
    ),
    HyperparamSpec(
        name="inner_tau_beta",
        section="la_mcts",
        field="inner_tau_beta",
        tier="C",
        kind="float",
        default=1.0,
        low=0.85,
        high=1.25,
        description="Proposal temperature at inner MCTS nodes",
    ),
    HyperparamSpec(
        name="reanalyze_every",
        section="la_mcts",
        field="reanalyze_every",
        tier="C",
        kind="int",
        default=2,
        low=1,
        high=5,
        cost_sensitive=True,
        description="Episode interval between MuZero Reanalyze updates",
    ),
    HyperparamSpec(
        name="reanalyze_states",
        section="la_mcts",
        field="reanalyze_states",
        tier="C",
        kind="categorical",
        default=4,
        choices=(2, 4, 8),
        cost_sensitive=True,
        description="Number of states re-evaluated per Reanalyze pass",
    ),
    HyperparamSpec(
        name="buffer_capacity",
        section="la_mcts",
        field="buffer_capacity",
        tier="C",
        kind="categorical",
        default=50000,
        choices=(25000, 50000, 75000),
        cost_sensitive=True,
        description="Replay buffer capacity in environment steps",
    ),
    HyperparamSpec(
        name="alpha_eq",
        section="brl",
        field="alpha_eq",
        tier="C",
        kind="float",
        default=10.0,
        low=5.0,
        high=20.0,
        description="Balancer equilibrium penalty weight",
    ),
    HyperparamSpec(
        name="alpha_prog",
        section="brl",
        field="alpha_prog",
        tier="C",
        kind="float",
        default=5.0,
        low=2.0,
        high=10.0,
        description="Balancer curriculum progression reward weight",
    ),
    HyperparamSpec(
        name="sigma_zpd",
        section="brl",
        field="sigma_zpd",
        tier="C",
        kind="float",
        default=0.15,
        low=0.08,
        high=0.25,
        description="Zone of Proximal Development Gaussian gate width",
    ),
    HyperparamSpec(
        name="lambda_entropy",
        section="brl",
        field="lambda_entropy",
        tier="C",
        kind="float",
        default=0.5,
        low=0.2,
        high=1.0,
        description="Balancer budget-allocation entropy bonus weight",
    ),
    HyperparamSpec(
        name="balancer_lr",
        section="brl",
        field="balancer_lr",
        tier="C",
        kind="float",
        default=1e-4,
        low=3e-5,
        high=5e-4,
        log=True,
        description="DDPG Balancer learning rate",
    ),
    HyperparamSpec(
        name="tweaker_lr",
        section="brl",
        field="tweaker_lr",
        tier="C",
        kind="float",
        default=3e-4,
        low=1e-4,
        high=1e-3,
        log=True,
        description="Stochastic Tweaker PPO learning rate",
    ),
    HyperparamSpec(
        name="eta_viol",
        section="brl",
        field="eta_viol",
        tier="C",
        kind="float",
        default=2.0,
        low=0.5,
        high=5.0,
        description="Tweaker pre-projection budget violation penalty weight",
    ),
    HyperparamSpec(
        name="beta_ent",
        section="brl",
        field="beta_ent",
        tier="C",
        kind="float",
        default=0.01,
        low=0.002,
        high=0.05,
        log=True,
        description="Tweaker Beta policy entropy bonus coefficient",
    ),
)

SPECS_BY_NAME: Dict[str, HyperparamSpec] = {s.name: s for s in HYPERPARAM_SPECS}


def default_param_dict(base_cfg: Optional[SystemConfig] = None) -> Dict[str, Any]:
    """Returns the dictionary of nominal default values for all tunable parameters."""
    cfg = base_cfg or SystemConfig()
    out: Dict[str, Any] = {}
    for spec in HYPERPARAM_SPECS:
        if spec.section == "coupled" and spec.name == "model_dim":
            out[spec.name] = int(cfg.la_mcts.d_model)
        elif spec.section == "la_mcts":
            val = getattr(cfg.la_mcts, spec.field, spec.default)
            if spec.name == "train_every_steps" and int(val) <= 0:
                val = spec.default
            out[spec.name] = val
        elif spec.section == "brl":
            out[spec.name] = getattr(cfg.brl, spec.field, spec.default)
        else:
            out[spec.name] = spec.default
    return out


def apply_params_to_config(
    base_cfg: SystemConfig, params: Dict[str, Any]
) -> SystemConfig:
    """Applies a flat hyperparameter dictionary onto a SystemConfig instance."""
    la_updates: Dict[str, Any] = {}
    brl_updates: Dict[str, Any] = {}

    for name, val in params.items():
        spec = SPECS_BY_NAME.get(name)
        if spec is None:
            continue
        cast_val: Any = val
        if spec.kind == "int" or (
            spec.kind == "categorical" and isinstance(spec.default, int)
        ):
            cast_val = int(round(float(val)))
        elif spec.kind == "float" or (
            spec.kind == "categorical" and isinstance(spec.default, float)
        ):
            cast_val = float(val)

        if spec.section == "coupled" and spec.name == "model_dim":
            la_updates["d_model"] = int(cast_val)
            la_updates["latent_dim"] = int(cast_val)
        elif spec.section == "la_mcts":
            la_updates[spec.field] = cast_val
            if spec.field == "n_sim":
                la_updates["n_sim_eval"] = max(int(cast_val), int(cast_val) * 2)
        elif spec.section == "brl":
            brl_updates[spec.field] = cast_val

    # Ensure n_heads divides d_model cleanly
    d_model = int(la_updates.get("d_model", base_cfg.la_mcts.d_model))
    n_heads = int(la_updates.get("n_heads", base_cfg.la_mcts.n_heads))
    if d_model % n_heads != 0:
        for candidate in (4, 2, 8, 1):
            if d_model % candidate == 0:
                la_updates["n_heads"] = candidate
                break

    new_la: LAMCTSConfig = replace(base_cfg.la_mcts, **la_updates)
    new_brl: BRLConfig = replace(base_cfg.brl, **brl_updates)
    return replace(base_cfg, la_mcts=new_la, brl=new_brl)


def estimate_static_model_stats(
    cfg: SystemConfig, max_steps: int = 250
) -> Dict[str, float]:
    """ analytically estimates parameter count, relative FLOP factor, and peak VRAM (MB).

    Used to pre-screen configurations before running rollouts and to evaluate model parsimony.
    """
    la = cfg.la_mcts
    d = int(la.d_model)
    z_d = int(la.latent_dim)
    k_tok = int(la.k_tok)
    l_enc = int(la.l_enc)
    l_dyn = int(la.l_dyn)
    m_peaks = int(la.num_peaks)
    act_dim = int(cfg.ant_morph.max_action_capacity)
    proj_d = int(la.proj_dim)
    pred_d = int(la.pred_dim)

    # 1. Backbone parameter count
    tok_params = la.obs_dim * (k_tok * d) + k_tok * d + 2 * (d * d)
    mhsa_block_params = 4 * (d * d) + 2 * (4 * d * d)  # QKV+Out + 4x FFN
    backbone_params = tok_params + l_enc * mhsa_block_params + d * z_d

    # 2. Actor & Critic heads
    actor_params = z_d * m_peaks + 2 * z_d * (m_peaks * act_dim)
    critic_params = z_d * int(la.num_bins)

    # 3. Representation trainer + SimSiam projector/predictor
    rep_params = (
        (act_dim * z_d + z_d * 2 * z_d)
        + l_dyn * (2 * z_d * z_d)
        + (z_d * (z_d // 2) + (z_d // 2))
        + (z_d * proj_d + proj_d * proj_d)
        + (proj_d * pred_d + pred_d * proj_d)
    )

    # 4. Tweaker & Balancer (~400k params combined, fixed)
    meta_params = 420_000
    total_params = int(backbone_params + actor_params + critic_params + rep_params + meta_params)

    # Relative compute cost multiplier vs baseline (n_sim=20, k_act=16, d=256, train_every=25)
    train_every = int(la.train_every_steps) if int(la.train_every_steps) > 0 else max(1, max_steps // 2)
    mcts_factor = (float(la.n_sim) / 20.0) * (0.7 + 0.3 * (float(la.k_act) / 16.0))
    net_factor = ((float(d) / 256.0) ** 1.4) * (
        0.5 * (float(l_enc) / 3.0) + 0.5 * (float(l_dyn) / 3.0)
    )
    update_factor = (
        (25.0 / max(1.0, float(train_every)))
        * (float(la.batch_size) / 64.0)
        * (float(la.unroll_steps) / 5.0)
    )
    rel_cost_factor = 0.65 * (mcts_factor * net_factor) + 0.35 * (update_factor * net_factor)

    # Estimated GPU memory (MB): weights + Adam moments + unrolled batch activations + CUDA context
    weights_mb = (total_params * 4 * 4) / (1024.0**2)
    batch_act_mb = (
        float(la.batch_size)
        * float(la.unroll_steps + 1)
        * float(k_tok + 2)
        * float(d)
        * float(l_enc + l_dyn)
        * 16.0
    ) / (1024.0**2)
    cuda_overhead_mb = 450.0
    est_vram_mb = weights_mb + batch_act_mb + cuda_overhead_mb

    return {
        "param_count": float(total_params),
        "rel_cost_factor": float(rel_cost_factor),
        "est_vram_mb": float(est_vram_mb),
    }


def compute_narrowed_bounds(
    top_param_dicts: Sequence[Dict[str, Any]],
    shrink_factor: float = 0.55,
) -> Dict[str, Tuple[Any, Any]]:
    """Computes narrowed search bounds around the top-performing configurations.

    For continuous/integer parameters, centers a window of width `shrink_factor * full_span`
    around the weighted mean of `top_param_dicts` (in log space if `spec.log` is True).
    For categorical parameters, retains the union of values appearing in `top_param_dicts`
    plus the adjacent/best category so at least 2 options remain.
    """
    if not top_param_dicts:
        return {}

    narrowed: Dict[str, Tuple[Any, Any]] = {}
    n_top = len(top_param_dicts)
    # Rank weights favoring the very best trials
    weights = [1.0 / (idx + 1.0) for idx in range(n_top)]
    w_sum = sum(weights)
    norm_w = [w / w_sum for w in weights]

    for spec in HYPERPARAM_SPECS:
        vals = [d[spec.name] for d in top_param_dicts if spec.name in d]
        if not vals:
            continue

        if spec.kind == "categorical":
            assert spec.choices is not None
            seen = []
            for v in vals:
                if v not in seen:
                    seen.append(v)
            if len(seen) == 1 and len(spec.choices) > 1:
                best_v = seen[0]
                idx = list(spec.choices).index(best_v) if best_v in spec.choices else 0
                neighbor_idx = max(0, idx - 1) if idx > 0 else min(len(spec.choices) - 1, idx + 1)
                seen.append(spec.choices[neighbor_idx])
            narrowed[spec.name] = tuple(seen)
            continue

        assert spec.low is not None and spec.high is not None
        top_sub = [float(v) for v in vals[: min(3, len(vals))]]
        if spec.log:
            log_low = math.log(spec.low)
            log_high = math.log(spec.high)
            log_span = log_high - log_low
            center = sum(w * math.log(max(spec.low, float(v))) for w, v in zip(norm_w, vals))
            half_w = 0.5 * max(0.20, shrink_factor) * log_span
            new_low = max(spec.low, min(min(top_sub), math.exp(max(log_low, center - half_w))))
            new_high = min(spec.high, max(max(top_sub), math.exp(min(log_high, center + half_w))))
        else:
            span = spec.high - spec.low
            center = sum(w * float(v) for w, v in zip(norm_w, vals))
            half_w = 0.5 * max(0.20, shrink_factor) * span
            new_low = max(spec.low, min(min(top_sub), center - half_w))
            new_high = min(spec.high, max(max(top_sub), center + half_w))

        if spec.kind == "int":
            i_low = int(math.floor(new_low))
            i_high = int(math.ceil(new_high))
            if i_low >= i_high:
                i_high = min(int(spec.high), i_low + 1)
                i_low = max(int(spec.low), i_high - 1)
            narrowed[spec.name] = (i_low, i_high)
        else:
            if new_high <= new_low * (1.0 + 1e-4):
                new_high = min(spec.high, new_low * 1.1 + 1e-5)
            narrowed[spec.name] = (float(new_low), float(new_high))

    return narrowed


def sample_trial_params(
    trial: optuna.Trial,
    active_names: Sequence[str],
    frozen_params: Dict[str, Any],
    narrowed_bounds: Optional[Dict[str, Tuple[Any, Any]]] = None,
    cma_continuous_only: bool = False,
) -> Dict[str, Any]:
    """Samples `active_names` from `trial` (respecting `narrowed_bounds`) and fills the rest from `frozen_params`."""
    params: Dict[str, Any] = dict(frozen_params)
    bounds_map = narrowed_bounds or {}
    active_set = set(active_names)

    for spec in HYPERPARAM_SPECS:
        if spec.name not in active_set:
            params.setdefault(spec.name, spec.default)
            continue

        if cma_continuous_only and spec.kind == "categorical":
            params.setdefault(spec.name, frozen_params.get(spec.name, spec.default))
            continue

        if spec.kind == "categorical":
            assert spec.choices is not None
            # Keep categorical choices fixed across trials within a study to satisfy Optuna's distribution check
            params[spec.name] = trial.suggest_categorical(spec.name, list(spec.choices))
        elif spec.kind == "int":
            assert spec.low is not None and spec.high is not None
            low_i, high_i = int(spec.low), int(spec.high)
            if spec.name in bounds_map:
                low_i, high_i = int(bounds_map[spec.name][0]), int(bounds_map[spec.name][1])
            params[spec.name] = trial.suggest_int(
                spec.name, low_i, high_i, log=spec.log
            )
        else:
            assert spec.low is not None and spec.high is not None
            low_f, high_f = float(spec.low), float(spec.high)
            if spec.name in bounds_map:
                low_f, high_f = float(bounds_map[spec.name][0]), float(bounds_map[spec.name][1])
            params[spec.name] = trial.suggest_float(
                spec.name, low_f, high_f, log=spec.log
            )

    return params
