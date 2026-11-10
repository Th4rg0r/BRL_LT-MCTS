"""Central configuration dataclasses and YAML loader for LA-MCTS and BRL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings
import yaml


@dataclass
class LAMCTSConfig:
    """Configuration for Framework 1: Latent-Action MCTS."""

    obs_dim: int = 27
    max_action_dim: int = 8
    k_tok: int = 8
    d_model: int = 256
    n_heads: int = 4
    l_enc: int = 3
    latent_dim: int = 256
    num_peaks: int = 5
    log_std_bounds: Tuple[float, float] = (-5.0, 2.0)
    use_categorical_critic: bool = True
    support_min: int = -300
    support_max: int = 300
    num_bins: int = 601
    gamma: float = 0.99
    td_steps: int = 5
    l_dyn: int = 3
    proj_dim: int = 512
    pred_dim: int = 256
    unroll_steps: int = 5
    lambda_rep: float = 4.0
    beta_mse: float = 1.0
    lambda_v: float = 0.5
    lambda_r: float = 1.0
    lambda_p: float = 1.0
    c_reg: float = 1e-4
    n_sim: int = 50
    n_sim_eval: int = 100
    k_act: int = 16
    c_1: float = 1.25
    c_2: float = 19652.0
    root_tau_beta: float = 1.25
    inner_tau_beta: float = 1.0
    tau_act_train: float = 1.0
    tau_act_eval: float = 0.05
    lr: float = 3e-4
    batch_size: int = 64
    buffer_capacity: int = 50000
    context_dim: int = 12
    history_len: int = 8
    reanalyze_ratio: float = 0.25
    # Agent gradient update every N env steps (0 = auto: max_steps // 2, the original behaviour)
    train_every_steps: int = 0
    # MuZero Reanalyze every N episodes (0 = disabled) and number of states per Reanalyze pass
    reanalyze_every: int = 2
    reanalyze_states: int = 4


@dataclass
class BRLConfig:
    """Configuration for Framework 2: Balanced Reinforcement Learning."""

    num_params: int = 12
    param_max: float = 1000.0
    budget_max: float = 12000.0
    cost_min: float = 0.1
    cost_max: float = 10.0
    target_win_rate: float = 0.50
    ema_beta_w: float = 0.05
    tau_s: float = 500.0
    initial_r_ref: float = 1000.0
    benchmark_r_ref: float = 6169.54
    window_k_w: int = 20
    alpha_eq: float = 10.0
    alpha_inst: float = 2.0
    alpha_prog: float = 5.0
    sigma_zpd: float = 0.15
    lambda_entropy: float = 0.5
    balancer_lr: float = 1e-4
    balancer_tau: float = 0.005
    balancer_noise_std: float = 0.05
    tweaker_lr: float = 3e-4
    beta_score: float = 1.0
    eta_viol: float = 2.0
    beta_ent: float = 0.01
    warmup_episodes: int = 50
    brl_episodes: int = 2500
    total_episodes: int = 3000
    warmup_limit: float = 50.0
    warmup_budget: float = 100.0


@dataclass
class AntMorphConfig:
    """Configuration for Morphological Ant-v4 Environment."""

    env_id: str = "Ant-v4"
    use_contact_forces: bool = False
    max_episode_steps: int = 1000
    xml_rebuild_threshold: float = 1.0
    healthy_z_range: Tuple[float, float] = (0.2, 1.0)
    max_action_capacity: int = 12
    default_num_legs: int = 4


@dataclass
class RunConfig:
    """Run protocol defaults for experiments/train_ant_brl.py (CLI flags override these).

    A value of 0 means "derive automatically" (see train_ant_brl.main).
    """

    seeds: int = 3
    seed_offset: int = 42
    episodes: int = 0  # 0 = brl.total_episodes
    max_steps: int = 0  # 0 = ant_morph.max_episode_steps
    device: str = "cpu"
    eval_interval: int = 0  # 0 = min(10, episodes)
    parallel_seeds: int = 1
    gpu_mem_fraction: float = 0.0  # per-process CUDA memory cap (0 = no cap)


@dataclass
class SystemConfig:
    """Aggregated system configuration."""

    la_mcts: LAMCTSConfig = field(default_factory=LAMCTSConfig)
    brl: BRLConfig = field(default_factory=BRLConfig)
    ant_morph: AntMorphConfig = field(default_factory=AntMorphConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _filter_known(cls: type, raw: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Drops (with a warning) keys that the dataclass does not define."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys in [{section}]: {unknown}")
    return {k: v for k, v in raw.items() if k in known}


def config_from_dict(raw: Dict[str, Any]) -> SystemConfig:
    """Builds a SystemConfig from a (YAML-style) nested dictionary."""
    la_raw = _filter_known(LAMCTSConfig, dict(raw.get("la_mcts", {}) or {}), "la_mcts")
    if "log_std_bounds" in la_raw:
        la_raw["log_std_bounds"] = tuple(la_raw["log_std_bounds"])

    brl_raw = _filter_known(BRLConfig, dict(raw.get("brl", {}) or {}), "brl")

    ant_raw = _filter_known(AntMorphConfig, dict(raw.get("ant_morph", {}) or {}), "ant_morph")
    if "healthy_z_range" in ant_raw:
        ant_raw["healthy_z_range"] = tuple(ant_raw["healthy_z_range"])

    run_raw = _filter_known(RunConfig, dict(raw.get("run", {}) or {}), "run")

    return SystemConfig(
        la_mcts=LAMCTSConfig(**la_raw),
        brl=BRLConfig(**brl_raw),
        ant_morph=AntMorphConfig(**ant_raw),
        run=RunConfig(**run_raw),
    )


def config_to_dict(cfg: SystemConfig) -> Dict[str, Any]:
    """Converts a SystemConfig into a plain, YAML-serializable nested dictionary."""

    def _plain(v: Any) -> Any:
        if isinstance(v, tuple):
            return [_plain(x) for x in v]
        if isinstance(v, dict):
            return {k: _plain(x) for k, x in v.items()}
        if hasattr(v, "item"):  # numpy scalars
            return v.item()
        return v

    return {k: _plain(v) for k, v in asdict(cfg).items()}


def load_config(path: str | Path = "config/default_config.yaml") -> SystemConfig:
    """Loads SystemConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        return SystemConfig()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return config_from_dict(raw)


def save_config(
    cfg: SystemConfig, path: str | Path, header_comment: Optional[str] = None
) -> Path:
    """Writes a SystemConfig to YAML, optionally prefixed by a '#' comment header."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(config_to_dict(cfg), sort_keys=False, default_flow_style=None)
    header = ""
    if header_comment:
        header = "".join(f"# {line}".rstrip() + "\n" for line in header_comment.splitlines())
        header += "\n"
    out_path.write_text(header + body, encoding="utf-8")
    return out_path
