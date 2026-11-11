"""Tests for MorphologicalHumanoidBRLWrapper and evaluate_humanoid_benchmark_suite."""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_morph_env.wrapper import (
    MorphologicalHumanoidBRLWrapper,
    evaluate_humanoid_benchmark_suite,
)


class DummyAgent:
    """Mock agent for benchmark evaluation testing."""

    def __init__(self, obs_dim: int = 348, act_dim: int = 17) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = type("Config", (), {"max_action_dim": act_dim})()

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(self.act_dim, dtype=np.float32)

    def set_episode_context(self, tweak_vector: np.ndarray, action_mask: np.ndarray) -> None:
        pass


def test_humanoid_wrapper_init_and_spaces() -> None:
    env = MorphologicalHumanoidBRLWrapper(include_cfrc_ext=True)
    assert env.observation_space.shape == (348,)
    assert env.action_space.shape == (17,)
    obs, info = env.reset()
    assert obs.shape == (348,)
    assert "tweak_vector" in info
    assert "morphology_mask" in info
    assert len(info["morphology_mask"]) == 17
    env.close()


def test_humanoid_wrapper_step_and_rewards() -> None:
    env = MorphologicalHumanoidBRLWrapper(include_cfrc_ext=True)
    obs, info = env.reset()
    act = np.zeros(17, dtype=np.float32)
    next_obs, rew, term, trunc, info = env.step(act)
    assert next_obs.shape == (348,)
    assert isinstance(rew, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "brl_metrics" in info
    assert "x_velocity" in info["brl_metrics"]
    assert "total_distance" in info["brl_metrics"]
    env.close()


def test_humanoid_wrapper_tweak_perturbations() -> None:
    env = MorphologicalHumanoidBRLWrapper(include_cfrc_ext=True)
    obs, info = env.reset()
    base_torso_mass = info.get("torso_mass", 1.0)

    # Apply heavy tweak on index 0 (torso mass) and index 6 (gravity)
    heavy_tweak = np.array([1000.0, 0, 0, 0, 0, 0, 800.0, 0, 0, 0, 0, 0], dtype=np.float32)
    obs, info = env.reset(options={"tweak_vector": heavy_tweak})
    tweaked_mass = info.get("torso_mass", 1.0)
    assert tweaked_mass > base_torso_mass
    env.close()


def test_humanoid_benchmark_suite() -> None:
    env = MorphologicalHumanoidBRLWrapper(include_cfrc_ext=True)
    agent = DummyAgent(obs_dim=348, act_dim=17)
    suite = evaluate_humanoid_benchmark_suite(agent, env, max_steps=5, verbose=False)

    expected_creatures = {
        "Standard-Humanoid-v5",
        "Heavy-Torso-Humanoid",
        "Ice-Floor-Humanoid",
        "Weak-Knee-Humanoid",
        "Windy-Humanoid",
    }
    assert set(suite.keys()) == expected_creatures
    for name, metrics in suite.items():
        assert "return" in metrics
        assert "steps" in metrics
        assert metrics["steps"] <= 5
    env.close()
