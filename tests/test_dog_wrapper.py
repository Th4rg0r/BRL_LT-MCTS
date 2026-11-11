"""Tests for MorphologicalDogBRLWrapper and evaluate_dog_benchmark_suite."""

from __future__ import annotations

import numpy as np
import pytest

from dog_morph_env.wrapper import (
    MorphologicalDogBRLWrapper,
    evaluate_dog_benchmark_suite,
)


class DummyAgent:
    """Mock agent for Dog benchmark evaluation testing."""

    def __init__(self, obs_dim: int = 48, act_dim: int = 12) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = type("Config", (), {"max_action_dim": act_dim})()

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(self.act_dim, dtype=np.float32)

    def set_episode_context(self, tweak_vector: np.ndarray, action_mask: np.ndarray) -> None:
        pass


def test_dog_wrapper_init_and_spaces() -> None:
    env = MorphologicalDogBRLWrapper()
    assert env.observation_space.shape == (48,)
    assert env.action_space.shape == (12,)
    obs, info = env.reset()
    assert obs.shape == (48,)
    assert "tweak_vector" in info
    assert "morphology_mask" in info
    assert len(info["morphology_mask"]) == 12
    # All legs active by default
    assert np.all(info["morphology_mask"] == 1.0)
    env.close()


def test_dog_wrapper_step_and_rewards() -> None:
    env = MorphologicalDogBRLWrapper()
    obs, info = env.reset()
    act = np.zeros(12, dtype=np.float32)
    next_obs, rew, term, trunc, info = env.step(act)
    assert next_obs.shape == (48,)
    assert isinstance(rew, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "brl_metrics" in info
    assert "x_velocity" in info["brl_metrics"]
    assert "total_distance" in info["brl_metrics"]
    env.close()


def test_dog_wrapper_tripod_mode() -> None:
    env = MorphologicalDogBRLWrapper()
    # Disable leg 0 (Front-Right)
    obs, info = env.reset(options={"disabled_leg": 0})
    mask = info["morphology_mask"]
    assert mask[0] == 0.0
    assert mask[1] == 0.0
    assert mask[2] == 0.0
    assert np.all(mask[3:] == 1.0)

    # Disable via tweak[11] > 750
    tweak = np.zeros(12, dtype=np.float32)
    tweak[11] = 800.0
    obs, info = env.reset(options={"tweak_vector": tweak})
    mask = info["morphology_mask"]
    assert mask[0] == 0.0
    assert mask[1] == 0.0
    assert mask[2] == 0.0
    assert np.all(mask[3:] == 1.0)
    env.close()


def test_dog_wrapper_tweak_perturbations() -> None:
    env = MorphologicalDogBRLWrapper()
    obs, info = env.reset()
    base_torso_mass = info.get("torso_mass", 5.5)

    # Heavy payload tweak on index 0
    heavy_tweak = np.array([1000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    obs, info = env.reset(options={"tweak_vector": heavy_tweak})
    tweaked_mass = info.get("torso_mass", 5.5)
    assert tweaked_mass > base_torso_mass
    env.close()


def test_dog_benchmark_suite() -> None:
    env = MorphologicalDogBRLWrapper()
    agent = DummyAgent(obs_dim=48, act_dim=12)
    suite = evaluate_dog_benchmark_suite(agent, env, max_steps=5, verbose=False)

    expected_creatures = {
        "Standard-Go1",
        "Tripod-Dog",
        "Heavy-Payload-Dog",
        "Ice-Surface-Dog",
        "Windy-Gale-Dog",
    }
    assert set(suite.keys()) == expected_creatures
    for name, metrics in suite.items():
        assert "return" in metrics
        assert "steps" in metrics
        assert metrics["steps"] <= 5
    env.close()
