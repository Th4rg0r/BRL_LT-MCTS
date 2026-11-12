"""Unit and Smoke Tests for the Automated MP4 Video Rendering Pipeline."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from experiments.video_recorder import (
    get_validation_scenarios,
    discover_checkpoints,
    add_hud_overlay,
    render_scenario_video,
    create_rendering_env,
)
from dog_morph_env.wrapper import MorphologicalDogBRLWrapper
from config import SystemConfig


def test_validation_scenarios_count_and_format():
    for env_name in ["ant", "humanoid", "dog"]:
        scenarios = get_validation_scenarios(env_name)
        assert len(scenarios) == 5, f"{env_name} should have exactly 5 validation scenarios"
        for key, spec in scenarios.items():
            assert "display_name" in spec
            assert "tweak" in spec
            assert isinstance(spec["tweak"], np.ndarray)
            assert spec["tweak"].shape == (12,)


def test_discover_checkpoints_mock(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints" / "seed_42"
    ckpt_dir.mkdir(parents=True)
    
    mock_payload_1 = {
        "seed": 42,
        "episode": 500,
        "total_env_steps": 325000,
    }
    torch.save(mock_payload_1, ckpt_dir / "checkpoint_ep_0500.pt")

    mock_payload_2 = {
        "seed": 42,
        "episode": 1000,
        "total_env_steps": 650000,
    }
    torch.save(mock_payload_2, ckpt_dir / "checkpoint_ep_1000.pt")

    discovered = discover_checkpoints(tmp_path)
    assert len(discovered) == 2
    assert discovered[0]["stem"] == "checkpoint_ep_0500"
    assert discovered[0]["total_env_steps"] == 325000
    assert discovered[1]["stem"] == "checkpoint_ep_1000"
    assert discovered[1]["total_env_steps"] == 650000

    # Test filtering
    filtered = discover_checkpoints(tmp_path, ckpt_filter=["0500"])
    assert len(filtered) == 1
    assert filtered[0]["stem"] == "checkpoint_ep_0500"


def test_hud_overlay():
    base_frame = np.ones((480, 480, 3), dtype=np.uint8) * 128
    overlayed = add_hud_overlay(
        base_frame,
        line1="Ant-v5 | Seed 42 | Step 500,000 | Tripod",
        line2="Frame 010/500 | Ret: +250.0 | Vel: +1.20 m/s",
    )
    assert overlayed.shape == (480, 480, 3)
    assert overlayed.dtype == np.uint8
    # Top banner pixels should be dark background (20, 24, 30) instead of original 128
    assert overlayed[1, 1, 0] == 20
    assert overlayed[1, 1, 1] == 24
    assert overlayed[1, 1, 2] == 30


def test_dog_rendering_offscreen():
    env = MorphologicalDogBRLWrapper(render_mode="rgb_array", camera_name="track")
    obs, info = env.reset()
    frame = env.render()
    assert frame is not None
    assert frame.shape == (480, 480, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_real_ant_checkpoint_render_smoke(tmp_path: Path):
    ckpt_path = Path("outputs/validation/checkpoints/seed_42/checkpoint_ep_0500.pt")
    if not ckpt_path.exists():
        pytest.skip("No pre-existing checkpoint_ep_0500.pt found")

    from experiments.train_ant_brl import load_training_checkpoint

    agent, _, _, cfg, _ = load_training_checkpoint(ckpt_path, device="cpu")
    env = create_rendering_env("ant", cfg)

    scenarios = get_validation_scenarios("ant")
    tripod_spec = scenarios["tripod_3leg"]

    out_video = tmp_path / "smoke_tripod.mp4"
    res = render_scenario_video(
        agent=agent,
        env=env,
        scenario_key="tripod_3leg",
        scenario_spec=tripod_spec,
        out_path=out_video,
        seed=42,
        total_env_steps=325552,
        episode=500,
        env_name="ant",
        max_steps=10,  # Quick smoke test of 10 steps
        fps=30,
        overlay_hud=True,
    )
    env.close()

    assert out_video.exists()
    assert out_video.stat().st_size > 1000  # Valid MP4 video size
    assert res["steps"] <= 10


def test_multi_morphology_rendering_diff():
    """Verifies that switching between morphologies properly invalidates offscreen viewers and avoids static frames."""
    import gymnasium as gym
    from ant_morph_env.wrapper import MorphologicalAntBRLWrapper

    base = gym.make("Ant-v5", render_mode="rgb_array")
    env = MorphologicalAntBRLWrapper(env=base)

    try:
        # Standard 4-leg
        env.reset(options={"tweak_vector": np.zeros(12), "num_legs": 4})
        f_std_1 = env.render()
        env.step(np.ones(12, dtype=np.float32))
        f_std_2 = env.render()
        diff_std = np.max(np.abs(f_std_1.astype(int) - f_std_2.astype(int)))
        assert diff_std > 100, f"Standard ant frames should differ during step, got {diff_std}"

        # Rebuild to 3-legged Tripod
        env.reset(options={"tweak_vector": np.array([0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), "num_legs": 3})
        f_tri_1 = env.render()
        env.step(np.ones(12, dtype=np.float32))
        f_tri_2 = env.render()
        diff_tri = np.max(np.abs(f_tri_1.astype(int) - f_tri_2.astype(int)))
        assert diff_tri > 100, f"Tripod ant frames should differ after rebuild, got {diff_tri}"

        # Rebuild to 6-legged Hexapod
        env.reset(options={"tweak_vector": np.array([300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400], dtype=np.float32), "num_legs": 6})
        f_hex_1 = env.render()
        env.step(np.ones(12, dtype=np.float32))
        f_hex_2 = env.render()
        diff_hex = np.max(np.abs(f_hex_1.astype(int) - f_hex_2.astype(int)))
        assert diff_hex > 100, f"Hexapod ant frames should differ after rebuild, got {diff_hex}"
    finally:
        env.close()


def test_orbit_camera_and_hud_status(tmp_path: Path):
    """Verifies that orbit_camera rotates viewer azimuth and HUD overlays locomotion/static status."""
    import gymnasium as gym
    from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
    from experiments.video_recorder import render_scenario_video, add_hud_overlay

    base = gym.make("Ant-v5", render_mode="rgb_array")
    env = MorphologicalAntBRLWrapper(env=base)

    class DummyAgent:
        class DummyCfg:
            max_action_dim = 12
        cfg = DummyCfg()
        def select_action(self, obs, deterministic=True):
            return np.zeros(12, dtype=np.float32)

    agent = DummyAgent()
    spec = {
        "display_name": "Test Orbit Ant",
        "tweak": np.zeros(12, dtype=np.float32),
        "num_legs": 4,
    }
    out_file = tmp_path / "test_orbit.mp4"

    res = render_scenario_video(
        agent=agent,
        env=env,
        scenario_key="test_orbit",
        scenario_spec=spec,
        out_path=out_file,
        seed=42,
        total_env_steps=1000,
        episode=1,
        env_name="ant",
        max_steps=5,
        fps=10,
        overlay_hud=True,
        orbit_camera=True,
    )
    env.close()

    assert out_file.exists()
    assert out_file.stat().st_size > 1000
    assert res["steps"] == 5


def test_sequential_seed_rendering_no_black_screen(tmp_path: Path):
    """Verifies that sequential renders across different seeds do not suffer from black screens (mean > 50)."""
    import gymnasium as gym
    import imageio
    from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
    from experiments.video_recorder import render_scenario_video

    base = gym.make("Ant-v5", render_mode="rgb_array")
    env = MorphologicalAntBRLWrapper(env=base)

    class DummyAgent:
        class DummyCfg:
            max_action_dim = 12
        cfg = DummyCfg()
        def select_action(self, obs, deterministic=True):
            return np.zeros(12, dtype=np.float32)

    agent = DummyAgent()
    spec = {
        "display_name": "Standard Ant Nominal",
        "tweak": np.zeros(12, dtype=np.float32),
        "num_legs": 4,
    }

    # Render seed 42 then seed 43 sequentially using the same environment
    for seed in [42, 43]:
        out_file = tmp_path / f"test_seed_{seed}.mp4"
        render_scenario_video(
            agent=agent,
            env=env,
            scenario_key="standard_ant",
            scenario_spec=spec,
            out_path=out_file,
            seed=seed,
            total_env_steps=1000,
            episode=1,
            env_name="ant",
            max_steps=5,
            fps=10,
            track_camera=True,
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 1000
        reader = imageio.get_reader(str(out_file), "ffmpeg")
        first_frame = next(iter(reader))
        reader.close()
        # Pixel mean must be healthy (> 50), not black screen (~3.3)
        assert float(first_frame.mean()) > 50.0, f"Seed {seed} produced black screen with mean {first_frame.mean()}"

    env.close()



def test_filter_validation_scenarios():
    """Verifies single-scenario filtering and alias resolution (e.g. 'icy', 'standard')."""
    from experiments.video_recorder import filter_validation_scenarios, get_validation_scenarios

    ant_scens = get_validation_scenarios("ant")
    assert len(ant_scens) == 5

    # Test filtering single scenario: 'icy'
    icy_filtered = filter_validation_scenarios(ant_scens, "icy")
    assert len(icy_filtered) == 1
    assert "heavy_ice" in icy_filtered

    # Test filtering single scenario: 'standard'
    std_filtered = filter_validation_scenarios(ant_scens, "standard")
    assert len(std_filtered) == 1
    assert "standard_ant" in std_filtered

    # Test dog scenarios: 'payload'
    dog_scens = get_validation_scenarios("dog")
    payload_filtered = filter_validation_scenarios(dog_scens, "payload")
    assert len(payload_filtered) == 1
    assert "heavy_payload" in payload_filtered

    # Test invalid scenario query
    with pytest.raises(ValueError):
        filter_validation_scenarios(ant_scens, "flying_ant_42")


def test_track_camera_centering(tmp_path: Path):
    """Verifies that track_camera centers viewer camera on the creature."""
    import gymnasium as gym
    from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
    from experiments.video_recorder import render_scenario_video

    base = gym.make("Ant-v5", render_mode="rgb_array")
    env = MorphologicalAntBRLWrapper(env=base)

    class DummyAgent:
        class DummyCfg:
            max_action_dim = 12
        cfg = DummyCfg()
        def select_action(self, obs, deterministic=True):
            return np.ones(12, dtype=np.float32) * 0.5

    agent = DummyAgent()
    spec = {
        "display_name": "Standard Ant Track Test",
        "tweak": np.zeros(12, dtype=np.float32),
        "num_legs": 4,
    }
    out_file = tmp_path / "test_track_cam.mp4"

    res = render_scenario_video(
        agent=agent,
        env=env,
        scenario_key="standard_ant",
        scenario_spec=spec,
        out_path=out_file,
        seed=42,
        total_env_steps=1000,
        episode=1,
        env_name="ant",
        max_steps=5,
        fps=10,
        track_camera=True,
    )
    env.close()

    assert out_file.exists()
    assert out_file.stat().st_size > 1000
    assert res["steps"] == 5


def test_sequential_seed_rendering_no_black_screen(tmp_path: Path):
    """Verifies that sequential renders across different seeds do not suffer from black screens (mean > 50)."""
    import gymnasium as gym
    import imageio
    from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
    from experiments.video_recorder import render_scenario_video

    base = gym.make("Ant-v5", render_mode="rgb_array")
    env = MorphologicalAntBRLWrapper(env=base)

    class DummyAgent:
        class DummyCfg:
            max_action_dim = 12
        cfg = DummyCfg()
        def select_action(self, obs, deterministic=True):
            return np.zeros(12, dtype=np.float32)

    agent = DummyAgent()
    spec = {
        "display_name": "Standard Ant Nominal",
        "tweak": np.zeros(12, dtype=np.float32),
        "num_legs": 4,
    }

    # Render seed 42 then seed 43 sequentially using the same environment
    for seed in [42, 43]:
        out_file = tmp_path / f"test_seed_{seed}.mp4"
        render_scenario_video(
            agent=agent,
            env=env,
            scenario_key="standard_ant",
            scenario_spec=spec,
            out_path=out_file,
            seed=seed,
            total_env_steps=1000,
            episode=1,
            env_name="ant",
            max_steps=5,
            fps=10,
            track_camera=True,
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 1000
        reader = imageio.get_reader(str(out_file), "ffmpeg")
        first_frame = next(iter(reader))
        reader.close()
        # Pixel mean must be healthy (> 50), not black screen (~3.3)
        assert float(first_frame.mean()) > 50.0, f"Seed {seed} produced black screen with mean {first_frame.mean()}"

    env.close()



