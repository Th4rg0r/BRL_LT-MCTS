"""Automated MP4 Video Rendering Engine for BRL + LA-MCTS Validation Scenarios.

Scans training checkpoints across seeds and models, evaluates the agent on the
canonical 5 validation scenarios per creature (Standard, 3-Leg/Amputation, High Mass,
Low Friction/Ice, Stress/Morphology), and writes high-definition MP4 videos to
`{output_dir}/videos/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import torch
import imageio

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper
from humanoid_morph_env.wrapper import MorphologicalHumanoidBRLWrapper
from dog_morph_env.wrapper import MorphologicalDogBRLWrapper
from config import SystemConfig


# ==============================================================================
# Scenario Specifications (5 Scenarios per Creature)
# ==============================================================================

def get_validation_scenarios(env_type: str) -> Dict[str, Dict[str, Any]]:
    """Returns canonical 5 validation scenarios for the requested creature."""
    env_type = env_type.lower().strip()
    if env_type == "ant":
        return {
            "standard_ant": {
                "display_name": "Standard Ant (Nominal)",
                "tweak": np.zeros(12, dtype=np.float32),
                "num_legs": 4,
            },
            "tripod_3leg": {
                "display_name": "Tripod Ant (3 Legs)",
                "tweak": np.array([0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "num_legs": 3,
            },
            "spider_longleg": {
                "display_name": "Spider Ant (Long Legs)",
                "tweak": np.array([800, 0, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "num_legs": 4,
            },
            "heavy_ice": {
                "display_name": "Heavy Ice Ant (Friction 0.3x, Mass 1.6x)",
                "tweak": np.array([0, 0, 0, 0, 600, 700, 0, 0, 0, 300, 0, 0], dtype=np.float32),
                "num_legs": 4,
            },
            "hexapod_6leg": {
                "display_name": "Hexapod Ant (6 Legs)",
                "tweak": np.array([300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400], dtype=np.float32),
                "num_legs": 6,
            },
        }

    if env_type == "humanoid":
        return {
            "standard_humanoid": {
                "display_name": "Standard Humanoid (Nominal)",
                "tweak": np.zeros(12, dtype=np.float32),
            },
            "weak_knee": {
                "display_name": "Weak Knee Humanoid (Low Damping)",
                "tweak": np.array([0, 0, 0, 0, 800, 0, 0, 600, 0, 0, 0, 0], dtype=np.float32),
            },
            "heavy_torso": {
                "display_name": "Heavy Torso Humanoid (+80% Mass)",
                "tweak": np.array([800, 0, 0, 0, 0, 0, 300, 0, 0, 0, 0, 0], dtype=np.float32),
            },
            "ice_floor": {
                "display_name": "Ice Floor Humanoid (Slippery)",
                "tweak": np.array([0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            },
            "windy_gale": {
                "display_name": "Windy Humanoid (Turbulent Wind)",
                "tweak": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 800, 0, 0], dtype=np.float32),
            },
        }

    if env_type in ("dog", "unitree_go1", "go1"):
        return {
            "standard_dog": {
                "display_name": "Standard Dog Go1 (Nominal)",
                "tweak": np.zeros(12, dtype=np.float32),
                "disabled_leg": None,
            },
            "tripod_3leg": {
                "display_name": "Tripod Dog (FR Leg Disabled)",
                "tweak": np.zeros(12, dtype=np.float32),
                "disabled_leg": 0,
            },
            "heavy_payload": {
                "display_name": "Heavy Payload Dog (+80% Chassis)",
                "tweak": np.array([800, 0, 0, 0, 0, 0, 300, 0, 0, 0, 0, 0], dtype=np.float32),
                "disabled_leg": None,
            },
            "ice_surface": {
                "display_name": "Ice Surface Dog (Low Friction)",
                "tweak": np.array([0, 0, 0, 0, 0, 800, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "disabled_leg": None,
            },
            "windy_gale": {
                "display_name": "Windy Gale Dog (Wind Gusts)",
                "tweak": np.array([0, 0, 0, 0, 0, 0, 0, 0, 700, 600, 0, 0], dtype=np.float32),
                "disabled_leg": None,
            },
        }

    raise ValueError(f"Unknown env_type '{env_type}'. Expected 'ant', 'humanoid', or 'dog'.")


# ==============================================================================
# Environment Creation Helper
# ==============================================================================

def create_rendering_env(env_type: str, cfg: SystemConfig) -> gym.Env:
    """Instantiates the appropriate creature environment in rgb_array render mode."""
    env_type = env_type.lower().strip()
    if env_type == "ant":
        ant_cfg = getattr(cfg, "ant_morph", None)
        env_id = getattr(ant_cfg, "env_id", "Ant-v5")
        use_cfrc = getattr(ant_cfg, "use_contact_forces", False)
        healthy_z = getattr(ant_cfg, "healthy_z_range", (0.2, 1.0))
        max_act = getattr(ant_cfg, "max_action_capacity", 12)
        if str(env_id).startswith("Ant-v5"):
            base_env = gym.make(
                env_id,
                render_mode="rgb_array",
                include_cfrc_ext_in_observation=use_cfrc,
                contact_cost_weight=(5e-4 if use_cfrc else 0.0),
                healthy_z_range=healthy_z,
            )
        else:
            base_env = gym.make(
                env_id,
                render_mode="rgb_array",
                use_contact_forces=use_cfrc,
                healthy_z_range=healthy_z,
            )
        return MorphologicalAntBRLWrapper(
            env=base_env,
            use_contact_forces=use_cfrc,
            max_action_capacity=max_act,
            healthy_z_range=healthy_z,
        )

    if env_type == "humanoid":
        hum_cfg = getattr(cfg, "humanoid_morph", None)
        healthy_z = getattr(hum_cfg, "healthy_z_range", (1.0, 2.0))
        max_act = getattr(hum_cfg, "max_action_capacity", 17)
        base_env = gym.make(
            "Humanoid-v5",
            render_mode="rgb_array",
            healthy_z_range=healthy_z,
        )
        return MorphologicalHumanoidBRLWrapper(
            env=base_env,
            max_action_capacity=max_act,
            healthy_z_range=healthy_z,
        )

    if env_type in ("dog", "unitree_go1", "go1"):
        dog_cfg = getattr(cfg, "dog_morph", None)
        healthy_z = getattr(dog_cfg, "healthy_z_range", (0.18, 0.60))
        return MorphologicalDogBRLWrapper(
            render_mode="rgb_array",
            camera_name="track",
            healthy_z_range=healthy_z,
        )

    raise ValueError(f"Unknown env_type '{env_type}'.")


# ==============================================================================
# Checkpoint Discovery
# ==============================================================================

def discover_checkpoints(
    output_dir: str | Path,
    seeds: Optional[List[int]] = None,
    ckpt_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Discovers all saved PyTorch checkpoints across seeds in output_dir.

    Returns a list of dicts with:
      - 'path': Path
      - 'stem': str (e.g. 'checkpoint_ep_0500')
      - 'seed': int
      - 'episode': int
      - 'total_env_steps': int
    """
    out_path = Path(output_dir)
    ckpt_root = out_path / "checkpoints" if (out_path / "checkpoints").exists() else out_path

    candidates: List[Path] = []
    # 1. Search seed subdirectories (checkpoints/seed_<N>/*.pt)
    for p in ckpt_root.glob("seed_*/*.pt"):
        if not p.name.endswith(".tmp"):
            candidates.append(p)

    # 2. Search root directory (best_overall_model.pt, final_model.pt)
    for p in out_path.glob("*.pt"):
        if not p.name.endswith(".tmp") and p.is_file():
            candidates.append(p)

    discovered: List[Dict[str, Any]] = []
    for cp in candidates:
        stem = cp.stem
        # Check filter if provided
        if ckpt_filter:
            matches_filter = any(f.lower() in stem.lower() for f in ckpt_filter)
            if not matches_filter:
                continue

        # Peek inside checkpoint header without loading all weights
        try:
            payload = torch.load(cp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Warning: could not load checkpoint {cp}: {e}")
            continue

        seed_val = payload.get("seed")
        if seed_val is None:
            # Try to infer from parent dir or stem
            m_seed = re.search(r"seed_?(\d+)", str(cp))
            seed_val = int(m_seed.group(1)) if m_seed else 0
        else:
            seed_val = int(seed_val)

        if seeds is not None and seed_val not in seeds:
            continue

        ep_val = int(payload.get("episode", 0))
        steps_val = int(payload.get("total_env_steps", 0))

        discovered.append({
            "path": cp,
            "stem": stem,
            "seed": seed_val,
            "episode": ep_val,
            "total_env_steps": steps_val,
        })

    # Sort deterministically by seed, then steps, then episode, then stem
    discovered.sort(key=lambda d: (d["seed"], d["total_env_steps"], d["episode"], d["stem"]))
    return discovered


# ==============================================================================
# HUD Overlay
# ==============================================================================

def add_hud_overlay(
    frame: np.ndarray,
    line1: str,
    line2: str,
) -> np.ndarray:
    """Draws a clean, high-contrast telemetry HUD banner at the top of the video frame."""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # Draw semi-translucent dark banner at top
    banner_height = 42
    draw.rectangle([(0, 0), (w, banner_height)], fill=(20, 24, 30))

    # Text lines
    draw.text((12, 4), line1, fill=(255, 255, 255))
    draw.text((12, 22), line2, fill=(130, 210, 255))

    return np.array(pil_img)


# ==============================================================================
# Video Rendering for Single Scenario
# ==============================================================================

def render_scenario_video(
    agent: Any,
    env: gym.Env,
    scenario_key: str,
    scenario_spec: Dict[str, Any],
    out_path: Path,
    seed: int,
    total_env_steps: int,
    episode: int,
    env_name: str,
    max_steps: int = 500,
    fps: int = 30,
    overlay_hud: bool = True,
    deterministic: bool = True,
    orbit_camera: bool = False,
) -> Dict[str, Any]:
    """Runs an evaluation episode and saves it as an MP4 video with progressive faststart."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Invalidate any stale viewer cache before setting new morphology/scenario
    if hasattr(env.unwrapped, "mujoco_renderer") and env.unwrapped.mujoco_renderer is not None:
        r = env.unwrapped.mujoco_renderer
        if hasattr(r, "_viewers"):
            for v in r._viewers.values():
                if hasattr(v, "close"):
                    try:
                        v.close()
                    except Exception:
                        pass
            r._viewers.clear()
            r.viewer = None

    # Configure reset options
    reset_opts: Dict[str, Any] = {"tweak_vector": scenario_spec["tweak"]}
    if "num_legs" in scenario_spec:
        reset_opts["num_legs"] = scenario_spec["num_legs"]
    if "disabled_leg" in scenario_spec:
        reset_opts["disabled_leg"] = scenario_spec["disabled_leg"]

    obs, info = env.reset(options=reset_opts)
    morph_mask = info.get(
        "morphology_mask",
        np.ones(getattr(agent.cfg, "max_action_dim", 12), dtype=np.float32),
    )
    if hasattr(agent, "set_episode_context"):
        agent.set_episode_context(
            tweak_vector=info.get("perceived_tweak_vector", scenario_spec["tweak"]),
            action_mask=morph_mask,
        )

    disp_name = scenario_spec.get("display_name", scenario_key)

    # Initialize video writer with faststart for progressive playback in Windows Media Player & web
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_params=["-movflags", "+faststart"],
    )

    ep_return = 0.0
    steps = 0
    start_time = time.time()
    last_info = info

    try:
        while steps < max_steps:
            if orbit_camera:
                try:
                    if hasattr(env.unwrapped, "mujoco_renderer") and env.unwrapped.mujoco_renderer is not None:
                        v = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
                        if hasattr(v, "cam") and hasattr(v.cam, "azimuth"):
                            v.cam.azimuth = (v.cam.azimuth + 0.6) % 360.0
                except Exception:
                    pass

            raw_frame = env.render()
            if raw_frame is not None:
                if overlay_hud:
                    brl_m = last_info.get("brl_metrics", {})
                    x_vel = float(brl_m.get("x_velocity", 0.0))
                    tot_dist = float(brl_m.get("total_distance", 0.0))
                    mode_str = "DET" if deterministic else "STOCH"
                    loco_str = "[LOCOMOTION]" if abs(x_vel) >= 0.05 else "[STATIC HOLD]"
                    l1 = f"{env_name.upper()} [{mode_str}] {loco_str} | Seed {seed} | Step {total_env_steps:,} (Ep {episode}) | {disp_name}"
                    l2 = f"Frame {steps:03d}/{max_steps} | Ret: {ep_return:+6.1f} | Dist: {tot_dist:+5.2f}m | Vel: {x_vel:+5.2f}m/s"
                    frame = add_hud_overlay(raw_frame, l1, l2)
                else:
                    frame = raw_frame
                writer.append_data(frame)

            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_return += float(reward)
            steps += 1

            if terminated or truncated:
                # Capture final frame and hold for 15 frames (~0.5s) so the clip does not abruptly end
                raw_frame = env.render()
                if raw_frame is not None:
                    if overlay_hud:
                        brl_m = last_info.get("brl_metrics", {})
                        final_x_vel = float(brl_m.get("x_velocity", 0.0))
                        tot_dist = float(brl_m.get("total_distance", 0.0))
                        if terminated:
                            term_tag = "[FALL / TERMINATED]"
                        elif abs(final_x_vel) < 0.05:
                            term_tag = "[TRUNCATED - STATIC]"
                        else:
                            term_tag = "[TRUNCATED - MOVING]"
                        l1 = f"{env_name.upper()} | Seed {seed} | Step {total_env_steps:,} | {disp_name} {term_tag}"
                        l2 = f"Total Steps: {steps:03d}/{max_steps} | Final Return: {ep_return:+6.1f} | Dist: {tot_dist:+5.2f}m"
                        term_frame = add_hud_overlay(raw_frame, l1, l2)
                    else:
                        term_frame = raw_frame
                    for _ in range(15):
                        writer.append_data(term_frame)
                break
    finally:
        writer.close()

    wall_sec = time.time() - start_time
    brl_m = last_info.get("brl_metrics", {})
    return {
        "video_path": str(out_path),
        "scenario": scenario_key,
        "display_name": disp_name,
        "steps": int(steps),
        "return": float(ep_return),
        "total_distance": float(brl_m.get("total_distance", 0.0)),
        "x_velocity": float(brl_m.get("x_velocity", 0.0)),
        "wall_sec": float(wall_sec),
    }


# ==============================================================================
# Batch Rendering Pipeline
# ==============================================================================

def run_batch_video_rendering(
    env_type: str,
    output_dir: str | Path,
    seeds: Optional[List[int]] = None,
    checkpoints: Optional[List[str]] = None,
    max_steps: int = 500,
    fps: int = 30,
    overwrite: bool = False,
    overlay_hud: bool = True,
    device: str = "cpu",
    deterministic: bool = True,
    orbit_camera: bool = False,
) -> Dict[str, Any]:
    """Runs video generation for all checkpoints and validation scenarios."""
    env_type = env_type.lower().strip()
    out_dir = Path(output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "Deterministic" if deterministic else "Stochastic"
    cam_str = "Orbiting (Dynamic Angle)" if orbit_camera else "Fixed (Static Angle)"
    print("=" * 80)
    print(f"  BRL Video Rendering Suite - [{env_type.upper()}]")
    print(f"  Target Directory  : {out_dir}")
    print(f"  Videos Output     : {videos_dir}")
    print(f"  Max Steps / Clip  : {max_steps} (approx {max_steps / fps:.1f}s @ {fps} fps)")
    print(f"  Action Mode       : {mode_str}")
    print(f"  Camera Mode       : {cam_str}")
    print(f"  Device            : {device}")
    print("=" * 80)

    # 1. Discover checkpoints
    discovered = discover_checkpoints(out_dir, seeds=seeds, ckpt_filter=checkpoints)
    if not discovered:
        print(f"No matching checkpoints found in {out_dir}.")
        return {"rendered_videos": [], "count": 0}

    print(f"Discovered {len(discovered)} checkpoint(s) to render:")
    for idx, ck in enumerate(discovered, 1):
        print(f"  [{idx:2d}] Seed {ck['seed']:2d} | Step {ck['total_env_steps']:8,d} | Ep {ck['episode']:4d} | {ck['path'].name}")
    print("-" * 80)

    # 2. Get scenario specs
    scenarios = get_validation_scenarios(env_type)
    print(f"Validation Scenarios ({len(scenarios)}): {', '.join(scenarios.keys())}")
    print("-" * 80)

    # 3. Import appropriate checkpoint loader
    if env_type == "ant":
        from experiments.train_ant_brl import load_training_checkpoint
    elif env_type == "humanoid":
        from experiments.train_humanoid_brl import load_training_checkpoint
    elif env_type in ("dog", "unitree_go1", "go1"):
        from experiments.train_dog_brl import load_training_checkpoint
    else:
        raise ValueError(f"Unknown env_type '{env_type}'.")

    # 4. Instantiate environment once
    # Load first checkpoint to get configuration
    first_ckpt = discovered[0]
    _, _, _, cfg, _ = load_training_checkpoint(first_ckpt["path"], device=device)
    env = create_rendering_env(env_type, cfg)

    rendered_results: List[Dict[str, Any]] = []
    total_videos = len(discovered) * len(scenarios)
    video_idx = 0

    try:
        for ck in discovered:
            ckpt_path = ck["path"]
            seed = ck["seed"]
            steps = ck["total_env_steps"]
            ep = ck["episode"]
            stem = ck["stem"]

            # Load checkpoint agent
            print(f"\n>> Loading Checkpoint: Seed {seed} | Steps {steps:,} | {stem}")
            agent, _, _, _, _ = load_training_checkpoint(ckpt_path, device=device)

            for scen_key, scen_spec in scenarios.items():
                video_idx += 1
                # Format standardized filename: seed_<ID>_step_<STEPS>_<CKPT>_<SCENARIO>.mp4
                filename = f"seed_{seed:02d}_step_{steps:08d}_{stem}_{scen_key}.mp4"
                video_path = videos_dir / filename

                if video_path.exists() and not overwrite:
                    print(f"  [{video_idx:3d}/{total_videos:3d}] [SKIP] {filename} (Already exists)")
                    rendered_results.append({
                        "video_path": str(video_path),
                        "status": "skipped",
                        "scenario": scen_key,
                        "seed": seed,
                        "total_env_steps": steps,
                    })
                    continue

                print(f"  [{video_idx:3d}/{total_videos:3d}] Rendering {scen_spec.get('display_name', scen_key)} ...", end=" ", flush=True)
                res = render_scenario_video(
                    agent=agent,
                    env=env,
                    scenario_key=scen_key,
                    scenario_spec=scen_spec,
                    out_path=video_path,
                    seed=seed,
                    total_env_steps=steps,
                    episode=ep,
                    env_name=env_type,
                    max_steps=max_steps,
                    fps=fps,
                    overlay_hud=overlay_hud,
                    deterministic=deterministic,
                    orbit_camera=orbit_camera,
                )
                print(f"Done! ({res['steps']} steps, Ret = {res['return']:+6.1f}, {res['wall_sec']:.1f}s)")
                res.update({
                    "status": "rendered",
                    "seed": seed,
                    "total_env_steps": steps,
                    "episode": ep,
                    "checkpoint_name": stem,
                })
                rendered_results.append(res)
    finally:
        env.close()

    # Save manifest summary
    manifest_path = videos_dir / "manifest.json"
    manifest_data = {
        "env_type": env_type,
        "output_dir": str(out_dir),
        "total_rendered": sum(1 for r in rendered_results if r.get("status") == "rendered"),
        "total_skipped": sum(1 for r in rendered_results if r.get("status") == "skipped"),
        "videos": rendered_results,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Rendering Complete!")
    print(f"  Total Videos Processed : {len(rendered_results)}")
    print(f"  Rendered New           : {manifest_data['total_rendered']}")
    print(f"  Skipped (Existing)     : {manifest_data['total_skipped']}")
    print(f"  Videos Directory       : {videos_dir}")
    print(f"  Manifest File          : {manifest_path}")
    print("=" * 80)
    return manifest_data
