"""CLI Script to Render Unitree Go1 Quadruped Dog Validation MP4 Videos across Checkpoints and Seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.video_recorder import run_batch_video_rendering


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MP4 videos for Unitree Go1 Quadruped Dog validation scenarios across checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dog_validation",
        help="Path to training output directory containing checkpoints/ (default: outputs/dog_validation).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seed IDs to render (e.g. --seeds 42). Default: all found.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Checkpoint name filters (e.g. --checkpoints best latest). Default: all found.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum simulation steps per validation scenario clip (default: 500).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30 fps).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video files instead of skipping.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable on-screen telemetry HUD banner overlay.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling (with exploration temperature) instead of deterministic greedy mode.",
    )
    parser.add_argument(
        "--orbit-camera",
        action="store_true",
        help="Smoothly orbit camera angle around the agent during recording.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load checkpoints on (default: cpu).",
    )
    args = parser.parse_args()

    run_batch_video_rendering(
        env_type="dog",
        output_dir=args.output_dir,
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        max_steps=args.max_steps,
        fps=args.fps,
        overwrite=args.overwrite,
        overlay_hud=not args.no_hud,
        device=args.device,
        deterministic=not args.stochastic,
        orbit_camera=args.orbit_camera,
    )


if __name__ == "__main__":
    main()
