"""Unit tests for progress & validation plotting script."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from experiments.plot_progress import (
    compute_rolling_mean,
    export_stats_json,
    parse_accompanying_log,
    parse_progress_jsonl,
    plot_single_progress,
)


def test_compute_rolling_mean() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    roll = compute_rolling_mean(arr, window=3)
    assert len(roll) == 5
    assert roll[0] == 1.0
    assert roll[1] == 1.5
    assert roll[2] == 2.0
    assert roll[3] == 3.0
    assert roll[4] == 4.0


def test_parse_and_plot_progress(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "test_seed_10_progress.jsonl"
    log_path = tmp_path / "test_seed_10.log"

    # Write synthetic JSONL records
    records = []
    for ep in range(1, 15):
        records.append({
            "seed": 10,
            "episode": ep,
            "steps": 1000,
            "total_env_steps": ep * 1000,
            "return": -100.0 + ep * 10.0,
            "total_distance": float(ep) * 0.5,
            "x_velocity": 0.05,
            "ema_win_rate": 0.5,
            "win": False,
            "r_ref": 100.0,
            "budget": 50.0,
            "loss_total": 10.0 / ep,
            "ep_duration": 15.0,
        })
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Write synthetic evaluation lines in log
    log_content = (
        "[2026-11-12 00:00:00] >>> [Seed 10 | Eval @ Ep   10 (10,000 steps)] Score: +50.0\n"
        "    * Standard-Ant-v5 : Return =  +50.00 | Steps = 1000/1000 | Dist =  +2.50m | Vel = +0.025 m/s\n"
        "    * Tripod-Ant      : Return =  +30.00 | Steps = 1000/1000 | Dist =  +1.20m | Vel = +0.012 m/s\n"
    )
    with log_path.open("w", encoding="utf-8") as f:
        f.write(log_content)

    data = parse_progress_jsonl(jsonl_path)
    assert data["seed"] == 10
    assert len(data["episodes"]) == 14
    assert data["has_direct_dist"] is True
    assert 10 in data["eval_by_ep"]
    assert "Standard-Ant-v5" in data["eval_by_ep"][10]

    out_img = tmp_path / "test_chart.png"
    plot_single_progress(data, out_img)
    assert out_img.exists()
    assert out_img.stat().st_size > 50000

    out_json = tmp_path / "test_stats.json"
    export_stats_json(data, out_json)
    assert out_json.exists()
