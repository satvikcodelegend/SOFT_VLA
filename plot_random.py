#!/usr/bin/env python3
"""
Plotter for main_superellipse_unseen_STANDALONE.py

This file:
    - finds the newest standalone superellipse run
    - preserves chronological order
    - uses only the dedicated superellipse CSV
    - never sorts by X/Y
    - does not inspect or use any figure-8 run
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DESKTOP_ROOT = Path.home() / "Desktop" / "soft_robot_runs"


def fail(message):
    raise RuntimeError(message)


def find_run():
    matches = [
        p
        for p in DESKTOP_ROOT.rglob(
            "superellipse_xy_tracking.csv"
        )
        if p.is_file()
        and p.parent.name.startswith(
            "superellipse_unseen_standalone_"
        )
    ]

    if not matches:
        fail(
            "No standalone superellipse run found under:\n"
            f"{DESKTOP_ROOT}\n\n"
            "Run main_superellipse_unseen_STANDALONE.py first."
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    return matches[0].parent


def load_csv(path):
    if not path.is_file():
        fail(f"CSV does not exist:\n{path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        fail(
            "Fewer than two superellipse samples were recorded."
        )

    return rows


def arr(rows, key):
    try:
        return np.asarray(
            [float(row[key]) for row in rows],
            dtype=np.float64,
        )
    except (KeyError, ValueError, TypeError) as exc:
        fail(f"Could not read numeric column '{key}': {exc}")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved:", path)


def load_summary(run_dir):
    path = run_dir / "summary.json"
    if not path.is_file():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main():
    print()
    print("=" * 82)
    print("STANDALONE UNSEEN SUPERELLIPSE RESULT ANALYSIS")
    print("=" * 82)

    run_dir = find_run()
    csv_path = run_dir / "superellipse_xy_tracking.csv"

    print("Run :", run_dir)
    print("CSV :", csv_path)

    rows = load_csv(csv_path)

    required = [
        "time_s",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "actual_x_m",
        "actual_y_m",
        "actual_z_m",
        "error_mm",
        "accuracy_percent",
    ]

    for key in required:
        if key not in rows[0]:
            fail(f"Missing required column: {key}")

    # Preserve the recorded chronological order.
    # The main file already writes this CSV in simulation-time order.
    t = arr(rows, "time_s")
    target_x = arr(rows, "target_x_m") * 1000.0
    target_y = arr(rows, "target_y_m") * 1000.0
    target_z = arr(rows, "target_z_m") * 1000.0

    actual_x = arr(rows, "actual_x_m") * 1000.0
    actual_y = arr(rows, "actual_y_m") * 1000.0
    actual_z = arr(rows, "actual_z_m") * 1000.0

    error = arr(rows, "error_mm")
    accuracy = arr(rows, "accuracy_percent")

    target = np.column_stack([target_x, target_y])
    actual = np.column_stack([actual_x, actual_y])

    # Local Frenet diagnostics reconstructed from the target path.
    dx = np.gradient(target_x)
    dy = np.gradient(target_y)

    speed = np.hypot(dx, dy)
    speed = np.maximum(speed, 1e-12)

    tx = dx / speed
    ty = dy / speed

    nx = -ty
    ny = tx

    xy_error_x = target_x - actual_x
    xy_error_y = target_y - actual_y

    normal_error = (
        xy_error_x * nx
        + xy_error_y * ny
    )

    tangential_error = (
        xy_error_x * tx
        + xy_error_y * ty
    )

    z_error = target_z - actual_z

    summary = load_summary(run_dir)

    # ------------------------------------------------------------------
    # 1. XY TRAJECTORY
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.plot(
        target_x,
        target_y,
        linestyle="--",
        linewidth=2.0,
        label="Target superellipse",
    )

    ax.plot(
        actual_x,
        actual_y,
        linewidth=2.2,
        label="Actual EE",
    )

    ax.scatter(
        actual_x[0],
        actual_y[0],
        s=80,
        label="Actual start",
        zorder=5,
    )

    ax.scatter(
        actual_x[-1],
        actual_y[-1],
        s=80,
        label="Actual end",
        zorder=5,
    )

    ax.set_title(
        "XY Unseen Superellipse Trajectory Tracking"
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_xy_trajectory.png",
    )

    # ------------------------------------------------------------------
    # 2. POSITION ERROR
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        error,
        linewidth=1.8,
        label="Position error",
    )

    ax.set_title("Unseen Superellipse Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_error.png",
    )

    # ------------------------------------------------------------------
    # 3. ACCURACY
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        accuracy,
        linewidth=1.8,
        label="Accuracy",
    )

    ax.axhline(
        92.0,
        linestyle="--",
        linewidth=1.4,
        label="92% benchmark",
    )

    ax.set_title("Unseen Superellipse Tracking Accuracy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_accuracy.png",
    )

    # ------------------------------------------------------------------
    # 4. NORMAL / CROSS-TRACK ERROR
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        normal_error,
        linewidth=1.8,
        label="Normal / cross-track error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title("Superellipse Cross-Track Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_cross_track_error.png",
    )

    # ------------------------------------------------------------------
    # 5. TANGENTIAL ERROR
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        tangential_error,
        linewidth=1.8,
        label="Tangential / phase error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title("Superellipse Tangential Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tangential error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_tangential_error.png",
    )

    # ------------------------------------------------------------------
    # 6. Z ERROR
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        z_error,
        linewidth=1.8,
        label="Z error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title("Superellipse Z Tracking")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "superellipse_z_error.png",
    )

    # ------------------------------------------------------------------
    # 7. SUMMARY TEXT
    # ------------------------------------------------------------------

    curve_error = error
    curve_accuracy = accuracy

    report = run_dir / "superellipse_analysis.txt"

    with report.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "STANDALONE UNSEEN SUPERELLIPSE ANALYSIS\n"
            "========================================\n\n"
        )
        f.write("Figure-8 dependency: NO\n")
        f.write("Figure-8 data used: NO\n\n")

        f.write(
            f"Samples: {len(curve_error)}\n"
        )
        f.write(
            f"Mean error: {np.mean(curve_error):.4f} mm\n"
        )
        f.write(
            f"RMSE: {np.sqrt(np.mean(curve_error ** 2)):.4f} mm\n"
        )
        f.write(
            f"P95 error: {np.percentile(curve_error, 95):.4f} mm\n"
        )
        f.write(
            f"Peak error: {np.max(curve_error):.4f} mm\n"
        )
        f.write(
            f"Mean accuracy: {np.mean(curve_accuracy):.4f}%\n"
        )
        f.write(
            f"Samples >= 92%: "
            f"{100.0 * np.mean(curve_accuracy >= 92.0):.2f}%\n"
        )
        f.write(
            f"Mean cross-track error: "
            f"{np.mean(np.abs(normal_error)):.4f} mm\n"
        )
        f.write(
            f"P95 cross-track error: "
            f"{np.percentile(np.abs(normal_error), 95):.4f} mm\n"
        )
        f.write(
            f"Mean tangential error: "
            f"{np.mean(np.abs(tangential_error)):.4f} mm\n"
        )
        f.write(
            f"P95 tangential error: "
            f"{np.percentile(np.abs(tangential_error), 95):.4f} mm\n"
        )
        f.write(
            f"Mean |Z error|: "
            f"{np.mean(np.abs(z_error)):.4f} mm\n"
        )
        f.write(
            f"P95 |Z error|: "
            f"{np.percentile(np.abs(z_error), 95):.4f} mm\n"
        )

        if summary:
            f.write("\nController summary from main run:\n")
            controller = summary.get("controller", {})
            for key, value in controller.items():
                f.write(f"  {key}: {value}\n")

    print()
    print("Analysis report:", report)
    print()
    print("Mean error     :", f"{np.mean(curve_error):.3f} mm")
    print("Mean accuracy  :", f"{np.mean(curve_accuracy):.2f}%")
    print("P95 error      :", f"{np.percentile(curve_error, 95):.3f} mm")
    print()


if __name__ == "__main__":
    main()
