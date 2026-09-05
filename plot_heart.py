#!/usr/bin/env python3
"""
SOFT ROBOT - HEART RUN PLOTTER

Usage:
    python plot_heart.py

or:

    python plot_heart.py ~/Desktop/soft_robot_runs/heart_vla_mlp_YYYYMMDD_HHMMSS

The newest heart run is selected automatically when no path is supplied.

The dedicated heart_xy_tracking.csv contains only the heart phase and is
plotted in chronological/sample order.  X/Y sorting is never used.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DESKTOP_ROOT = (
    Path.home()
    / "Desktop"
    / "soft_robot_runs"
)


def find_run():
    if len(sys.argv) > 1:
        path = (
            Path(sys.argv[1])
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Specified path does not exist:\n{path}"
            )

        if path.is_file():
            if path.name.lower() != "trajectory.csv":
                raise FileNotFoundError(
                    "Specified file must be trajectory.csv:\n"
                    f"{path}"
                )
            return path.parent

        direct = path / "trajectory.csv"
        if direct.is_file():
            return path

        matches = [
            p
            for p in path.rglob("trajectory.csv")
            if p.is_file()
        ]

        if not matches:
            raise FileNotFoundError(
                f"No trajectory.csv found inside:\n{path}"
            )

        matches.sort(
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )

        return matches[0].parent

    if not DESKTOP_ROOT.exists():
        raise FileNotFoundError(
            f"Desktop result directory not found:\n{DESKTOP_ROOT}"
        )

    matches = [
        p
        for p in DESKTOP_ROOT.rglob("trajectory.csv")
        if p.is_file()
        and p.parent.name.lower().startswith("heart_")
    ]

    if not matches:
        raise FileNotFoundError(
            "No heart run containing trajectory.csv was found under:\n"
            f"{DESKTOP_ROOT}"
        )

    matches.sort(
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    return matches[0].parent


def require_columns(df, columns):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "trajectory.csv is missing required columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )


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

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    print()
    print("=" * 78)
    print("TEMPORAL HEART RESULT ANALYSIS")
    print("=" * 78)

    run_dir = find_run()
    csv_path = run_dir / "trajectory.csv"

    print()
    print("Run:")
    print(run_dir)
    print()
    print("CSV:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    require_columns(
        df,
        [
            "time_s",
            "phase",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "actual_x_m",
            "actual_y_m",
            "actual_z_m",
            "error_mm",
            "percentage_error",
            "accuracy_percent",
            "radial_error_mm",
            "radial_error_abs_mm",
            "z_error_mm",
            "tangential_error_mm",
            "pressure_1_bar",
            "pressure_2_bar",
            "pressure_3_bar",
        ],
    )

    # Preserve physical time order.
    df = (
        df.sort_values(
            "time_s",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    numeric_columns = [
        "time_s",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "actual_x_m",
        "actual_y_m",
        "actual_z_m",
        "error_mm",
        "percentage_error",
        "accuracy_percent",
        "radial_error_mm",
        "radial_error_abs_mm",
        "z_error_mm",
        "tangential_error_mm",
        "pressure_1_bar",
        "pressure_2_bar",
        "pressure_3_bar",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    heart = df[
        df["phase"]
        .astype(str)
        .str.lower()
        == "heart"
    ].copy()

    if len(heart) < 2:
        raise RuntimeError(
            "Fewer than two heart-phase samples were found."
        )

    summary = load_summary(run_dir)

    # --------------------------------------------------------
    # 1. HEART XY MAP
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.plot(
        heart["target_x_m"].to_numpy()
        * 1000.0,
        heart["target_y_m"].to_numpy()
        * 1000.0,
        linestyle="--",
        linewidth=2.0,
        label="Target heart",
    )

    ax.plot(
        heart["actual_x_m"].to_numpy()
        * 1000.0,
        heart["actual_y_m"].to_numpy()
        * 1000.0,
        linewidth=2.2,
        label="Actual EE",
    )

    ax.scatter(
        heart["actual_x_m"].iloc[0]
        * 1000.0,
        heart["actual_y_m"].iloc[0]
        * 1000.0,
        s=80,
        label="Actual start",
        zorder=5,
    )

    ax.scatter(
        heart["actual_x_m"].iloc[-1]
        * 1000.0,
        heart["actual_y_m"].iloc[-1]
        * 1000.0,
        s=80,
        label="Actual end",
        zorder=5,
    )

    ax.set_title("XY Heart Trajectory Tracking")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_xy_trajectory.png",
    )

    # --------------------------------------------------------
    # 2. POSITION ERROR
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        heart["time_s"],
        heart["error_mm"],
        linewidth=1.8,
        label="Position error",
    )

    ax.set_title("Heart Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_error.png",
    )

    # --------------------------------------------------------
    # 3. ACCURACY
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        heart["time_s"],
        heart["accuracy_percent"],
        linewidth=1.8,
        label="Accuracy",
    )

    ax.axhline(
        92.0,
        linestyle="--",
        linewidth=1.4,
        label="92% benchmark",
    )

    ax.set_title("Heart Tracking Accuracy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(
        0.0,
        100.0,
    )
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_accuracy.png",
    )

    # --------------------------------------------------------
    # 4. CROSS-TRACK ERROR
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        heart["time_s"],
        heart["radial_error_mm"],
        linewidth=1.8,
        label="Local cross-track error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title("Heart Local Cross-Track Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cross-track error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_cross_track_error.png",
    )

    # --------------------------------------------------------
    # 5. Z TRACKING
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        heart["time_s"],
        heart["z_error_mm"],
        linewidth=1.8,
        label="Z error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title("Heart Z-Plane Tracking")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z error (mm)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_z_error.png",
    )

    # --------------------------------------------------------
    # 6. PRESSURES
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        heart["time_s"],
        heart["pressure_1_bar"],
        label="Pressure 1",
    )

    ax.plot(
        heart["time_s"],
        heart["pressure_2_bar"],
        label="Pressure 2",
    )

    ax.plot(
        heart["time_s"],
        heart["pressure_3_bar"],
        label="Pressure 3",
    )

    ax.set_title("Heart Run Pressure Commands")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (bar)")
    ax.grid(True, alpha=0.30)
    ax.legend()

    save(
        fig,
        run_dir / "heart_pressures.png",
    )

    # --------------------------------------------------------
    # 7. TEXT SUMMARY
    # --------------------------------------------------------

    heart_summary = summary.get(
        "heart_only",
        {},
    )

    benchmark = summary.get(
        "benchmark",
        {},
    )

    summary_txt = (
        run_dir
        / "heart_analysis.txt"
    )

    with summary_txt.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HEART TRAJECTORY ANALYSIS\n"
        )
        f.write(
            "=========================\n\n"
        )

        f.write(
            f"Run: {run_dir}\n"
        )
        f.write(
            f"Samples: {len(heart)}\n\n"
        )

        f.write(
            "BENCHMARK\n"
        )
        f.write(
            f"Required accuracy: "
            f"{benchmark.get('required_accuracy_percent', 92.0):.3f}%\n"
        )
        f.write(
            f"Heart accuracy: "
            f"{benchmark.get('heart_accuracy_percent', float('nan')):.3f}%\n"
        )
        f.write(
            f"Pass: "
            f"{benchmark.get('passed', False)}\n\n"
        )

        f.write(
            "HEART ONLY\n"
        )
        f.write(
            f"Mean error: "
            f"{heart_summary.get('mean_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"Median error: "
            f"{heart_summary.get('median_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"RMSE: "
            f"{heart_summary.get('rmse_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"P95 error: "
            f"{heart_summary.get('p95_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"Peak error: "
            f"{heart_summary.get('peak_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"Mean cross-track abs error: "
            f"{heart_summary.get('mean_radial_abs_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"P95 cross-track abs error: "
            f"{heart_summary.get('p95_radial_abs_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"Mean Z abs error: "
            f"{heart_summary.get('mean_z_abs_error_mm', float('nan')):.4f} mm\n"
        )
        f.write(
            f"P95 Z abs error: "
            f"{heart_summary.get('p95_z_abs_error_mm', float('nan')):.4f} mm\n"
        )

    print("Saved:", summary_txt)

    print()
    print("=" * 78)
    print("HEART PLOTTING COMPLETE")
    print("=" * 78)
    print("Output folder:")
    print(run_dir)
    print()


if __name__ == "__main__":
    main()
