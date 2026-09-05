#!/usr/bin/env python3
"""
ROBUST FIGURE-8 RESULT PLOTTER
==============================

This plotter is written to match main_figure8_FINAL_FIXED.py.

Important design choice:
    The figure-8 segment is selected by the recorded simulation time
    (approach_time_s <= time_s <= approach_time_s + figure8_time_s),
    NOT by the phase string.

That prevents the exact class of phase-label problem that previously caused
the controller to complete successfully but the plotting stage to fail.

Usage
-----
1) Automatically plot the newest figure-8 run:

    python plot_figure8_FINAL.py

2) Plot a specific run directory:

    python plot_figure8_FINAL.py ~/Desktop/soft_robot_runs/figure8_vla_mlp_YYYYMMDD_HHMMSS

3) Plot a specific trajectory CSV:

    python plot_figure8_FINAL.py /path/to/trajectory.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DESKTOP_ROOT = (
    Path.home()
    / "Desktop"
    / "soft_robot_runs"
)

RESULTS_ROOT = (
    Path.cwd()
    / "results"
)


REQUIRED_COLUMNS = (
    "time_s",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "actual_x_m",
    "actual_y_m",
    "actual_z_m",
    "error_mm",
    "accuracy_percent",
    "radial_error_mm",
    "radial_error_abs_mm",
    "z_error_mm",
    "tangential_error_mm",
    "pressure_1_bar",
    "pressure_2_bar",
    "pressure_3_bar",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def find_run() -> Path:
    """
    Resolve the requested run without assuming a particular current directory.
    """
    if len(sys.argv) > 1:
        requested = (
            Path(sys.argv[1])
            .expanduser()
            .resolve()
        )

        if not requested.exists():
            fail(
                "Specified path does not exist:\n"
                f"  {requested}"
            )

        if requested.is_file():
            if requested.name.lower() != "trajectory.csv":
                fail(
                    "Specified file must be trajectory.csv:\n"
                    f"  {requested}"
                )
            return requested.parent

        direct_csv = (
            requested
            / "trajectory.csv"
        )

        if direct_csv.is_file():
            return requested

        candidates = sorted(
            requested.rglob("trajectory.csv"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )

        if candidates:
            return candidates[0].parent

        fail(
            "No trajectory.csv found inside:\n"
            f"  {requested}"
        )

    if not DESKTOP_ROOT.exists():
        fail(
            "Desktop result directory was not found:\n"
            f"  {DESKTOP_ROOT}"
        )

    candidates = [
        p
        for p in DESKTOP_ROOT.rglob("trajectory.csv")
        if p.is_file()
        and p.parent.name.lower().startswith("figure8_")
    ]

    if not candidates:
        fail(
            "No figure-8 run containing trajectory.csv was found under:\n"
            f"  {DESKTOP_ROOT}"
        )

    candidates.sort(
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    return candidates[0].parent


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except Exception as exc:
        print(
            "WARNING: could not read",
            path,
            "->",
            exc,
        )
        return {}


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        fail(
            "CSV file does not exist:\n"
            f"  {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fail(
                "CSV has no header:\n"
                f"  {path}"
            )

        rows = list(reader)

    if not rows:
        fail(
            "CSV contains no trajectory rows:\n"
            f"  {path}"
        )

    return rows


def to_float(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return float("nan")

    return value


def rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    keys = set()

    for row in rows:
        keys.update(row.keys())

    for key in keys:
        arrays[key] = np.asarray(
            [
                to_float(row, key)
                for row in rows
            ],
            dtype=np.float64,
        )

    return arrays


def get_figure8_window(config: dict) -> tuple[float, float]:
    """
    Read the authoritative trajectory timing from config.json.

    main_figure8_FINAL_FIXED.py writes:
        approach_time_s
        figure8_time_s

    If config is unavailable, use the same constants as the controller.
    """
    figure_cfg = (
        config.get("figure-8", {})
        if isinstance(config, dict)
        else {}
    )

    try:
        approach = float(
            figure_cfg.get(
                "approach_time_s",
                6.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        approach = 6.0

    try:
        duration = float(
            figure_cfg.get(
                "figure8_time_s",
                45.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        duration = 45.0

    end = (
        approach
        + duration
    )

    return approach, end


def select_figure8(
    rows: list[dict],
    config: dict,
) -> tuple[list[dict], str]:
    """
    Select the actual figure-8 samples robustly.

    Priority:
        1. Authoritative time window from config.json.
        2. CSV time window using controller defaults.
        3. Dedicated figure-8 phase labels.
        4. Last-resort full trajectory segment excluding final hold.

    The first method is the intended one.
    """
    arrays = rows_to_arrays(rows)

    times = arrays.get(
        "time_s",
        np.full(
            len(rows),
            np.nan,
        ),
    )

    approach, end = get_figure8_window(
        config
    )

    finite_time = np.isfinite(times)

    mask = (
        finite_time
        & (times >= approach - 1e-9)
        & (times <= end + 1e-9)
    )

    selected = [
        row
        for row, keep in zip(
            rows,
            mask,
        )
        if keep
    ]

    if len(selected) >= 2:
        return (
            selected,
            "simulation-time window",
        )

    # Fallback: recognize all reasonable labels.
    phase_names = {
        "figure8",
        "figure-8",
        "figure_8",
        "figure8active",
        "figure-8active",
        "figure_8_active",
    }

    selected = []

    for row in rows:
        phase = str(
            row.get(
                "phase",
                "",
            )
        ).strip().lower()

        normalized = (
            phase
            .replace(
                " ",
                "",
            )
        )

        if normalized in phase_names:
            selected.append(row)

    if len(selected) >= 2:
        return (
            selected,
            "phase-label fallback",
        )

    # Last resort: use the expected figure-8 duration immediately before
    # the final hold. This is only reached for malformed/incomplete metadata.
    time_values = [
        to_float(
            row,
            "time_s",
        )
        for row in rows
    ]

    finite = [
        t
        for t in time_values
        if np.isfinite(t)
    ]

    if len(finite) >= 2:
        final_time = max(finite)

        hold_duration = 1.0

        figure8_end = (
            final_time
            - hold_duration
        )

        figure8_start = (
            figure8_end
            - (end - approach)
        )

        selected = [
            row
            for row, t in zip(
                rows,
                time_values,
            )
            if np.isfinite(t)
            and t >= figure8_start
            and t <= figure8_end + 1e-9
        ]

        if len(selected) >= 2:
            return (
                selected,
                "final-run time fallback",
            )

    fail(
        "Could not isolate at least two figure-8 samples.\n"
        "The trajectory.csv may be incomplete."
    )


def numeric_frame(
    rows: list[dict],
) -> dict[str, np.ndarray]:
    arrays = rows_to_arrays(rows)

    missing = [
        key
        for key in REQUIRED_COLUMNS
        if key not in arrays
    ]

    if missing:
        fail(
            "trajectory.csv is missing required columns:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing
            )
        )

    # Preserve chronological order. Never sort by X or Y.
    times = arrays["time_s"]

    order = np.argsort(
        times,
        kind="stable",
    )

    cleaned: dict[str, np.ndarray] = {}

    for key in REQUIRED_COLUMNS:
        cleaned[key] = arrays[key][order]

    # Remove rows where the core geometry is invalid.
    valid = (
        np.isfinite(
            cleaned["time_s"]
        )
        & np.isfinite(
            cleaned["target_x_m"]
        )
        & np.isfinite(
            cleaned["target_y_m"]
        )
        & np.isfinite(
            cleaned["actual_x_m"]
        )
        & np.isfinite(
            cleaned["actual_y_m"]
        )
    )

    for key in cleaned:
        cleaned[key] = cleaned[key][valid]

    if len(cleaned["time_s"]) < 2:
        fail(
            "Fewer than two valid figure-8 samples remain after "
            "numeric validation."
        )

    return cleaned


def load_dedicated_csv(
    run_dir: Path,
) -> tuple[list[dict] | None, Path | None]:
    """
    Prefer a dedicated figure8 CSV if one is inside the run directory.

    The main controller also writes timestamped figure8 CSV files directly
    under Desktop and a project-level results/ file. Those are intentionally
    not required here; trajectory.csv + config.json are sufficient and are
    the most reliable pair.
    """
    candidates = sorted(
        run_dir.glob(
            "figure8_xy_tracking*.csv"
        ),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    if not candidates:
        return None, None

    path = candidates[0]

    try:
        rows = load_csv(path)

        if len(rows) >= 2:
            return rows, path

    except Exception as exc:
        print(
            "WARNING: dedicated CSV could not be used:",
            exc,
        )

    return None, None


def save_figure(
    fig,
    path: Path,
) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(
        "Saved:",
        path,
    )


def add_endpoint_markers(
    ax,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    ax.scatter(
        x[0],
        y[0],
        s=80,
        label="Actual start",
        zorder=5,
    )

    ax.scatter(
        x[-1],
        y[-1],
        s=80,
        label="Actual end",
        zorder=5,
    )


def make_xy_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    target_x = (
        data["target_x_m"]
        * 1000.0
    )

    target_y = (
        data["target_y_m"]
        * 1000.0
    )

    actual_x = (
        data["actual_x_m"]
        * 1000.0
    )

    actual_y = (
        data["actual_y_m"]
        * 1000.0
    )

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.plot(
        target_x,
        target_y,
        linestyle="--",
        linewidth=2.0,
        label="Target figure-8",
    )

    ax.plot(
        actual_x,
        actual_y,
        linewidth=2.2,
        label="Actual EE",
    )

    add_endpoint_markers(
        ax,
        actual_x,
        actual_y,
    )

    ax.set_title(
        "XY Figure-8 Trajectory Tracking"
    )

    ax.set_xlabel(
        "X (mm)"
    )

    ax.set_ylabel(
        "Y (mm)"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_error_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        data["time_s"],
        data["error_mm"],
        linewidth=1.8,
        label="Position error",
    )

    ax.set_title(
        "Figure-8 Position Tracking Error"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Error (mm)"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_accuracy_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        data["time_s"],
        data["accuracy_percent"],
        linewidth=1.8,
        label="Accuracy",
    )

    ax.axhline(
        92.0,
        linestyle="--",
        linewidth=1.4,
        label="92% benchmark",
    )

    ax.set_title(
        "Figure-8 Tracking Accuracy"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_cross_track_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        data["time_s"],
        data["radial_error_mm"],
        linewidth=1.8,
        label="Local cross-track error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Figure-8 Local Cross-Track Error"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Cross-track error (mm)"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_z_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        data["time_s"],
        data["z_error_mm"],
        linewidth=1.8,
        label="Z error",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Figure-8 Z-Plane Tracking"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Z error (mm)"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_pressure_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        data["time_s"],
        data["pressure_1_bar"],
        label="Pressure 1",
    )

    ax.plot(
        data["time_s"],
        data["pressure_2_bar"],
        label="Pressure 2",
    )

    ax.plot(
        data["time_s"],
        data["pressure_3_bar"],
        label="Pressure 3",
    )

    ax.set_title(
        "Figure-8 Pressure Commands"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Pressure (bar)"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_target_actual_components_plot(
    data: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    time = data["time_s"]

    ax.plot(
        time,
        data["target_x_m"] * 1000.0,
        linestyle="--",
        linewidth=1.5,
        label="Target X",
    )

    ax.plot(
        time,
        data["actual_x_m"] * 1000.0,
        linewidth=1.8,
        label="Actual X",
    )

    ax.plot(
        time,
        data["target_y_m"] * 1000.0,
        linestyle="--",
        linewidth=1.5,
        label="Target Y",
    )

    ax.plot(
        time,
        data["actual_y_m"] * 1000.0,
        linewidth=1.8,
        label="Actual Y",
    )

    ax.set_title(
        "Figure-8 Target vs Actual XY Components"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Position (mm)"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend()

    save_figure(
        fig,
        output,
    )


def make_summary(
    data: dict[str, np.ndarray],
    config: dict,
    run_dir: Path,
    selection_method: str,
    output: Path,
) -> None:
    error = data["error_mm"]
    accuracy = data["accuracy_percent"]
    cross = data["radial_error_abs_mm"]
    z_error = np.abs(
        data["z_error_mm"]
    )

    target_x = (
        data["target_x_m"]
        * 1000.0
    )

    target_y = (
        data["target_y_m"]
        * 1000.0
    )

    actual_x = (
        data["actual_x_m"]
        * 1000.0
    )

    actual_y = (
        data["actual_y_m"]
        * 1000.0
    )

    target_path = float(
        np.sum(
            np.hypot(
                np.diff(target_x),
                np.diff(target_y),
            )
        )
    )

    actual_path = float(
        np.sum(
            np.hypot(
                np.diff(actual_x),
                np.diff(actual_y),
            )
        )
    )

    mean_error = float(
        np.mean(error)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    p95 = float(
        np.percentile(
            error,
            95,
        )
    )

    peak = float(
        np.max(error)
    )

    mean_accuracy = float(
        np.mean(accuracy)
    )

    mean_cross = float(
        np.mean(cross)
    )

    p95_cross = float(
        np.percentile(
            cross,
            95,
        )
    )

    mean_z = float(
        np.mean(z_error)
    )

    p95_z = float(
        np.percentile(
            z_error,
            95,
        )
    )

    approach, end = get_figure8_window(
        config
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "FIGURE-8 TRAJECTORY ANALYSIS\n"
        )
        f.write(
            "============================\n\n"
        )

        f.write(
            f"Run: {run_dir}\n"
        )

        f.write(
            f"Selection method: {selection_method}\n"
        )

        f.write(
            f"Samples: {len(error)}\n"
        )

        f.write(
            f"First time: {data['time_s'][0]:.6f} s\n"
        )

        f.write(
            f"Last time: {data['time_s'][-1]:.6f} s\n"
        )

        f.write(
            f"Expected figure-8 window: "
            f"{approach:.6f} -> {end:.6f} s\n\n"
        )

        f.write(
            "TRACKING\n"
        )
        f.write(
            f"Mean position error: {mean_error:.6f} mm\n"
        )
        f.write(
            f"RMSE: {rmse:.6f} mm\n"
        )
        f.write(
            f"P95 position error: {p95:.6f} mm\n"
        )
        f.write(
            f"Peak position error: {peak:.6f} mm\n"
        )
        f.write(
            f"Mean accuracy: {mean_accuracy:.6f} %\n"
        )
        f.write(
            f"92% benchmark: "
            f"{'PASS' if mean_accuracy >= 92.0 else 'FAIL'}\n\n"
        )

        f.write(
            "GEOMETRIC TRACKING\n"
        )
        f.write(
            f"Mean |cross-track| error: {mean_cross:.6f} mm\n"
        )
        f.write(
            f"P95 |cross-track| error: {p95_cross:.6f} mm\n"
        )
        f.write(
            f"Mean |Z| error: {mean_z:.6f} mm\n"
        )
        f.write(
            f"P95 |Z| error: {p95_z:.6f} mm\n\n"
        )

        f.write(
            "PATH LENGTH\n"
        )
        f.write(
            f"Target XY path length: {target_path:.6f} mm\n"
        )
        f.write(
            f"Actual XY path length: {actual_path:.6f} mm\n"
        )

        if target_path > 1e-9:
            f.write(
                f"Path-length ratio: "
                f"{actual_path / target_path:.6f}\n"
            )

        f.write(
            "\n"
        )

        figure_cfg = (
            config.get(
                "figure-8",
                {},
            )
            if isinstance(config, dict)
            else {}
        )

        f.write(
            "TRAJECTORY CONFIG\n"
        )
        f.write(
            f"Scale: "
            f"{figure_cfg.get('scale_m', 'unknown')} m\n"
        )
        f.write(
            f"Half width: "
            f"{figure_cfg.get('half_width_mm', 'unknown')} mm\n"
        )
        f.write(
            f"Approach time: "
            f"{figure_cfg.get('approach_time_s', approach)} s\n"
        )
        f.write(
            f"Figure-8 duration: "
            f"{figure_cfg.get('figure8_time_s', end - approach)} s\n"
        )


def main() -> None:
    print()
    print("=" * 82)
    print("ROBUST FIGURE-8 TRAJECTORY ANALYSIS")
    print("=" * 82)

    run_dir = find_run()

    trajectory_csv = (
        run_dir
        / "trajectory.csv"
    )

    config_path = (
        run_dir
        / "config.json"
    )

    config = load_json(
        config_path
    )

    print()
    print("Run:")
    print(" ", run_dir)

    print()
    print("Trajectory:")
    print(" ", trajectory_csv)

    if config_path.is_file():
        print()
        print("Config:")
        print(" ", config_path)

    rows = load_csv(
        trajectory_csv
    )

    # Prefer the controller's dedicated CSV if it exists, but only if it is
    # actually populated. Otherwise use trajectory.csv and isolate by time.
    dedicated_rows, dedicated_path = (
        load_dedicated_csv(
            run_dir
        )
    )

    if dedicated_rows is not None:
        try:
            figure8_rows, method = (
                select_figure8(
                    dedicated_rows,
                    config,
                )
            )
            source_description = (
                f"dedicated CSV: {dedicated_path}"
            )
        except Exception:
            figure8_rows, method = (
                select_figure8(
                    rows,
                    config,
                )
            )
            source_description = (
                "trajectory.csv"
            )
    else:
        figure8_rows, method = (
            select_figure8(
                rows,
                config,
            )
        )
        source_description = (
            "trajectory.csv"
        )

    data = numeric_frame(
        figure8_rows
    )

    output_dir = run_dir

    print()
    print("Figure-8 source:")
    print(" ", source_description)

    print(
        "Selection:",
        method,
    )

    print(
        "Samples:",
        len(data["time_s"]),
    )

    print(
        "Time:",
        f"{data['time_s'][0]:.4f} -> "
        f"{data['time_s'][-1]:.4f} s",
    )

    # Seven separate figures. No subplots, so each diagnostic remains easy
    # to inspect individually.
    make_xy_plot(
        data,
        output_dir
        / "figure8_xy_trajectory.png",
    )

    make_error_plot(
        data,
        output_dir
        / "figure8_error.png",
    )

    make_accuracy_plot(
        data,
        output_dir
        / "figure8_accuracy.png",
    )

    make_cross_track_plot(
        data,
        output_dir
        / "figure8_cross_track_error.png",
    )

    make_z_plot(
        data,
        output_dir
        / "figure8_z_error.png",
    )

    make_pressure_plot(
        data,
        output_dir
        / "figure8_pressures.png",
    )

    make_target_actual_components_plot(
        data,
        output_dir
        / "figure8_xy_components.png",
    )

    make_summary(
        data,
        config,
        run_dir,
        method,
        output_dir
        / "figure8_analysis.txt",
    )

    print(
        "Saved:",
        output_dir
        / "figure8_analysis.txt",
    )

    print()
    print("=" * 82)
    print("FIGURE-8 PLOTTING COMPLETE")
    print("=" * 82)


if __name__ == "__main__":
    main()
