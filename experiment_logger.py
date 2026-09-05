#!/usr/bin/env python3

"""
SOFT ROBOT EXPERIMENT LOGGER

This file does NOT modify main.py.

It:
    1. Runs:
           mjpython main.py

    2. Captures the terminal output.

    3. Extracts:
           target position
           measured EE position
           error
           pressure

    4. Calculates:
           mean error
           RMSE
           maximum error
           mean accuracy
           minimum accuracy
           final accuracy
           percentage of samples within 5% error
           percentage of samples within 10% error

    5. Saves everything to:

       PROJECT:
           data/experiments/<RUN_ID>/

       DESKTOP:
           ~/Desktop/soft_robot_runs/<RUN_ID>/

    Files:
           trajectory.csv
           trajectory.npz
           summary.json
           trajectory_3d.png
           position_vs_time.png
           tracking_error.png
           pressure_vs_time.png
           console_output.txt

Run:

    cd ~/mujoco_projects
    conda activate openvla-oft
    PYTHONPATH=. python experiment_logger.py
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# MATPLOTLIB
# ============================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MAIN_FILE = ROOT / "main.py"

PROJECT_RESULTS = (
    ROOT
    / "data"
    / "experiments"
)

DESKTOP_RESULTS = (
    Path.home()
    / "Desktop"
    / "soft_robot_runs"
)


# ============================================================
# TRAJECTORY LINE REGEX
# ============================================================

"""
Supported examples:

J 001/133 | target [0.038 0.000 0.155] | EE [0.020 0.001 0.110] | err 48.2 mm | P [1.0 2.0 0.5]

WP 001/538 | target [...] | EE [...] | err 8.20 mm | dP [...] | P [...]

Also allows:

J 87.1% | target [...] | EE [...] | err 14.3 mm | P [...]

The important part is that the parser does NOT require
001/133 anymore.
"""

TRAJECTORY_RE = re.compile(
    r"""
    ^\s*

    (?P<label>J|WP)

    \s+

    (?:
        (?P<index>\d+)
        \s*/\s*
        (?P<total>\d+)
        |
        (?P<accuracy_printed>
        [-+]?[0-9]*\.?[0-9]+
        )
        \s*%
    )

    \s*\|\s*

    target
    \s*
    \[
        (?P<target>[^\]]+)
    \]

    \s*\|\s*

    EE
    \s*
    \[
        (?P<ee>[^\]]+)
    \]

    \s*\|\s*

    err
    \s*
    (?P<err>
        [-+]?[0-9]*\.?[0-9]+
        (?:[eE][-+]?[0-9]+)?
    )
    \s*mm

    (?:

        \s*\|\s*

        dP
        \s*
        \[
            (?P<dp>[^\]]+)
        \]

    )?

    \s*\|\s*

    P
    \s*
    \[
        (?P<pressure>[^\]]+)
    \]

    """,
    re.VERBOSE,
)


# ============================================================
# FINAL RESULT REGEX
# ============================================================

FINAL_TARGET_RE = re.compile(
    r"Final target:\s*\[([^\]]+)\]"
)

FINAL_EE_RE = re.compile(
    r"Final EE:\s*\[([^\]]+)\]"
)

FINAL_ERROR_RE = re.compile(
    r"Final error magnitude:\s*"
    r"([-+]?[0-9]*\.?[0-9]+)"
    r"\s*mm"
)

FINAL_ACCURACY_RE = re.compile(
    r"Final\s+"
    r"(?:percentage\s+)?"
    r"accuracy:\s*"
    r"([-+]?[0-9]*\.?[0-9]+)"
    r"\s*%"
)


# ============================================================
# HELPERS
# ============================================================

def clean_line(line):
    """
    Remove ANSI terminal escape sequences.
    """

    return re.sub(
        r"\x1b\[[0-?]*[ -/]*[@-~]",
        "",
        line,
    )


def parse_vector(text):
    """
    Convert:

        '0.1 0.2 0.3'

    or:

        '0.1, 0.2, 0.3'

    into numpy array.
    """

    text = text.replace(",", " ")

    values = np.fromstring(
        text,
        sep=" ",
        dtype=np.float64,
    )

    if values.size != 3:

        raise ValueError(
            f"Expected 3 values, received: {text}"
        )

    return values


def make_run_id():

    return time.strftime(
        "%Y%m%d_%H%M%S"
    )


def create_result_directories(run_id):

    project_dir = (
        PROJECT_RESULTS
        / run_id
    )

    desktop_dir = (
        DESKTOP_RESULTS
        / run_id
    )

    project_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    desktop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        project_dir,
        desktop_dir,
    )


# ============================================================
# EXPERIMENT CAPTURE
# ============================================================

class ExperimentCapture:

    def __init__(self):

        self.rows = []

        self.raw_lines = []

        self.final_target = None

        self.final_ee = None

        self.final_error_mm_printed = None

        self.final_accuracy_printed = None

        self.start_time = time.time()


    # --------------------------------------------------------
    # FEED ONE CONSOLE LINE
    # --------------------------------------------------------

    def feed(self, raw_line):

        line = clean_line(
            raw_line.rstrip("\n")
        )

        self.raw_lines.append(line)

        # ====================================================
        # TRAJECTORY
        # ====================================================

        match = TRAJECTORY_RE.search(line)

        if match:

            try:

                target = parse_vector(
                    match.group("target")
                )

                ee = parse_vector(
                    match.group("ee")
                )

                pressure = parse_vector(
                    match.group("pressure")
                )

                dp_text = match.group("dp")

                if dp_text is None:

                    dp = np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                        ],
                        dtype=np.float64,
                    )

                else:

                    dp = parse_vector(
                        dp_text
                    )


                # --------------------------------------------
                # ERROR
                # --------------------------------------------

                error_vector = (
                    target - ee
                )

                calculated_error_mm = (
                    np.linalg.norm(
                        error_vector
                    )
                    * 1000.0
                )


                # --------------------------------------------
                # RELATIVE ERROR
                # --------------------------------------------

                target_norm = max(
                    np.linalg.norm(target),
                    1e-12,
                )

                error_percent = (
                    np.linalg.norm(
                        error_vector
                    )
                    / target_norm
                    * 100.0
                )


                # --------------------------------------------
                # ACCURACY
                # --------------------------------------------

                accuracy_percent = max(
                    0.0,
                    100.0 - error_percent,
                )


                # --------------------------------------------
                # INDEX
                # --------------------------------------------

                index_text = match.group(
                    "index"
                )

                total_text = match.group(
                    "total"
                )

                if index_text is not None:

                    trajectory_index = int(
                        index_text
                    )

                else:

                    trajectory_index = (
                        len(self.rows) + 1
                    )


                if total_text is not None:

                    trajectory_total = int(
                        total_text
                    )

                else:

                    trajectory_total = (
                        len(self.rows) + 1
                    )


                # --------------------------------------------
                # PRINTED ACCURACY
                # --------------------------------------------

                printed_accuracy = (
                    match.group(
                        "accuracy_printed"
                    )
                )

                if printed_accuracy is not None:

                    printed_accuracy = float(
                        printed_accuracy
                    )


                # --------------------------------------------
                # SAVE ROW
                # --------------------------------------------

                row = {

                    "sample":
                        len(self.rows) + 1,

                    "trajectory_index":
                        trajectory_index,

                    "trajectory_total":
                        trajectory_total,

                    "label":
                        match.group("label"),

                    "elapsed_s":
                        time.time()
                        - self.start_time,


                    # TARGET
                    "target_x_m":
                        float(target[0]),

                    "target_y_m":
                        float(target[1]),

                    "target_z_m":
                        float(target[2]),


                    # EE
                    "ee_x_m":
                        float(ee[0]),

                    "ee_y_m":
                        float(ee[1]),

                    "ee_z_m":
                        float(ee[2]),


                    # ERROR VECTOR
                    "error_x_mm":
                        float(
                            error_vector[0]
                            * 1000.0
                        ),

                    "error_y_mm":
                        float(
                            error_vector[1]
                            * 1000.0
                        ),

                    "error_z_mm":
                        float(
                            error_vector[2]
                            * 1000.0
                        ),


                    # ERROR
                    "error_mm":
                        float(
                            calculated_error_mm
                        ),

                    "error_percent":
                        float(
                            error_percent
                        ),

                    "accuracy_percent":
                        float(
                            accuracy_percent
                        ),


                    # PRINTED ACCURACY
                    "main_printed_accuracy":
                        (
                            printed_accuracy
                            if printed_accuracy
                            is not None
                            else np.nan
                        ),


                    # PRESSURE
                    "pressure_1":
                        float(pressure[0]),

                    "pressure_2":
                        float(pressure[1]),

                    "pressure_3":
                        float(pressure[2]),


                    # PRESSURE CHANGE
                    "dpressure_1":
                        float(dp[0]),

                    "dpressure_2":
                        float(dp[1]),

                    "dpressure_3":
                        float(dp[2]),
                }


                self.rows.append(row)


            except Exception as exc:

                print(
                    "\n[LOGGER WARNING]"
                    f" Could not parse trajectory:"
                    f" {exc}",
                    file=sys.stderr,
                )


        # ====================================================
        # FINAL TARGET
        # ====================================================

        match = FINAL_TARGET_RE.search(
            line
        )

        if match:

            try:

                self.final_target = (
                    parse_vector(
                        match.group(1)
                    )
                )

            except Exception:

                pass


        # ====================================================
        # FINAL EE
        # ====================================================

        match = FINAL_EE_RE.search(
            line
        )

        if match:

            try:

                self.final_ee = (
                    parse_vector(
                        match.group(1)
                    )
                )

            except Exception:

                pass


        # ====================================================
        # FINAL ERROR
        # ====================================================

        match = FINAL_ERROR_RE.search(
            line
        )

        if match:

            try:

                self.final_error_mm_printed = (
                    float(
                        match.group(1)
                    )
                )

            except Exception:

                pass


        # ====================================================
        # FINAL ACCURACY
        # ====================================================

        match = FINAL_ACCURACY_RE.search(
            line
        )

        if match:

            try:

                self.final_accuracy_printed = (
                    float(
                        match.group(1)
                    )
                )

            except Exception:

                pass


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [

    "sample",

    "trajectory_index",

    "trajectory_total",

    "label",

    "elapsed_s",


    "target_x_m",

    "target_y_m",

    "target_z_m",


    "ee_x_m",

    "ee_y_m",

    "ee_z_m",


    "error_x_mm",

    "error_y_mm",

    "error_z_mm",

    "error_mm",


    "error_percent",

    "accuracy_percent",

    "main_printed_accuracy",


    "pressure_1",

    "pressure_2",

    "pressure_3",


    "dpressure_1",

    "dpressure_2",

    "dpressure_3",
]


def save_csv(rows, path):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(row)


# ============================================================
# NPZ
# ============================================================

def save_npz(rows, path):

    if not rows:

        np.savez(
            path,

            target=np.empty(
                (0, 3)
            ),

            ee=np.empty(
                (0, 3)
            ),

            error_mm=np.empty(
                (0, 3)
            ),

            pressure=np.empty(
                (0, 3)
            ),

            dpressure=np.empty(
                (0, 3)
            ),

            elapsed_s=np.empty(
                (0,)
            ),
        )

        return


    target = np.array(
        [
            [
                r["target_x_m"],
                r["target_y_m"],
                r["target_z_m"],
            ]

            for r in rows
        ]
    )


    ee = np.array(
        [
            [
                r["ee_x_m"],
                r["ee_y_m"],
                r["ee_z_m"],
            ]

            for r in rows
        ]
    )


    error_mm = np.array(
        [
            [
                r["error_x_mm"],
                r["error_y_mm"],
                r["error_z_mm"],
            ]

            for r in rows
        ]
    )


    pressure = np.array(
        [
            [
                r["pressure_1"],
                r["pressure_2"],
                r["pressure_3"],
            ]

            for r in rows
        ]
    )


    dpressure = np.array(
        [
            [
                r["dpressure_1"],
                r["dpressure_2"],
                r["dpressure_3"],
            ]

            for r in rows
        ]
    )


    elapsed_s = np.array(
        [
            r["elapsed_s"]
            for r in rows
        ]
    )


    np.savez(
        path,

        target=target,

        ee=ee,

        error_mm=error_mm,

        pressure=pressure,

        dpressure=dpressure,

        elapsed_s=elapsed_s,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(capture):

    rows = capture.rows


    if not rows:

        return {

            "samples":
                0,

            "status":
                "NO TRAJECTORY SAMPLES CAPTURED",

        }


    errors = np.array(
        [
            r["error_mm"]
            for r in rows
        ],
        dtype=np.float64,
    )


    error_percent = np.array(
        [
            r["error_percent"]
            for r in rows
        ],
        dtype=np.float64,
    )


    accuracy = np.array(
        [
            r["accuracy_percent"]
            for r in rows
        ],
        dtype=np.float64,
    )


    # ========================================================
    # FINAL TARGET
    # ========================================================

    if capture.final_target is not None:

        final_target = (
            capture.final_target
        )

    else:

        final_target = np.array(
            [
                rows[-1]["target_x_m"],
                rows[-1]["target_y_m"],
                rows[-1]["target_z_m"],
            ]
        )


    # ========================================================
    # FINAL EE
    # ========================================================

    if capture.final_ee is not None:

        final_ee = (
            capture.final_ee
        )

    else:

        final_ee = np.array(
            [
                rows[-1]["ee_x_m"],
                rows[-1]["ee_y_m"],
                rows[-1]["ee_z_m"],
            ]
        )


    # ========================================================
    # FINAL ERROR
    # ========================================================

    final_error_vector = (
        final_target
        - final_ee
    )


    final_error_mm = (
        np.linalg.norm(
            final_error_vector
        )
        * 1000.0
    )


    final_target_norm = max(
        np.linalg.norm(
            final_target
        ),
        1e-12,
    )


    final_error_percent = (
        np.linalg.norm(
            final_error_vector
        )
        / final_target_norm
        * 100.0
    )


    final_accuracy = max(
        0.0,
        100.0 - final_error_percent,
    )


    # ========================================================
    # WITHIN THRESHOLDS
    # ========================================================

    within_5 = (
        np.mean(
            error_percent <= 5.0
        )
        * 100.0
    )


    within_10 = (
        np.mean(
            error_percent <= 10.0
        )
        * 100.0
    )


    return {

        "status":
            "OK",

        "samples":
            int(len(rows)),

        "trajectory_total":
            int(
                rows[-1]
                ["trajectory_total"]
            ),

        "trajectory_label":
            rows[-1]["label"],


        # ERROR
        "mean_error_mm":
            float(
                np.mean(errors)
            ),

        "median_error_mm":
            float(
                np.median(errors)
            ),

        "rmse_error_mm":
            float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),

        "minimum_error_mm":
            float(
                np.min(errors)
            ),

        "maximum_error_mm":
            float(
                np.max(errors)
            ),


        # ACCURACY
        "mean_accuracy_percent":
            float(
                np.mean(accuracy)
            ),

        "median_accuracy_percent":
            float(
                np.median(accuracy)
            ),

        "minimum_accuracy_percent":
            float(
                np.min(accuracy)
            ),

        "maximum_accuracy_percent":
            float(
                np.max(accuracy)
            ),


        # THRESHOLDS
        "samples_within_5_percent_error":
            float(within_5),

        "samples_within_10_percent_error":
            float(within_10),


        # FINAL
        "final_target_m":
            final_target.tolist(),

        "final_ee_m":
            final_ee.tolist(),

        "final_error_vector_mm":
            (
                final_error_vector
                * 1000.0
            ).tolist(),

        "final_error_mm":
            float(final_error_mm),

        "final_error_percent":
            float(final_error_percent),

        "final_accuracy_percent":
            float(final_accuracy),


        # VALUES PRINTED BY MAIN
        "main_final_error_mm":
            capture.final_error_mm_printed,

        "main_final_accuracy_percent":
            capture.final_accuracy_printed,


        "logged_duration_s":
            float(
                rows[-1]
                ["elapsed_s"]
            ),
    }


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    rows,
    output_dir,
):

    if not rows:

        return []


    t = np.array(
        [
            r["elapsed_s"]
            for r in rows
        ]
    )


    target = np.array(
        [
            [
                r["target_x_m"],
                r["target_y_m"],
                r["target_z_m"],
            ]

            for r in rows
        ]
    )


    ee = np.array(
        [
            [
                r["ee_x_m"],
                r["ee_y_m"],
                r["ee_z_m"],
            ]

            for r in rows
        ]
    )


    error_mm = np.array(
        [
            r["error_mm"]
            for r in rows
        ]
    )


    error_percent = np.array(
        [
            r["error_percent"]
            for r in rows
        ]
    )


    accuracy = np.array(
        [
            r["accuracy_percent"]
            for r in rows
        ]
    )


    pressure = np.array(
        [
            [
                r["pressure_1"],
                r["pressure_2"],
                r["pressure_3"],
            ]

            for r in rows
        ]
    )


    created = []


    # ========================================================
    # 1. 3D TRAJECTORY
    # ========================================================

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )


    ax.plot(
        target[:, 0],
        target[:, 1],
        target[:, 2],
        label="Target trajectory",
        linewidth=2.5,
    )


    ax.plot(
        ee[:, 0],
        ee[:, 1],
        ee[:, 2],
        label="Measured EE trajectory",
        linewidth=2.0,
    )


    ax.scatter(
        target[0, 0],
        target[0, 1],
        target[0, 2],
        s=45,
        label="Start",
    )


    ax.scatter(
        target[-1, 0],
        target[-1, 1],
        target[-1, 2],
        s=45,
        label="End",
    )


    ax.set_xlabel(
        "x (m)"
    )

    ax.set_ylabel(
        "y (m)"
    )

    ax.set_zlabel(
        "z (m)"
    )

    ax.set_title(
        "J Trajectory: Target vs Measured"
    )

    ax.legend()

    fig.tight_layout()


    path = (
        output_dir
        / "trajectory_3d.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    created.append(path)


    # ========================================================
    # 2. POSITION VS TIME
    # ========================================================

    fig = plt.figure(
        figsize=(10, 6)
    )

    ax = fig.add_subplot(111)


    names = [
        "x",
        "y",
        "z",
    ]


    for i, name in enumerate(names):

        ax.plot(
            t,
            target[:, i],
            "--",
            label=f"Target {name}",
        )

        ax.plot(
            t,
            ee[:, i],
            label=f"Measured {name}",
        )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Position (m)"
    )

    ax.set_title(
        "Target vs Measured Position"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        ncol=2
    )


    fig.tight_layout()


    path = (
        output_dir
        / "position_vs_time.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    created.append(path)


    # ========================================================
    # 3. TRACKING ERROR
    # ========================================================

    fig = plt.figure(
        figsize=(10, 6)
    )

    ax = fig.add_subplot(111)


    ax.plot(
        t,
        error_mm,
        label="Euclidean error (mm)",
        linewidth=2.0,
    )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Error (mm)"
    )

    ax.set_title(
        "Trajectory Tracking Error"
    )

    ax.grid(
        True,
        alpha=0.25,
    )


    ax2 = ax.twinx()


    ax2.plot(
        t,
        error_percent,
        "--",
        label="Relative error (%)",
        linewidth=1.5,
    )


    ax2.set_ylabel(
        "Relative error (%)"
    )


    handles1, labels1 = (
        ax.get_legend_handles_labels()
    )

    handles2, labels2 = (
        ax2.get_legend_handles_labels()
    )


    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
    )


    fig.tight_layout()


    path = (
        output_dir
        / "tracking_error.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    created.append(path)


    # ========================================================
    # 4. ACCURACY VS TIME
    # ========================================================

    fig = plt.figure(
        figsize=(10, 6)
    )

    ax = fig.add_subplot(111)


    ax.plot(
        t,
        accuracy,
        linewidth=2.0,
        label="Tracking accuracy",
    )


    ax.axhline(
        90.0,
        linestyle="--",
        label="90% accuracy",
    )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Trajectory Tracking Accuracy"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()


    fig.tight_layout()


    path = (
        output_dir
        / "accuracy_vs_time.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    created.append(path)


    # ========================================================
    # 5. PRESSURE
    # ========================================================

    fig = plt.figure(
        figsize=(10, 6)
    )

    ax = fig.add_subplot(111)


    for i in range(3):

        ax.plot(
            t,
            pressure[:, i],
            label=f"Pressure {i + 1}",
        )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Pressure"
    )

    ax.set_title(
        "Pressure Command History"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()


    fig.tight_layout()


    path = (
        output_dir
        / "pressure_vs_time.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    created.append(path)


    return created


# ============================================================
# HISTORY CSV
# ============================================================

def update_history(
    summary,
    project_dir,
    desktop_dir,
):

    fields = [

        "run_id",

        "timestamp",

        "samples",

        "mean_error_mm",

        "rmse_error_mm",

        "maximum_error_mm",

        "mean_accuracy_percent",

        "minimum_accuracy_percent",

        "final_error_mm",

        "final_accuracy_percent",

        "samples_within_5_percent_error",

        "samples_within_10_percent_error",

        "logged_duration_s",

        "trajectory_file",

        "summary_file",
    ]


    row = {

        "run_id":
            summary["run_id"],

        "timestamp":
            summary["timestamp"],

        "samples":
            summary.get(
                "samples",
                "",
            ),

        "mean_error_mm":
            summary.get(
                "mean_error_mm",
                "",
            ),

        "rmse_error_mm":
            summary.get(
                "rmse_error_mm",
                "",
            ),

        "maximum_error_mm":
            summary.get(
                "maximum_error_mm",
                "",
            ),

        "mean_accuracy_percent":
            summary.get(
                "mean_accuracy_percent",
                "",
            ),

        "minimum_accuracy_percent":
            summary.get(
                "minimum_accuracy_percent",
                "",
            ),

        "final_error_mm":
            summary.get(
                "final_error_mm",
                "",
            ),

        "final_accuracy_percent":
            summary.get(
                "final_accuracy_percent",
                "",
            ),

        "samples_within_5_percent_error":
            summary.get(
                "samples_within_5_percent_error",
                "",
            ),

        "samples_within_10_percent_error":
            summary.get(
                "samples_within_10_percent_error",
                "",
            ),

        "logged_duration_s":
            summary.get(
                "logged_duration_s",
                "",
            ),

        "trajectory_file":
            f"{summary['run_id']}/trajectory.csv",

        "summary_file":
            f"{summary['run_id']}/summary.json",
    }


    project_history = (
        PROJECT_RESULTS
        / "run_history.csv"
    )

    desktop_history = (
        DESKTOP_RESULTS
        / "run_history.csv"
    )


    for history_path in (
        project_history,
        desktop_history,
    ):

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        write_header = (
            not history_path.exists()
        )


        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )


            if write_header:

                writer.writeheader()


            writer.writerow(row)


# ============================================================
# RUN MAIN.PY
# ============================================================

def run_experiment(capture):

    if not MAIN_FILE.exists():

        raise FileNotFoundError(
            f"\nmain.py not found:\n"
            f"{MAIN_FILE}\n"
        )


    mjpython = shutil.which(
        "mjpython"
    )


    if mjpython is None:

        raise RuntimeError(
            "\nmjpython was not found.\n"
            "Make sure the correct conda environment is active.\n"
        )


    command = [

        mjpython,

        str(MAIN_FILE),
    ]


    env = os.environ.copy()


    # Make output immediately available.
    env["PYTHONUNBUFFERED"] = "1"


    # Keep project import path.
    current_pythonpath = (
        env.get(
            "PYTHONPATH",
            "",
        )
    )


    if current_pythonpath:

        env["PYTHONPATH"] = (
            str(ROOT)
            + os.pathsep
            + current_pythonpath
        )

    else:

        env["PYTHONPATH"] = str(ROOT)


    print()
    print("=" * 72)
    print(
        "SOFT ROBOT EXPERIMENT LOGGER"
    )
    print("=" * 72)
    print()

    print(
        "Project:",
        ROOT,
    )

    print(
        "main.py:",
        MAIN_FILE,
    )

    print(
        "mjpython:",
        mjpython,
    )

    print()

    print(
        "main.py will NOT be modified."
    )

    print(
        "Capturing trajectory output..."
    )

    print()


    process = subprocess.Popen(

        command,

        cwd=str(ROOT),

        env=env,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True,

        bufsize=1,

        universal_newlines=True,
    )


    try:

        for line in process.stdout:

            # Show normal output live.
            print(
                line,
                end="",
            )

            # Give logger the same line.
            capture.feed(line)


    except KeyboardInterrupt:

        print()

        print(
            "[LOGGER] Ctrl+C received."
        )

        print(
            "[LOGGER] Stopping main.py..."
        )


        try:

            process.terminate()

            process.wait(
                timeout=5
            )

        except subprocess.TimeoutExpired:

            process.kill()

            process.wait()


    finally:

        if process.stdout is not None:

            process.stdout.close()


    return_code = (
        process.returncode
    )


    print()

    print("=" * 72)

    print(
        "EXPERIMENT FINISHED"
    )

    print("=" * 72)

    print()

    print(
        "Return code:",
        return_code,
    )

    print(
        "Captured trajectory samples:",
        len(capture.rows),
    )

    return return_code


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    capture,
    run_id,
    return_code,
):

    project_dir, desktop_dir = (
        create_result_directories(
            run_id
        )
    )


    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    console_text = (
        "\n".join(
            capture.raw_lines
        )
        + "\n"
    )


    for directory in (
        project_dir,
        desktop_dir,
    ):

        (
            directory
            / "console_output.txt"
        ).write_text(
            console_text,
            encoding="utf-8",
        )


    # ========================================================
    # CSV
    # ========================================================

    project_csv = (
        project_dir
        / "trajectory.csv"
    )

    desktop_csv = (
        desktop_dir
        / "trajectory.csv"
    )


    save_csv(
        capture.rows,
        project_csv,
    )

    shutil.copy2(
        project_csv,
        desktop_csv,
    )


    # ========================================================
    # NPZ
    # ========================================================

    project_npz = (
        project_dir
        / "trajectory.npz"
    )

    desktop_npz = (
        desktop_dir
        / "trajectory.npz"
    )


    save_npz(
        capture.rows,
        project_npz,
    )

    shutil.copy2(
        project_npz,
        desktop_npz,
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        capture
    )


    summary = {

        "run_id":
            run_id,

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "project_root":
            str(ROOT),

        "main_file":
            str(MAIN_FILE),

        "process_return_code":
            return_code,

        **metrics,
    }


    # ========================================================
    # JSON
    # ========================================================

    project_json = (
        project_dir
        / "summary.json"
    )

    desktop_json = (
        desktop_dir
        / "summary.json"
    )


    with project_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    shutil.copy2(
        project_json,
        desktop_json,
    )


    # ========================================================
    # PLOTS
    # ========================================================

    plots = make_plots(
        capture.rows,
        project_dir,
    )


    for plot in plots:

        shutil.copy2(
            plot,
            desktop_dir
            / plot.name,
        )


    # ========================================================
    # HISTORY
    # ========================================================

    update_history(
        summary,
        project_dir,
        desktop_dir,
    )


    return (
        project_dir,
        desktop_dir,
        summary,
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    project_dir,
    desktop_dir,
    summary,
):

    print()

    print("=" * 72)

    print(
        "RUN SAVED"
    )

    print("=" * 72)

    print()


    print(
        "Project folder:"
    )

    print(
        project_dir
    )

    print()


    print(
        "Desktop folder:"
    )

    print(
        desktop_dir
    )

    print()


    if summary.get(
        "samples",
        0,
    ) == 0:

        print(
            "WARNING:"
        )

        print(
            "No trajectory samples were captured."
        )

        print()

        print(
            "Open:"
        )

        print(
            project_dir
            / "console_output.txt"
        )

        print(
            "to inspect the raw output."
        )

        return


    # ========================================================
    # MAIN RESULTS
    # ========================================================

    print(
        f"Samples: "
        f"{summary['samples']}"
    )


    print(
        f"Mean error: "
        f"{summary['mean_error_mm']:.3f} mm"
    )


    print(
        f"RMSE error: "
        f"{summary['rmse_error_mm']:.3f} mm"
    )


    print(
        f"Maximum error: "
        f"{summary['maximum_error_mm']:.3f} mm"
    )


    print()


    print(
        f"MEAN ACCURACY: "
        f"{summary['mean_accuracy_percent']:.2f}%"
    )


    print(
        f"Minimum accuracy: "
        f"{summary['minimum_accuracy_percent']:.2f}%"
    )


    print(
        f"Maximum accuracy: "
        f"{summary['maximum_accuracy_percent']:.2f}%"
    )


    print()


    print(
        f"Samples within 5% error: "
        f"{summary['samples_within_5_percent_error']:.2f}%"
    )


    print(
        f"Samples within 10% error: "
        f"{summary['samples_within_10_percent_error']:.2f}%"
    )


    print()


    print(
        f"Final error: "
        f"{summary['final_error_mm']:.3f} mm"
    )


    print(
        f"Final accuracy: "
        f"{summary['final_accuracy_percent']:.2f}%"
    )


    print()


    # ========================================================
    # FILES
    # ========================================================

    print(
        "Saved files:"
    )

    print(
        "  trajectory.csv"
    )

    print(
        "  trajectory.npz"
    )

    print(
        "  summary.json"
    )

    print(
        "  trajectory_3d.png"
    )

    print(
        "  position_vs_time.png"
    )

    print(
        "  tracking_error.png"
    )

    print(
        "  accuracy_vs_time.png"
    )

    print(
        "  pressure_vs_time.png"
    )

    print(
        "  console_output.txt"
    )

    print()

    print(
        "The same files were copied to Desktop."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    run_id = make_run_id()

    capture = ExperimentCapture()


    return_code = run_experiment(
        capture
    )


    (
        project_dir,
        desktop_dir,
        summary,
    ) = save_results(
        capture,
        run_id,
        return_code,
    )


    print_summary(
        project_dir,
        desktop_dir,
        summary,
    )


    print()

    print("=" * 72)

    print(
        "LOGGER FINISHED"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()