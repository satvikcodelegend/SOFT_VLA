#!/usr/bin/env python3

"""
======================================================================
SOFT ROBOT VLA - LARGE CURVED TRAJECTORY DATASET GENERATOR
======================================================================

Generates a large dataset of continuous curved end-effector
trajectories and corresponding pressure commands.

Trajectory families:
    circle
    ellipse
    figure8
    parabola
    heart
    spiral
    lissajous
    wave
    multi_frequency
    random_smooth
    hybrid

The inverse MLP converts:

    desired XYZ position -> [p1, p2, p3]

The generator is intentionally independent of the exact MLP
implementation. It reconstructs the architecture directly from the
checkpoint so checkpoint architecture mismatches do not occur.

Output:

    dataset/vla_data/curved_trajectories.npz
    dataset/vla_data/manifest.jsonl
    dataset/vla_data/metadata.json

Target:
    100,000 trajectories

======================================================================
"""

import os
import json
import time
import math
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    ROOT,
    "model",
    "inverse_mlp.pt"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "dataset",
    "vla_data"
)

NPZ_PATH = os.path.join(
    OUTPUT_DIR,
    "curved_trajectories.npz"
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "manifest.jsonl"
)

METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)


# ----------------------------------------------------------------------
# Dataset size
# ----------------------------------------------------------------------

NUM_TRAJECTORIES = 100_000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ----------------------------------------------------------------------
# Trajectory resolution
# ----------------------------------------------------------------------

MIN_POINTS = 80
MAX_POINTS = 220


# ----------------------------------------------------------------------
# Robot workspace
#
# Based on the inverse dataset generated previously.
# Keep a small margin from the observed workspace boundaries.
# ----------------------------------------------------------------------

WORKSPACE_MIN = np.array(
    [-0.285, -0.270, 0.050],
    dtype=np.float32
)

WORKSPACE_MAX = np.array(
    [0.285, 0.270, 0.205],
    dtype=np.float32
)


# ----------------------------------------------------------------------
# Pressure limits
# ----------------------------------------------------------------------

PRESSURE_MIN = 0.0
PRESSURE_MAX = 3.0


# ----------------------------------------------------------------------
# Trajectory validity
# ----------------------------------------------------------------------

MIN_PATH_LENGTH = 0.08
MAX_PATH_LENGTH = 1.60

MAX_POSITION_STEP = 0.035

MAX_PRESSURE_STEP = 0.20


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# ======================================================================
# UTILITIES
# ======================================================================

def clamp(x, lo, hi):
    return np.minimum(
        np.maximum(x, lo),
        hi
    )


def smoothstep(x):
    x = np.asarray(x)
    return (
        x * x *
        (3.0 - 2.0 * x)
    )


def normalize_vector(v):
    n = np.linalg.norm(v)

    if n < 1e-12:
        return np.zeros_like(v)

    return v / n


def path_length(points):
    if len(points) < 2:
        return 0.0

    diff = np.diff(
        points,
        axis=0
    )

    return float(
        np.sum(
            np.linalg.norm(
                diff,
                axis=1
            )
        )
    )


def point_spacing(points):
    if len(points) < 2:
        return np.zeros(0)

    return np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )


def trajectory_hash(points):
    rounded = np.round(
        points,
        decimals=4
    )

    return hashlib.sha1(
        rounded.tobytes()
    ).hexdigest()


# ======================================================================
# INVERSE MLP
# ======================================================================

class InverseModel:

    def __init__(
        self,
        checkpoint_path,
        device
    ):

        self.device = device

        print()
        print("=" * 70)
        print("LOADING INVERSE MLP")
        print("=" * 70)

        print()
        print("Checkpoint:")
        print(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        self.checkpoint = checkpoint

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print()
        print("Checkpoint keys:")
        for key in state_dict.keys():
            print(" ", key)

        # --------------------------------------------------------------
        # Handle both:
        #
        # model.network.0.weight
        # network.0.weight
        # 0.weight
        #
        # --------------------------------------------------------------

        cleaned = {}

        for key, value in state_dict.items():

            if key.startswith(
                "model.network."
            ):
                new_key = key[
                    len("model.network.") :
                ]

            elif key.startswith(
                "network."
            ):
                new_key = key[
                    len("network.") :
                ]

            else:
                new_key = key

            cleaned[new_key] = value

        # --------------------------------------------------------------
        # Infer Linear architecture from checkpoint.
        # --------------------------------------------------------------

        linear_weights = []

        for key in sorted(
            cleaned.keys()
        ):

            if key.endswith(
                ".weight"
            ):

                tensor = cleaned[key]

                if tensor.ndim == 2:

                    linear_weights.append(
                        (
                            key,
                            tensor.shape[1],
                            tensor.shape[0]
                        )
                    )

        if not linear_weights:
            raise RuntimeError(
                "Could not determine MLP architecture "
                "from checkpoint."
            )

        architecture = [
            linear_weights[0][1]
        ]

        for _, in_features, out_features in linear_weights:

            if architecture[-1] != in_features:

                raise RuntimeError(
                    "Checkpoint contains an inconsistent "
                    "MLP architecture."
                )

            architecture.append(
                out_features
            )

        self.architecture = architecture

        print()
        print("Detected architecture:")
        print(
            " -> ".join(
                str(x)
                for x in architecture
            )
        )

        # --------------------------------------------------------------
        # Build network dynamically.
        # --------------------------------------------------------------

        layers = []

        for i in range(
            len(architecture) - 1
        ):

            layers.append(
                nn.Linear(
                    architecture[i],
                    architecture[i + 1]
                )
            )

            if i < len(architecture) - 2:

                layers.append(
                    nn.ReLU()
                )

        self.network = nn.Sequential(
            *layers
        ).to(device)

        # --------------------------------------------------------------
        # Verify shapes before loading.
        # --------------------------------------------------------------

        expected = self.network.state_dict()

        if set(expected.keys()) != set(cleaned.keys()):

            missing = sorted(
                set(expected.keys())
                -
                set(cleaned.keys())
            )

            unexpected = sorted(
                set(cleaned.keys())
                -
                set(expected.keys())
            )

            raise RuntimeError(
                "\nCheckpoint mismatch after normalization.\n"
                f"Missing: {missing}\n"
                f"Unexpected: {unexpected}"
            )

        for key in expected:

            if (
                expected[key].shape
                !=
                cleaned[key].shape
            ):

                raise RuntimeError(
                    f"Shape mismatch for {key}: "
                    f"expected "
                    f"{tuple(expected[key].shape)}, "
                    f"got "
                    f"{tuple(cleaned[key].shape)}"
                )

        self.network.load_state_dict(
            cleaned,
            strict=True
        )

        self.network.eval()

        # --------------------------------------------------------------
        # Normalization
        # --------------------------------------------------------------

        if (
            "position_mean"
            not in checkpoint
        ):

            raise RuntimeError(
                "Checkpoint does not contain "
                "'position_mean'."
            )

        if (
            "position_std"
            not in checkpoint
        ):

            raise RuntimeError(
                "Checkpoint does not contain "
                "'position_std'."
            )

        if (
            "pressure_mean"
            not in checkpoint
        ):

            raise RuntimeError(
                "Checkpoint does not contain "
                "'pressure_mean'."
            )

        if (
            "pressure_std"
            not in checkpoint
        ):

            raise RuntimeError(
                "Checkpoint does not contain "
                "'pressure_std'."
            )

        self.position_mean = np.asarray(
            checkpoint[
                "position_mean"
            ],
            dtype=np.float32
        )

        self.position_std = np.asarray(
            checkpoint[
                "position_std"
            ],
            dtype=np.float32
        )

        self.pressure_mean = np.asarray(
            checkpoint[
                "pressure_mean"
            ],
            dtype=np.float32
        )

        self.pressure_std = np.asarray(
            checkpoint[
                "pressure_std"
            ],
            dtype=np.float32
        )

        self.position_std = np.maximum(
            self.position_std,
            1e-8
        )

        self.pressure_std = np.maximum(
            self.pressure_std,
            1e-8
        )

        print()
        print("Position mean:")
        print(self.position_mean)

        print()
        print("Position std:")
        print(self.position_std)

        print()
        print("Pressure mean:")
        print(self.pressure_mean)

        print()
        print("Pressure std:")
        print(self.pressure_std)

        print()
        print("Inverse MLP loaded successfully.")


    @torch.no_grad()
    def predict(
        self,
        positions
    ):

        positions = np.asarray(
            positions,
            dtype=np.float32
        )

        original_shape = positions.shape

        if positions.ndim == 1:

            positions = positions.reshape(
                1,
                -1
            )

        normalized = (
            positions
            -
            self.position_mean
        ) / self.position_std

        x = torch.from_numpy(
            normalized
        ).to(
            self.device
        )

        prediction = self.network(
            x
        ).cpu().numpy()

        pressure = (
            prediction
            *
            self.pressure_std
            +
            self.pressure_mean
        )

        pressure = np.clip(
            pressure,
            PRESSURE_MIN,
            PRESSURE_MAX
        )

        return pressure.reshape(
            original_shape
        )


# ======================================================================
# TRAJECTORY PARAMETER SAMPLING
# ======================================================================

def choose_difficulty():

    r = np.random.rand()

    if r < 0.15:
        return "easy"

    if r < 0.42:
        return "medium"

    if r < 0.78:
        return "hard"

    return "very_hard"


def difficulty_parameters(
    difficulty
):

    if difficulty == "easy":

        return {
            "scale": np.random.uniform(
                0.55,
                0.85
            ),
            "frequency": np.random.uniform(
                0.5,
                1.3
            ),
            "noise": np.random.uniform(
                0.0,
                0.003
            ),
            "points": np.random.randint(
                100,
                150
            )
        }

    if difficulty == "medium":

        return {
            "scale": np.random.uniform(
                0.65,
                1.00
            ),
            "frequency": np.random.uniform(
                0.7,
                2.0
            ),
            "noise": np.random.uniform(
                0.001,
                0.006
            ),
            "points": np.random.randint(
                120,
                180
            )
        }

    if difficulty == "hard":

        return {
            "scale": np.random.uniform(
                0.75,
                1.10
            ),
            "frequency": np.random.uniform(
                1.0,
                3.0
            ),
            "noise": np.random.uniform(
                0.002,
                0.010
            ),
            "points": np.random.randint(
                140,
                200
            )
        }

    return {
        "scale": np.random.uniform(
            0.85,
            1.20
        ),
        "frequency": np.random.uniform(
            1.5,
            4.0
        ),
        "noise": np.random.uniform(
            0.003,
            0.015
        ),
        "points": np.random.randint(
            160,
            MAX_POINTS + 1
        )
    }


# ======================================================================
# COMMON 3D CURVE TRANSFORM
# ======================================================================

def transform_curve(
    x,
    y,
    difficulty,
    z_base=None
):

    params = difficulty_parameters(
        difficulty
    )

    scale = params["scale"]

    frequency = params[
        "frequency"
    ]

    noise_level = params[
        "noise"
    ]

    n = len(x)

    # --------------------------------------------------------------
    # Random center
    # --------------------------------------------------------------

    cx = np.random.uniform(
        -0.08,
        0.08
    )

    cy = np.random.uniform(
        -0.07,
        0.07
    )

    if z_base is None:

        z_base = np.random.uniform(
            0.075,
            0.165
        )

    # --------------------------------------------------------------
    # Random orientation in XY.
    # --------------------------------------------------------------

    angle = np.random.uniform(
        0,
        2 * np.pi
    )

    ca = np.cos(angle)
    sa = np.sin(angle)

    xr = (
        ca * x
        -
        sa * y
    )

    yr = (
        sa * x
        +
        ca * y
    )

    # --------------------------------------------------------------
    # Independent axis scaling.
    # --------------------------------------------------------------

    sx = np.random.uniform(
        0.020,
        0.115
    ) * scale

    sy = np.random.uniform(
        0.020,
        0.105
    ) * scale

    x_final = (
        cx
        +
        sx * xr
    )

    y_final = (
        cy
        +
        sy * yr
    )

    # --------------------------------------------------------------
    # 3D variation.
    # --------------------------------------------------------------

    t = np.linspace(
        0.0,
        1.0,
        n
    )

    z_amp = np.random.uniform(
        0.003,
        0.035
    ) * scale

    z_freq = np.random.uniform(
        0.5,
        2.5
    ) * frequency

    z_phase = np.random.uniform(
        0,
        2 * np.pi
    )

    z_final = (
        z_base
        +
        z_amp
        *
        np.sin(
            2 * np.pi
            * z_freq
            * t
            +
            z_phase
        )
    )

    # --------------------------------------------------------------
    # Coupled nonlinear 3D deformation.
    # --------------------------------------------------------------

    coupling = np.random.uniform(
        -0.25,
        0.25
    )

    z_final += (
        coupling
        *
        x_final
        *
        y_final
    )

    # --------------------------------------------------------------
    # Smooth perturbation.
    # --------------------------------------------------------------

    if noise_level > 0:

        noise = np.random.normal(
            0.0,
            noise_level,
            size=(n, 2)
        )

        kernel = np.ones(
            7
        ) / 7.0

        noise[:, 0] = np.convolve(
            noise[:, 0],
            kernel,
            mode="same"
        )

        noise[:, 1] = np.convolve(
            noise[:, 1],
            kernel,
            mode="same"
        )

        x_final += noise[:, 0]
        y_final += noise[:, 1]

    return np.column_stack(
        [
            x_final,
            y_final,
            z_final
        ]
    ).astype(
        np.float32
    )


# ======================================================================
# TRAJECTORY GENERATORS
# ======================================================================

def generate_circle(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    radius = np.random.uniform(
        0.55,
        1.0
    )

    x = (
        radius
        * np.cos(t)
    )

    y = (
        radius
        * np.sin(t)
    )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_ellipse(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    a = np.random.uniform(
        0.6,
        1.15
    )

    b = np.random.uniform(
        0.35,
        1.0
    )

    x = a * np.cos(t)
    y = b * np.sin(t)

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_figure8(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    x = np.sin(t)

    y = (
        np.sin(t)
        *
        np.cos(t)
    )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_parabola(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    u = np.linspace(
        -1.0,
        1.0,
        n
    )

    curvature = np.random.uniform(
        0.45,
        1.25
    )

    x = u

    y = (
        curvature
        *
        u ** 2
    )

    # Add asymmetric deformation for harder samples.

    if difficulty in (
        "hard",
        "very_hard"
    ):

        y += (
            np.random.uniform(
                -0.25,
                0.25
            )
            * u ** 3
        )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_heart(
    difficulty
):

    """
    Parametric heart:

        x = 16 sin^3(t)

        y =
            13 cos(t)
            - 5 cos(2t)
            - 2 cos(3t)
            - cos(4t)

    The curve is normalized before being mapped into the robot
    workspace.
    """

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    x = (
        16
        *
        np.sin(t) ** 3
    )

    y = (
        13 * np.cos(t)
        -
        5 * np.cos(2 * t)
        -
        2 * np.cos(3 * t)
        -
        np.cos(4 * t)
    )

    x /= np.max(
        np.abs(x)
    )

    y /= np.max(
        np.abs(y)
    )

    # Make heart slightly asymmetric in difficult cases.

    if difficulty in (
        "hard",
        "very_hard"
    ):

        y += (
            np.random.uniform(
                -0.08,
                0.08
            )
            * np.sin(2 * t)
        )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_spiral(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    turns = np.random.uniform(
        1.2,
        3.5
    )

    t = np.linspace(
        0,
        2 * np.pi * turns,
        n
    )

    r0 = np.random.uniform(
        0.15,
        0.40
    )

    r1 = np.random.uniform(
        0.75,
        1.10
    )

    r = np.linspace(
        r0,
        r1,
        n
    )

    x = r * np.cos(t)
    y = r * np.sin(t)

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_lissajous(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    a = np.random.randint(
        1,
        6
    )

    b = np.random.randint(
        1,
        6
    )

    delta = np.random.uniform(
        0,
        2 * np.pi
    )

    x = np.sin(
        a * t + delta
    )

    y = np.sin(
        b * t
    )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_wave(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    u = np.linspace(
        -1,
        1,
        n
    )

    frequency = np.random.uniform(
        1.0,
        5.0
    )

    amplitude = np.random.uniform(
        0.35,
        1.0
    )

    phase = np.random.uniform(
        0,
        2 * np.pi
    )

    x = u

    y = (
        amplitude
        *
        np.sin(
            frequency
            * np.pi
            * u
            +
            phase
        )
    )

    if difficulty in (
        "hard",
        "very_hard"
    ):

        y += (
            0.25
            *
            np.sin(
                (frequency + 1.5)
                * np.pi
                * u
            )
        )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_multi_frequency(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    x = np.zeros(n)
    y = np.zeros(n)

    components = np.random.randint(
        2,
        5
    )

    for _ in range(
        components
    ):

        freq = np.random.uniform(
            0.5,
            5.0
        )

        phase = np.random.uniform(
            0,
            2 * np.pi
        )

        amp = np.random.uniform(
            0.10,
            0.55
        )

        x += (
            amp
            *
            np.sin(
                freq * t
                +
                phase
            )
        )

        y += (
            amp
            *
            np.cos(
                (freq + np.random.uniform(
                    -0.7,
                    0.7
                ))
                *
                t
                +
                phase
            )
        )

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_random_smooth(
    difficulty
):

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        1,
        n
    )

    harmonics = np.random.randint(
        3,
        8
    )

    x = np.zeros(n)
    y = np.zeros(n)

    for k in range(
        1,
        harmonics + 1
    ):

        ax = np.random.uniform(
            -1.0,
            1.0
        ) / k

        ay = np.random.uniform(
            -1.0,
            1.0
        ) / k

        px = np.random.uniform(
            0,
            2 * np.pi
        )

        py = np.random.uniform(
            0,
            2 * np.pi
        )

        x += (
            ax
            *
            np.sin(
                2 * np.pi
                * k
                * t
                +
                px
            )
        )

        y += (
            ay
            *
            np.cos(
                2 * np.pi
                * k
                * t
                +
                py
            )
        )

    max_abs = max(
        np.max(np.abs(x)),
        np.max(np.abs(y)),
        1e-8
    )

    x /= max_abs
    y /= max_abs

    return transform_curve(
        x,
        y,
        difficulty
    )


def generate_hybrid(
    difficulty
):

    """
    Combines several curve bases.

    This is important for generalization because the VLA should
    not learn only canonical mathematical shapes.
    """

    p = difficulty_parameters(
        difficulty
    )

    n = p["points"]

    t = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    mode1 = random.choice(
        [
            "circle",
            "figure8",
            "lissajous",
            "wave"
        ]
    )

    mode2 = random.choice(
        [
            "wave",
            "multi",
            "spiral"
        ]
    )

    if mode1 == "circle":

        x1 = np.cos(t)
        y1 = np.sin(t)

    elif mode1 == "figure8":

        x1 = np.sin(t)
        y1 = np.sin(t) * np.cos(t)

    elif mode1 == "lissajous":

        a = np.random.randint(
            1,
            4
        )

        b = np.random.randint(
            1,
            5
        )

        x1 = np.sin(a * t)
        y1 = np.sin(
            b * t
            +
            np.random.uniform(
                0,
                2 * np.pi
            )
        )

    else:

        x1 = t / np.pi - 1.0

        y1 = np.sin(
            np.random.uniform(
                1.0,
                4.0
            )
            * t
        )

    if mode2 == "wave":

        x2 = np.cos(
            2.0 * t
        )

        y2 = np.sin(
            3.0 * t
        )

    elif mode2 == "multi":

        x2 = (
            0.6 * np.sin(2*t)
            +
            0.3 * np.sin(5*t)
        )

        y2 = (
            0.6 * np.cos(3*t)
            +
            0.2 * np.sin(7*t)
        )

    else:

        r = np.linspace(
            0.1,
            0.7,
            n
        )

        x2 = r * np.cos(
            2*t
        )

        y2 = r * np.sin(
            2*t
        )

    alpha = np.random.uniform(
        0.15,
        0.45
    )

    x = (
        (1.0 - alpha)
        * x1
        +
        alpha
        * x2
    )

    y = (
        (1.0 - alpha)
        * y1
        +
        alpha
        * y2
    )

    return transform_curve(
        x,
        y,
        difficulty
    )


# ======================================================================
# GENERATOR DISPATCH
# ======================================================================

TRAJECTORY_GENERATORS = {

    "circle":
        generate_circle,

    "ellipse":
        generate_ellipse,

    "figure8":
        generate_figure8,

    "parabola":
        generate_parabola,

    "heart":
        generate_heart,

    "spiral":
        generate_spiral,

    "lissajous":
        generate_lissajous,

    "wave":
        generate_wave,

    "multi_frequency":
        generate_multi_frequency,

    "random_smooth":
        generate_random_smooth,

    "hybrid":
        generate_hybrid,
}


# ======================================================================
# TRAJECTORY DISTRIBUTION
# ======================================================================

TRAJECTORY_WEIGHTS = {

    "circle": 0.12,

    "ellipse": 0.09,

    "figure8": 0.12,

    "parabola": 0.08,

    "heart": 0.10,

    "spiral": 0.09,

    "lissajous": 0.08,

    "wave": 0.08,

    "multi_frequency": 0.08,

    "random_smooth": 0.09,

    "hybrid": 0.07,
}


def choose_trajectory_type():

    names = list(
        TRAJECTORY_WEIGHTS.keys()
    )

    weights = np.array(
        [
            TRAJECTORY_WEIGHTS[x]
            for x in names
        ],
        dtype=np.float64
    )

    weights /= weights.sum()

    return np.random.choice(
        names,
        p=weights
    )


# ======================================================================
# RESAMPLING
# ======================================================================

def resample_by_arclength(
    points,
    n_points
):

    if len(points) < 3:
        return points

    distances = np.linalg.norm(
        np.diff(
            points,
            axis=0
        ),
        axis=1
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(distances)
        ]
    )

    total = cumulative[-1]

    if total < 1e-10:
        return points

    desired = np.linspace(
        0.0,
        total,
        n_points
    )

    result = np.zeros(
        (
            n_points,
            3
        ),
        dtype=np.float32
    )

    for axis in range(3):

        result[:, axis] = np.interp(
            desired,
            cumulative,
            points[:, axis]
        )

    return result


# ======================================================================
# VALIDITY CHECKS
# ======================================================================

def validate_trajectory(
    positions
):

    if positions.ndim != 2:
        return False

    if positions.shape[1] != 3:
        return False

    if not np.all(
        np.isfinite(positions)
    ):
        return False

    if np.any(
        positions < WORKSPACE_MIN
    ):
        return False

    if np.any(
        positions > WORKSPACE_MAX
    ):
        return False

    length = path_length(
        positions
    )

    if length < MIN_PATH_LENGTH:
        return False

    if length > MAX_PATH_LENGTH:
        return False

    spacing = point_spacing(
        positions
    )

    if len(spacing) > 0:

        if np.max(
            spacing
        ) > MAX_POSITION_STEP:

            return False

    return True


def validate_pressure(
    pressure
):

    if not np.all(
        np.isfinite(
            pressure
        )
    ):
        return False

    if np.any(
        pressure < PRESSURE_MIN - 1e-6
    ):
        return False

    if np.any(
        pressure > PRESSURE_MAX + 1e-6
    ):
        return False

    if len(pressure) > 1:

        steps = np.linalg.norm(
            np.diff(
                pressure,
                axis=0
            ),
            axis=1
        )

        if np.max(
            steps
        ) > MAX_PRESSURE_STEP:

            return False

    return True


# ======================================================================
# DIFFICULTY REFINEMENT
# ======================================================================

def calculate_difficulty(
    positions,
    pressure
):

    length = path_length(
        positions
    )

    spacing = point_spacing(
        positions
    )

    if len(spacing) == 0:
        spacing_cv = 0.0

    else:

        mean_spacing = np.mean(
            spacing
        )

        if mean_spacing < 1e-8:

            spacing_cv = 0.0

        else:

            spacing_cv = (
                np.std(spacing)
                /
                mean_spacing
            )

    curvature_score = 0.0

    if len(positions) >= 3:

        d1 = np.diff(
            positions,
            axis=0
        )

        d1_norm = np.linalg.norm(
            d1,
            axis=1,
            keepdims=True
        )

        d1_norm = np.maximum(
            d1_norm,
            1e-8
        )

        directions = (
            d1
            /
            d1_norm
        )

        changes = np.linalg.norm(
            np.diff(
                directions,
                axis=0
            ),
            axis=1
        )

        curvature_score = float(
            np.mean(changes)
        )

    pressure_variation = (
        float(
            np.mean(
                np.std(
                    pressure,
                    axis=0
                )
            )
        )
    )

    complexity = (
        0.45
        * min(
            curvature_score * 10.0,
            1.0
        )
        +
        0.25
        * min(
            spacing_cv,
            1.0
        )
        +
        0.30
        * min(
            pressure_variation / 0.8,
            1.0
        )
    )

    if complexity < 0.22:
        return "easy"

    if complexity < 0.47:
        return "medium"

    if complexity < 0.72:
        return "hard"

    return "very_hard"


# ======================================================================
# SINGLE SAMPLE
# ======================================================================

def generate_single_trajectory(
    inverse_model
):

    trajectory_type = (
        choose_trajectory_type()
    )

    requested_difficulty = (
        choose_difficulty()
    )

    generator = TRAJECTORY_GENERATORS[
        trajectory_type
    ]

    positions = generator(
        requested_difficulty
    )

    # --------------------------------------------------------------
    # Resample to approximately uniform spatial spacing.
    # This is important for continuous trajectory learning.
    # --------------------------------------------------------------

    n_points = len(
        positions
    )

    positions = resample_by_arclength(
        positions,
        n_points
    )

    # --------------------------------------------------------------
    # Validate position trajectory.
    # --------------------------------------------------------------

    if not validate_trajectory(
        positions
    ):

        return None

    # --------------------------------------------------------------
    # Convert every point through inverse MLP.
    # --------------------------------------------------------------

    pressures = inverse_model.predict(
        positions
    )

    pressures = np.asarray(
        pressures,
        dtype=np.float32
    )

    # --------------------------------------------------------------
    # Pressure validity.
    # --------------------------------------------------------------

    if not validate_pressure(
        pressures
    ):

        return None

    # --------------------------------------------------------------
    # Final difficulty based on actual geometry and pressure.
    # --------------------------------------------------------------

    actual_difficulty = calculate_difficulty(
        positions,
        pressures
    )

    return {
        "trajectory_type":
            trajectory_type,

        "difficulty":
            actual_difficulty,

        "positions":
            positions.astype(
                np.float32
            ),

        "pressures":
            pressures.astype(
                np.float32
            ),

        "path_length":
            path_length(
                positions
            ),

        "mean_pressure_step":
            float(
                np.mean(
                    np.linalg.norm(
                        np.diff(
                            pressures,
                            axis=0
                        ),
                        axis=1
                    )
                )
            ),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("SOFT ROBOT VLA CURVED TRAJECTORY DATASET")
    print("=" * 70)

    print()
    print("Target trajectories:")
    print(NUM_TRAJECTORIES)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print("Trajectory types:")

    for name in TRAJECTORY_GENERATORS:
        print(
            "  -",
            name
        )

    print()
    print("=" * 70)
    print("INITIALIZATION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    device = torch.device(
        "cpu"
    )

    print()
    print("Device:")
    print(device)

    # --------------------------------------------------------------
    # Inverse MLP
    # --------------------------------------------------------------

    inverse_model = InverseModel(
        MODEL_PATH,
        device
    )

    # --------------------------------------------------------------
    # Storage
    #
    # Since trajectories have variable length, object arrays are used
    # inside the NPZ.
    # --------------------------------------------------------------

    positions_list = []
    pressures_list = []

    trajectory_types = []
    difficulties = []

    path_lengths = []
    pressure_steps = []

    hashes = set()

    accepted = 0
    rejected = 0

    type_counts = {
        key: 0
        for key in TRAJECTORY_GENERATORS
    }

    difficulty_counts = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "very_hard": 0,
    }

    print()
    print("=" * 70)
    print("GENERATING")
    print("=" * 70)

    generation_start = time.time()

    last_print = generation_start

    attempts = 0

    while accepted < NUM_TRAJECTORIES:

        attempts += 1

        sample = generate_single_trajectory(
            inverse_model
        )

        if sample is None:

            rejected += 1

        else:

            h = trajectory_hash(
                sample["positions"]
            )

            if h in hashes:

                rejected += 1

            else:

                hashes.add(h)

                positions_list.append(
                    sample["positions"]
                )

                pressures_list.append(
                    sample["pressures"]
                )

                trajectory_types.append(
                    sample["trajectory_type"]
                )

                difficulties.append(
                    sample["difficulty"]
                )

                path_lengths.append(
                    sample["path_length"]
                )

                pressure_steps.append(
                    sample[
                        "mean_pressure_step"
                    ]
                )

                type_counts[
                    sample["trajectory_type"]
                ] += 1

                difficulty_counts[
                    sample["difficulty"]
                ] += 1

                accepted += 1

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------

        if (
            accepted % 500 == 0
            and accepted > 0
        ):

            now = time.time()

            elapsed = (
                now
                -
                generation_start
            )

            rate = (
                accepted / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"[{accepted:7d} / "
                f"{NUM_TRAJECTORIES:7d}] "
                f"accepted={accepted:7d} "
                f"rejected={rejected:7d} "
                f"rate={rate:8.1f}/s"
            )

    generation_time = (
        time.time()
        -
        generation_start
    )

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Generation time: "
        f"{generation_time:.2f} s"
    )

    print()
    print(
        f"Attempts: {attempts}"
    )

    print(
        f"Accepted: {accepted}"
    )

    print(
        f"Rejected: {rejected}"
    )

    # ==================================================================
    # SPLITS
    # ==================================================================

    indices = np.arange(
        accepted
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(
        indices
    )

    train_end = int(
        TRAIN_RATIO
        * accepted
    )

    val_end = (
        train_end
        +
        int(
            VAL_RATIO
            * accepted
        )
    )

    split = np.full(
        accepted,
        "test",
        dtype="<U10"
    )

    split[
        indices[:train_end]
    ] = "train"

    split[
        indices[train_end:val_end]
    ] = "validation"

    # ==================================================================
    # SAVE NPZ
    # ==================================================================

    print()
    print("=" * 70)
    print("SAVING NPZ")
    print("=" * 70)

    positions_object = np.empty(
        accepted,
        dtype=object
    )

    pressures_object = np.empty(
        accepted,
        dtype=object
    )

    for i in range(
        accepted
    ):

        positions_object[i] = (
            positions_list[i]
        )

        pressures_object[i] = (
            pressures_list[i]
        )

    np.savez_compressed(
        NPZ_PATH,

        positions=
            positions_object,

        pressures=
            pressures_object,

        trajectory_type=
            np.asarray(
                trajectory_types
            ),

        difficulty=
            np.asarray(
                difficulties
            ),

        path_length=
            np.asarray(
                path_lengths,
                dtype=np.float32
            ),

        mean_pressure_step=
            np.asarray(
                pressure_steps,
                dtype=np.float32
            ),

        split=
            split,
    )

    # ==================================================================
    # MANIFEST
    # ==================================================================

    print()
    print("=" * 70)
    print("SAVING MANIFEST")
    print("=" * 70)

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for i in range(
            accepted
        ):

            entry = {

                "index":
                    i,

                "trajectory_type":
                    trajectory_types[i],

                "difficulty":
                    difficulties[i],

                "split":
                    str(split[i]),

                "num_points":
                    int(
                        len(
                            positions_list[i]
                        )
                    ),

                "path_length":
                    float(
                        path_lengths[i]
                    ),

                "mean_pressure_step":
                    float(
                        pressure_steps[i]
                    ),
            }

            f.write(
                json.dumps(
                    entry
                )
                +
                "\n"
            )

    # ==================================================================
    # METADATA
    # ==================================================================

    print()
    print("=" * 70)
    print("SAVING METADATA")
    print("=" * 70)

    all_positions = np.concatenate(
        positions_list,
        axis=0
    )

    all_pressures = np.concatenate(
        pressures_list,
        axis=0
    )

    metadata = {

        "dataset_name":
            "soft_robot_vla_curved_trajectories",

        "version":
            "2.0",

        "num_trajectories":
            accepted,

        "seed":
            SEED,

        "trajectory_types":
            list(
                TRAJECTORY_GENERATORS.keys()
            ),

        "trajectory_distribution":
            type_counts,

        "difficulty_distribution":
            difficulty_counts,

        "splits": {

            "train":
                int(
                    np.sum(
                        split == "train"
                    )
                ),

            "validation":
                int(
                    np.sum(
                        split == "validation"
                    )
                ),

            "test":
                int(
                    np.sum(
                        split == "test"
                    )
                ),
        },

        "workspace": {

            "min":
                WORKSPACE_MIN.tolist(),

            "max":
                WORKSPACE_MAX.tolist(),
        },

        "pressure_limits": [

            PRESSURE_MIN,

            PRESSURE_MAX
        ],

        "position_statistics": {

            "min":
                np.min(
                    all_positions,
                    axis=0
                ).tolist(),

            "max":
                np.max(
                    all_positions,
                    axis=0
                ).tolist(),

            "mean":
                np.mean(
                    all_positions,
                    axis=0
                ).tolist(),

            "std":
                np.std(
                    all_positions,
                    axis=0
                ).tolist(),
        },

        "pressure_statistics": {

            "min":
                np.min(
                    all_pressures,
                    axis=0
                ).tolist(),

            "max":
                np.max(
                    all_pressures,
                    axis=0
                ).tolist(),

            "mean":
                np.mean(
                    all_pressures,
                    axis=0
                ).tolist(),

            "std":
                np.std(
                    all_pressures,
                    axis=0
                ).tolist(),
        },

        "path_statistics": {

            "mean":
                float(
                    np.mean(
                        path_lengths
                    )
                ),

            "std":
                float(
                    np.std(
                        path_lengths
                    )
                ),

            "minimum":
                float(
                    np.min(
                        path_lengths
                    )
                ),

            "maximum":
                float(
                    np.max(
                        path_lengths
                    )
                ),
        },

        "pressure_step_statistics": {

            "mean":
                float(
                    np.mean(
                        pressure_steps
                    )
                ),

            "std":
                float(
                    np.std(
                        pressure_steps
                    )
                ),

            "maximum":
                float(
                    np.max(
                        pressure_steps
                    )
                ),
        },

        "inverse_model": {

            "checkpoint":
                os.path.relpath(
                    MODEL_PATH,
                    ROOT
                ),

            "architecture":
                inverse_model.architecture,

            "input":
                "XYZ target position",

            "output":
                "3 actuator pressures",

            "pressure_units":
                "bar",
        },

        "generation": {

            "min_points":
                MIN_POINTS,

            "max_points":
                MAX_POINTS,

            "min_path_length":
                MIN_PATH_LENGTH,

            "max_path_length":
                MAX_PATH_LENGTH,

            "max_position_step":
                MAX_POSITION_STEP,

            "max_pressure_step":
                MAX_PRESSURE_STEP,
        },

        "generation_time_seconds":
            generation_time,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # ==================================================================
    # FINAL REPORT
    # ==================================================================

    total_time = (
        time.time()
        -
        start_time
    )

    print()
    print("=" * 70)
    print("FINAL DATASET STATISTICS")
    print("=" * 70)

    print()
    print(
        f"Trajectories: {accepted}"
    )

    print()
    print("Trajectory distribution:")

    for key in (
        TRAJECTORY_GENERATORS
    ):

        count = type_counts[key]

        percentage = (
            100.0
            * count
            /
            accepted
        )

        print(
            f"{key:20s}: "
            f"{count:7d} "
            f"({percentage:5.1f}%)"
        )

    print()
    print("Difficulty distribution:")

    for key in (
        [
            "easy",
            "medium",
            "hard",
            "very_hard"
        ]
    ):

        count = difficulty_counts[
            key
        ]

        percentage = (
            100.0
            * count
            /
            accepted
        )

        print(
            f"{key:20s}: "
            f"{count:7d} "
            f"({percentage:5.1f}%)"
        )

    print()
    print("Position minimum:")
    print(
        np.min(
            all_positions,
            axis=0
        )
    )

    print()
    print("Position maximum:")
    print(
        np.max(
            all_positions,
            axis=0
        )
    )

    print()
    print("Pressure minimum:")
    print(
        np.min(
            all_pressures,
            axis=0
        )
    )

    print()
    print("Pressure maximum:")
    print(
        np.max(
            all_pressures,
            axis=0
        )
    )

    print()
    print(
        "Mean path length:"
    )

    print(
        np.mean(
            path_lengths
        )
    )

    print()
    print(
        "Mean pressure step:"
    )

    print(
        np.mean(
            pressure_steps
        )
    )

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print()
    print(
        "NPZ:"
    )
    print(
        NPZ_PATH
    )

    print()
    print(
        "Manifest:"
    )
    print(
        MANIFEST_PATH
    )

    print()
    print(
        "Metadata:"
    )
    print(
        METADATA_PATH
    )

    print()
    print(
        f"Total generation time: "
        f"{total_time:.2f} s"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()