#!/usr/bin/env python3
"""
SOFT ROBOT — VLA + TEMPORAL INVERSE MLP — V9 ZERO-VIBRATION PROFILE
HIGH-ACCURACY 75 mm HEART TRACKING

IMPORTANT
---------
This version fixes the VLA checkpoint-loading problem from the previous file.

The supplied best_model.pt is the ORIGINAL/simple SoftRobotVLA checkpoint:
    image -> VisualEncoder
    instruction -> InstructionEncoder
    state [x,y,z,vx,vy,vz] -> StateEncoder
    target_error [dx,dy,dz] -> TargetEncoder
    fusion -> 4 residual blocks -> action_head

The checkpoint is loaded against that exact architecture.

CONTROL ARCHITECTURE
--------------------
                  desired circle point
                           |
                           v
                 +-------------------+
camera --------> |       VLA         |
state ----------> |  pressure delta  |
instruction ----> |    residual      |
target error ---> +---------+---------+
                           |
                           v
                  +----------------+
actual state ---->| Temporal MLP   |
target ---------->| absolute P1-3  |
previous P ------>|                |
                  +-------+--------+
                          |
                          v
                 bounded local DLS
                 geometry correction
                          |
                          v
                    pressure P1-3
                          |
                          v
                    Kinematics
                          |
                          v
                    15 actuators
                          |
                          v
                       MuJoCo
                          |
                          v
                         EE

The temporal inverse MLP remains the primary controller.
The VLA is used as a bounded residual pressure-delta signal.
The local DLS stage compensates for the known pressure/model mismatch.

No PID and no integral accumulator.
Physics timing is NOT accelerated. Only wall-clock/rendering is accelerated.
"""

import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import mujoco.viewer

from control.kinematics import Kinematics


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "model" / "scene.xml"
VLA_PATH = ROOT / "soft_robot_vla" / "checkpoints" / "best_model.pt"

MLP_CANDIDATES = [
    ROOT / "models" / "temporal_inverse_mlp.pt",
    ROOT / "model" / "temporal_inverse_mlp.pt",
    ROOT / "models" / "inverse_mlp.pt",
    ROOT / "model" / "inverse_mlp.pt",
]

DESKTOP_ROOT = Path.home() / "Desktop" / "soft_robot_runs"
DESKTOP_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DEVICE
# ============================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================================
# GLOBAL CONFIG
# ============================================================================

IMAGE_SIZE = 224
MAX_INSTRUCTION_LENGTH = 32

# This is the instruction used by the trained VLA family.
INSTRUCTION = "Move the soft robot along the J trajectory."

P_MIN = 0.0
P_MAX = 3.0

INITIAL_PRESSURE = np.array([1.5, 1.5, 1.5], dtype=np.float64)

# MuJoCo physics step.
CONTROL_DT = 0.01

# Learned-controller update.
CONTROL_UPDATE_DT = 0.02

# Pressure response.
PRESSURE_FILTER_TAU = 0.045
MAX_PRESSURE_RATE = 3.50
PRESSURE_DEADBAND = 0.00015

# ============================================================================
# HEART
# ============================================================================

FIGURE8_SCALE = 0.035
FIGURE8_HEIGHT_ABOVE_TIP = 0.025

APPROACH_TIME = 6.0
FIGURE8_TIME = 45.0
FINAL_HOLD = 1.0

# Predict a little ahead so the soft robot does not permanently lag.
LOOKAHEAD_TIME = 0.075
PREDICTION_HORIZON = 0.050

# ============================================================================
# VLA RESIDUAL
# ============================================================================

# VLA checkpoint outputs normalized pressure delta.
# The original training convention was:
#     normalized_action = pressure_change / 0.015
# Therefore:
#     pressure_change = normalized_action * 0.015
MAX_VLA_PRESSURE_DELTA = 0.015

# Keep the J-trained VLA as a small residual because the benchmark trajectory
# here is a circle, not the J trajectory used to train this checkpoint.
VLA_RESIDUAL_WEIGHT = 0.025

# Run VLA at this interval. It is intentionally slower than the MLP loop.
VLA_UPDATE_DT = 0.05

# ============================================================================
# LOCAL GEOMETRY CORRECTION
# ============================================================================

JACOBIAN_PRESSURE_STEP = 0.020
MAX_PRESSURE_CORRECTION = 0.48
DLS_DAMPING = 2.5e-4

POSITION_WEIGHT = 1.0
RADIAL_WEIGHT = 8.0
TANGENTIAL_WEIGHT = 0.75
Z_WEIGHT = 1.5
PRESSURE_CHANGE_WEIGHT = 0.02

MAX_POSITION_CORRECTION = 0.022

# ============================================================================
# FAST ANIMATION
# ============================================================================

ANIMATION_SPEEDUP = 15.0

# Render only every N physics samples. Physics/logging are unchanged.
RENDER_EVERY_N_STEPS = 2

# ============================================================================
# STABILITY
# ============================================================================

JOINT_DAMPING_SCALE = 2.5
MIN_JOINT_DAMPING = 0.012

ACTUATOR_DAMPING = 0.030
ACTUATOR_ARMATURE = 0.003
ACTUATOR_DAMPING_RATIO = 1.25


# ============================================================================
# EXACT VLA ARCHITECTURE USED BY best_model.pt
# ============================================================================

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class VisualEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, image):
        return self.projection(self.network(image))


class InstructionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        output_dim=256,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, tokens, padding_mask=None):
        x = self.embedding(tokens)

        if padding_mask is None:
            x = x.mean(dim=1)
        else:
            mask = (~padding_mask).unsqueeze(-1)
            x = x * mask
            denominator = mask.sum(dim=1).clamp(min=1)
            x = x.sum(dim=1) / denominator

        return self.projection(x)


class StateEncoder(nn.Module):
    def __init__(self, input_dim=6, output_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, state):
        return self.network(state)


class TargetEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, target_error):
        return self.network(target_error)


class SoftRobotVLA(nn.Module):
    def __init__(
        self,
        vocab_size,
        action_dim=3,
        state_dim=6,
        hidden_dim=512,
        language_dim=256,
        visual_dim=256,
        state_feature_dim=256,
        target_feature_dim=128,
        num_residual_blocks=4,
    ):
        super().__init__()

        self.visual_encoder = VisualEncoder(output_dim=visual_dim)

        self.language_encoder = InstructionEncoder(
            vocab_size=vocab_size,
            embedding_dim=128,
            output_dim=language_dim,
        )

        self.state_encoder = StateEncoder(
            input_dim=state_dim,
            output_dim=state_feature_dim,
        )

        self.target_encoder = TargetEncoder(
            input_dim=3,
            output_dim=target_feature_dim,
        )

        fusion_dim = (
            visual_dim
            + language_dim
            + state_feature_dim
            + target_feature_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim)
                for _ in range(num_residual_blocks)
            ]
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        image,
        instruction_tokens,
        state,
        target_error,
        instruction_padding_mask=None,
    ):
        visual_features = self.visual_encoder(image)

        language_features = self.language_encoder(
            instruction_tokens,
            instruction_padding_mask,
        )

        state_features = self.state_encoder(state)

        target_features = self.target_encoder(target_error)

        x = torch.cat(
            [
                visual_features,
                language_features,
                state_features,
                target_features,
            ],
            dim=1,
        )

        x = self.fusion(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.action_head(x)


# ============================================================================
# VLA LOADER
# ============================================================================

def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def tokenize_instruction(text, vocab):
    words = text.lower().split()

    tokens = [
        vocab.get(word, vocab.get("<UNK>", 1))
        for word in words
    ]

    tokens = tokens[:MAX_INSTRUCTION_LENGTH]

    while len(tokens) < MAX_INSTRUCTION_LENGTH:
        tokens.append(vocab.get("<PAD>", 0))

    return torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE,
    )


def load_vla():
    if not VLA_PATH.exists():
        raise FileNotFoundError(
            f"VLA checkpoint not found:\n{VLA_PATH}"
        )

    checkpoint = torch.load(
        VLA_PATH,
        map_location="cpu",
        weights_only=False,
    )

    vocab = checkpoint.get(
        "vocab",
        checkpoint.get("vocabulary"),
    )

    if vocab is None:
        raise RuntimeError(
            "best_model.pt does not contain vocab/vocabulary."
        )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state is None:
        raise RuntimeError(
            "best_model.pt does not contain model_state_dict/state_dict."
        )

    state = clean_state_dict(state)

    # The actual checkpoint proves the intended architecture:
    # state_encoder / target_encoder / fusion / action_head /
    # visual_encoder.network / language_encoder.embedding / residual_blocks.
    model = SoftRobotVLA(
        vocab_size=len(vocab),
        action_dim=3,
        state_dim=6,
        hidden_dim=512,
        language_dim=256,
        visual_dim=256,
        state_feature_dim=256,
        target_feature_dim=128,
        num_residual_blocks=4,
    )

    expected = model.state_dict()

    # Fail with a useful message rather than PyTorch's huge wall of keys.
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))

    shape_errors = []
    for key in expected.keys():
        if key in state and tuple(expected[key].shape) != tuple(state[key].shape):
            shape_errors.append(
                (
                    key,
                    tuple(state[key].shape),
                    tuple(expected[key].shape),
                )
            )

    if missing or unexpected or shape_errors:
        message = [
            "VLA checkpoint does not match the exact checkpoint architecture.",
            "",
            f"Missing keys: {len(missing)}",
            f"Unexpected keys: {len(unexpected)}",
            f"Shape errors: {len(shape_errors)}",
        ]

        if shape_errors:
            message.append("")
            message.append("First shape errors:")
            for key, got, wanted in shape_errors[:10]:
                message.append(
                    f"  {key}: checkpoint={got}, model={wanted}"
                )

        raise RuntimeError("\n".join(message))

    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    print("VLA loaded successfully.")
    print("  Architecture : original 6-state VLA")
    print("  State        : [x y z vx vy vz]")
    print("  Target input  : [dx dy dz]")
    print("  Action        : normalized pressure delta")
    print("  Vocab size    :", len(vocab))
    print("  Device        :", DEVICE)

    return model, vocab


class VLAResidualController:
    def __init__(self, model, vocab):
        self.model = model
        self.tokens = tokenize_instruction(
            INSTRUCTION,
            vocab,
        )

        pad_id = vocab.get("<PAD>", 0)

        self.padding_mask = (
            self.tokens == pad_id
        )

    @torch.inference_mode()
    def predict(self, image, current, target, velocity):
        state = np.concatenate(
            [
                current,
                velocity,
            ]
        ).astype(np.float32)

        target_error = (
            target - current
        ).astype(np.float32)

        image_tensor = torch.as_tensor(
            image,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, 6)

        error_tensor = torch.as_tensor(
            target_error,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, 3)

        normalized_delta = self.model(
            image_tensor,
            self.tokens,
            state_tensor,
            error_tensor,
            self.padding_mask,
        )

        delta = (
            normalized_delta
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        delta = np.clip(
            delta,
            -1.0,
            1.0,
        )

        return (
            delta
            * MAX_VLA_PRESSURE_DELTA
        ).astype(np.float64)


# ============================================================================
# TEMPORAL INVERSE MLP
# ============================================================================

EXPECTED_FEATURE_NAMES = [
    "current_x",
    "current_y",
    "current_z",
    "target_x",
    "target_y",
    "target_z",
    "error_x",
    "error_y",
    "error_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "previous_pressure_1",
    "previous_pressure_2",
    "previous_pressure_3",
]


class TemporalInverseMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(15, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


class TemporalInverseController:
    def __init__(self, path):
        self.device = torch.device("cpu")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.feature_mean = np.asarray(
            checkpoint["feature_mean"],
            dtype=np.float64,
        )

        self.feature_std = np.asarray(
            checkpoint["feature_std"],
            dtype=np.float64,
        )

        self.target_mean = np.asarray(
            checkpoint["target_mean"],
            dtype=np.float64,
        )

        self.target_std = np.asarray(
            checkpoint["target_std"],
            dtype=np.float64,
        )

        if self.feature_mean.shape != (15,):
            raise RuntimeError("feature_mean must have shape (15,)")

        if self.feature_std.shape != (15,):
            raise RuntimeError("feature_std must have shape (15,)")

        if self.target_mean.shape != (3,):
            raise RuntimeError("target_mean must have shape (3,)")

        if self.target_std.shape != (3,):
            raise RuntimeError("target_std must have shape (3,)")

        self.feature_std = np.where(
            np.abs(self.feature_std) < 1e-8,
            1.0,
            self.feature_std,
        )

        self.target_std = np.where(
            np.abs(self.target_std) < 1e-8,
            1.0,
            self.target_std,
        )

        names = checkpoint.get("feature_names")

        if names is not None:
            names = [str(x) for x in names]

            if names != EXPECTED_FEATURE_NAMES:
                raise RuntimeError(
                    "Temporal MLP feature order does not match training.\n"
                    f"Expected: {EXPECTED_FEATURE_NAMES}\n"
                    f"Found:    {names}"
                )

        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )

        if state is None:
            state = checkpoint

        state = clean_state_dict(state)

        self.model = TemporalInverseMLP().to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        current_position,
        target_position,
        velocity,
        previous_pressure,
    ):
        current_position = np.asarray(
            current_position,
            dtype=np.float64,
        )

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        velocity = np.asarray(
            velocity,
            dtype=np.float64,
        )

        previous_pressure = np.asarray(
            previous_pressure,
            dtype=np.float64,
        )

        error = (
            target_position
            - current_position
        )

        features = np.concatenate(
            [
                current_position,
                target_position,
                error,
                velocity,
                previous_pressure,
            ]
        )

        normalized = (
            features - self.feature_mean
        ) / self.feature_std

        x = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 15)

        normalized_pressure = (
            self.model(x)
            .cpu()
            .numpy()
            .reshape(3)
        )

        pressure = (
            normalized_pressure
            * self.target_std
            + self.target_mean
        )

        pressure = np.nan_to_num(
            pressure,
            nan=1.5,
            posinf=P_MAX,
            neginf=P_MIN,
        )

        return np.clip(
            pressure,
            P_MIN,
            P_MAX,
        ).astype(np.float64)


# ============================================================================
# PRESSURE -> 15 ACTUATORS
# ============================================================================

class PressureMapper:
    def __init__(self):
        self.kinematics = Kinematics()

    def convert(self, model, pressure):
        pressure = np.clip(
            np.asarray(pressure, dtype=np.float64),
            P_MIN,
            P_MAX,
        )

        k = self.kinematics

        k.pressure_to_strain(pressure)
        k.compute_mean_strain()
        k.compute_extension()
        k.compute_curvature()

        slide, bend_x, bend_y = (
            k.compute_joint_targets()
        )

        actuators = np.zeros(
            model.nu,
            dtype=np.float64,
        )

        segments = min(
            5,
            model.nu // 3,
            len(slide),
            len(bend_x),
            len(bend_y),
        )

        for i in range(segments):
            j = 3 * i
            actuators[j] = bend_x[i]
            actuators[j + 1] = bend_y[i]
            actuators[j + 2] = slide[i]

        return actuators


# ============================================================================
# MUJOCO HELPERS
# ============================================================================

def stabilize_model(model):
    if model.nv > 0:
        model.dof_damping[:] = np.maximum(
            model.dof_damping * JOINT_DAMPING_SCALE,
            MIN_JOINT_DAMPING,
        )

    if model.nu > 0:
        model.actuator_damping[:] = np.maximum(
            model.actuator_damping,
            ACTUATOR_DAMPING,
        )

        model.actuator_armature[:] = np.maximum(
            model.actuator_armature,
            ACTUATOR_ARMATURE,
        )

    for actuator_id in range(model.nu):
        joint_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if joint_id < 0 or joint_id >= model.njnt:
            continue

        if int(
            model.actuator_biastype[actuator_id]
        ) != int(
            mujoco.mjtBias.mjBIAS_AFFINE
        ):
            continue

        kp = float(
            model.actuator_gainprm[
                actuator_id,
                0,
            ]
        )

        if kp <= 0.0:
            continue

        dof_adr = int(
            model.jnt_dofadr[joint_id]
        )

        inertia = max(
            float(model.dof_M0[dof_adr]),
            1e-8,
        )

        kv = (
            2.0
            * ACTUATOR_DAMPING_RATIO
            * math.sqrt(kp * inertia)
        )

        model.actuator_biasprm[
            actuator_id,
            2,
        ] = -kv

    model.opt.integrator = (
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    )

    model.opt.tolerance = 1e-6
    model.opt.iterations = 30
    model.opt.ls_iterations = 10


def finite_state(model, data):
    return (
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and np.all(np.isfinite(data.qacc))
    )


def find_site(model, names):
    for name in names:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id >= 0:
            return site_id

    return -1


def tip_position(model, data, site_id):
    return data.site_xpos[site_id].copy()


def hide_xml_target(model):
    target_id = find_site(
        model,
        (
            "target_site",
            "target",
            "goal_site",
            "goal",
        ),
    )

    if target_id >= 0:
        model.site_rgba[target_id, 3] = 0.0


def draw_target(viewer, target):
    scene = viewer.user_scn

    if scene.maxgeom < 1:
        return

    scene.ngeom = 0

    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [0.0045, 0.0045, 0.0045],
            dtype=np.float64,
        ),
        np.asarray(target, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.array(
            [0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    )

    scene.ngeom = 1


def render_vla_image(renderer, data):
    renderer.update_scene(
        data,
        camera=0,
    )

    image = renderer.render()

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(
            f"Unexpected renderer output: {image.shape}"
        )

    image = image[:, :, :3] / 255.0

    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .numpy()
    )


# ============================================================================
# PRESSURE SHAPING
# ============================================================================

# ============================================================================
# COMMAND HYSTERESIS — TINY, NON-DYNAMIC
# ============================================================================
#
# This is deliberately NOT another low-pass filter. It only prevents a
# numerically tiny target-pressure toggle from flipping sign every controller
# update. The physical response remains entirely inside pressure_step().
COMMAND_HYSTERESIS = 0.00020


def hold_tiny_pressure_changes(current, desired):
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)

    delta = desired - current
    out = desired.copy()

    tiny = np.abs(delta) < COMMAND_HYSTERESIS
    out[tiny] = current[tiny]

    return out


def pressure_step(current, desired):
    current = np.asarray(
        current,
        dtype=np.float64,
    )

    desired = np.asarray(
        desired,
        dtype=np.float64,
    )

    difference = desired - current

    desired = desired.copy()

    small = np.abs(difference) < PRESSURE_DEADBAND
    desired[small] = current[small]

    alpha = (
        CONTROL_DT
        /
        (PRESSURE_FILTER_TAU + CONTROL_DT)
    )

    filtered = (
        current
        + alpha * (desired - current)
    )

    max_change = (
        MAX_PRESSURE_RATE * CONTROL_DT
    )

    delta = np.clip(
        filtered - current,
        -max_change,
        max_change,
    )

    return np.clip(
        current + delta,
        P_MIN,
        P_MAX,
    )


# ============================================================================
# FIGURE-8 TRAJECTORY
# ============================================================================

class Figure8Trajectory:
    """
    Smooth closed figure-8 trajectory in the XY plane.

    Canonical curve:
        x = A sin(theta)
        y = A sin(2 theta)

    Arc-length reparameterization keeps the end-effector moving at nearly
    constant path speed. Z is constant, so the trajectory remains parallel
    to the top disc.
    """

    def __init__(self, z, scale, radial_start):
        self.z = float(z)
        self.scale = float(scale)
        self.radial_start = np.asarray(
            radial_start,
            dtype=np.float64,
        )

        self.samples = 8192

        theta = np.linspace(
            0.0,
            2.0 * math.pi,
            self.samples + 1,
        )

        x = self.scale * np.sin(theta)
        y = self.scale * np.sin(2.0 * theta)

        points = np.column_stack(
            [
                x + self.radial_start[0],
                y + self.radial_start[1],
                np.full_like(x, self.z),
            ]
        )

        ds = np.linalg.norm(
            np.diff(
                points[:, :2],
                axis=0,
            ),
            axis=1,
        )

        self.arc = np.concatenate(
            [
                np.array([0.0]),
                np.cumsum(ds),
            ]
        )

        self.points = points
        self.total_length = float(self.arc[-1])
        self.figure8_start = points[0].copy()

    @property
    def total_time(self):
        return APPROACH_TIME + FIGURE8_TIME

    def _figure8_position_by_fraction(self, fraction):
        fraction = float(
            np.clip(
                fraction,
                0.0,
                1.0,
            )
        )

        distance = fraction * self.total_length

        idx = int(
            np.searchsorted(
                self.arc,
                distance,
                side="right",
            )
        )

        idx = int(
            np.clip(
                idx,
                1,
                len(self.arc) - 1,
            )
        )

        d0 = self.arc[idx - 1]
        d1 = self.arc[idx]

        if d1 - d0 < 1e-12:
            alpha = 0.0
        else:
            alpha = (
                (distance - d0)
                / (d1 - d0)
            )

        return (
            self.points[idx - 1]
            + alpha
            * (
                self.points[idx]
                - self.points[idx - 1]
            )
        )

    def position(self, t):
        t = max(float(t), 0.0)

        if t < APPROACH_TIME:
            alpha = np.clip(
                t / APPROACH_TIME,
                0.0,
                1.0,
            )

            a2 = alpha * alpha
            a3 = a2 * alpha

            smooth = (
                10.0 * a3
                - 15.0 * a3 * alpha
                + 6.0 * a3 * alpha * alpha
            )

            return (
                self.radial_start
                + smooth
                * (
                    self.figure8_start
                    - self.radial_start
                )
            )

        u = np.clip(
            (t - APPROACH_TIME)
            / FIGURE8_TIME,
            0.0,
            1.0,
        )

        return self._figure8_position_by_fraction(u)

    def tangent(self, t):
        t = float(t)

        if t < APPROACH_TIME:
            p0 = self.position(
                max(0.0, t - 0.005)
            )
            p1 = self.position(
                min(APPROACH_TIME, t + 0.005)
            )
        else:
            dt = max(
                FIGURE8_TIME / self.samples * 8.0,
                1e-4,
            )

            p0 = self.position(
                max(APPROACH_TIME, t - dt)
            )
            p1 = self.position(
                min(self.total_time, t + dt)
            )

        tangent = p1 - p0
        tangent[2] = 0.0

        norm = float(
            np.linalg.norm(tangent)
        )

        if norm < 1e-10:
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        return tangent / norm

    def phase(self, t):
        if t < APPROACH_TIME:
            return "approach"

        if t < self.total_time:
            return "figure8"

        return "hold"


# ============================================================================
# LOCAL DLS CORRECTOR
# ============================================================================

class LocalFigure8Corrector:
    """
    Small local pressure correction around the temporal MLP command.

    The correction is derived from:
        pressure -> Kinematics -> MuJoCo -> future EE position

    It explicitly prioritizes:
        1. radial circle error
        2. Cartesian/tangential error
        3. Z error

    No accumulated error state is kept.
    """

    def __init__(self, model, mapper, site_id):
        self.model = model
        self.mapper = mapper
        self.site_id = site_id

        self.pred_data = mujoco.MjData(model)

    def predict_position(
        self,
        source,
        pressure,
    ):
        mujoco.mj_copyData(
            self.pred_data,
            self.model,
            source,
        )

        self.pred_data.ctrl[:] = (
            self.mapper.convert(
                self.model,
                pressure,
            )
        )

        end_time = (
            self.pred_data.time
            + PREDICTION_HORIZON
        )

        while (
            self.pred_data.time
            <
            end_time - 1e-12
        ):
            mujoco.mj_step(
                self.model,
                self.pred_data,
            )

        return (
            self.pred_data
            .site_xpos[self.site_id]
            .copy()
        )

    def jacobian(
        self,
        source,
        nominal,
    ):
        J = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        for axis in range(3):
            plus = nominal.copy()
            minus = nominal.copy()

            plus[axis] = min(
                P_MAX,
                plus[axis]
                + JACOBIAN_PRESSURE_STEP,
            )

            minus[axis] = max(
                P_MIN,
                minus[axis]
                - JACOBIAN_PRESSURE_STEP,
            )

            denom = (
                plus[axis]
                - minus[axis]
            )

            if denom <= 1e-12:
                continue

            p_plus = self.predict_position(
                source,
                plus,
            )

            p_minus = self.predict_position(
                source,
                minus,
            )

            J[:, axis] = (
                p_plus - p_minus
            ) / denom

        return J

    def score(
        self,
        source,
        target,
        pressure,
        center_xy,
        radius,
    ):
        predicted = self.predict_position(
            source,
            pressure,
        )

        position_error = float(
            np.linalg.norm(
                target - predicted
            )
        )

        radius_now = float(
            np.linalg.norm(
                predicted[:2]
                - center_xy
            )
        )

        radial_error = abs(
            radius_now - radius
        )

        target_xy = (
            target[:2] - center_xy
        )

        target_radius = float(
            np.linalg.norm(target_xy)
        )

        if target_radius > 1e-9:
            tangent = np.array(
                [
                    -target_xy[1],
                    target_xy[0],
                ],
                dtype=np.float64,
            )

            tangent /= target_radius

            tangential_error = abs(
                np.dot(
                    target[:2]
                    - predicted[:2],
                    tangent,
                )
            )
        else:
            tangential_error = position_error

        z_error = abs(
            target[2] - predicted[2]
        )

        return (
            POSITION_WEIGHT * position_error
            + RADIAL_WEIGHT * radial_error
            + TANGENTIAL_WEIGHT * tangential_error
            + Z_WEIGHT * z_error
            + PRESSURE_CHANGE_WEIGHT
            * float(np.linalg.norm(pressure))
        )

    def correct(
        self,
        source,
        target,
        nominal,
        center_xy,
        radius,
        circle_mode,
        tangent_dir=None,
        figure8_mode=False,
    ):
        nominal = np.clip(
            np.asarray(
                nominal,
                dtype=np.float64,
            ),
            P_MIN,
            P_MAX,
        )

        baseline = self.predict_position(
            source,
            nominal,
        )

        if figure8_mode:
            # For a non-circular curve, "radial" has no unique meaning.
            # Use the local Frenet-like tangent/normal frame of the target
            # trajectory. This keeps the correction on the figure-8 curve rather
            # than pulling the tip toward the origin.
            if tangent_dir is None:
                tangent_dir = np.array(
                    [0.0, 1.0, 0.0],
                    dtype=np.float64,
                )
            else:
                tangent_dir = np.asarray(
                    tangent_dir,
                    dtype=np.float64,
                ).copy()

            tangent_dir[2] = 0.0
            tangent_norm = float(
                np.linalg.norm(tangent_dir)
            )

            if tangent_norm < 1e-10:
                tangent_dir = np.array(
                    [0.0, 1.0, 0.0],
                    dtype=np.float64,
                )
            else:
                tangent_dir /= tangent_norm

            normal_dir = np.array(
                [
                    -tangent_dir[1],
                    tangent_dir[0],
                    0.0,
                ],
                dtype=np.float64,
            )

            xy_error = (
                target[:2]
                - baseline[:2]
            )

            normal_need = float(
                np.dot(
                    xy_error,
                    normal_dir[:2],
                )
            )

            tangential_need = float(
                np.dot(
                    xy_error,
                    tangent_dir[:2],
                )
            )

            desired = np.zeros(
                3,
                dtype=np.float64,
            )

            desired[:2] = (
                1.45
                * normal_need
                * normal_dir[:2]
                +
                0.82
                * tangential_need
                * tangent_dir[:2]
            )

            desired[2] = (
                1.25
                * (
                    target[2]
                    - baseline[2]
                )
            )

        elif circle_mode:
            radial_vec = (
                baseline[:2]
                - center_xy
            )

            radial_norm = float(
                np.linalg.norm(radial_vec)
            )

            target_vec = (
                target[:2]
                - center_xy
            )

            target_norm = float(
                np.linalg.norm(target_vec)
            )

            if radial_norm > 1e-9:
                radial_dir = (
                    radial_vec
                    / radial_norm
                )
            elif target_norm > 1e-9:
                radial_dir = (
                    target_vec
                    / target_norm
                )
            else:
                radial_dir = np.array(
                    [1.0, 0.0],
                    dtype=np.float64,
                )

            if target_norm > 1e-9:
                tangent_dir_2d = np.array(
                    [
                        -target_vec[1],
                        target_vec[0],
                    ],
                    dtype=np.float64,
                )
                tangent_dir_2d /= target_norm
            else:
                tangent_dir_2d = np.array(
                    [0.0, 1.0],
                    dtype=np.float64,
                )

            predicted_signed_radial_need = (
                radius - radial_norm
            )

            current_radius = float(
                np.linalg.norm(
                    source.site_xpos[self.site_id][:2]
                    - center_xy
                )
            )

            measured_signed_radial_need = (
                radius - current_radius
            )

            signed_radial_need = (
                0.70 * predicted_signed_radial_need
                + 0.30 * measured_signed_radial_need
            )

            tangential_need = float(
                np.dot(
                    target[:2]
                    - baseline[:2],
                    tangent_dir_2d,
                )
            )

            desired = np.zeros(
                3,
                dtype=np.float64,
            )

            desired[:2] = (
                1.70
                * signed_radial_need
                * radial_dir
                +
                0.75
                * tangential_need
                * tangent_dir_2d
            )

            desired[2] = (
                1.25
                * (
                    target[2]
                    - baseline[2]
                )
            )
        else:
            desired = target - baseline
            desired[2] *= 1.15

        desired_norm = float(
            np.linalg.norm(desired)
        )

        if desired_norm > MAX_POSITION_CORRECTION:
            desired *= (
                MAX_POSITION_CORRECTION
                / desired_norm
            )

        J = self.jacobian(
            source,
            nominal,
        )

        try:
            lhs = (
                J.T @ J
                + DLS_DAMPING
                * np.eye(3)
            )

            rhs = J.T @ desired

            delta = np.linalg.solve(
                lhs,
                rhs,
            )
        except np.linalg.LinAlgError:
            return nominal

        delta = np.clip(
            delta,
            -MAX_PRESSURE_CORRECTION,
            MAX_PRESSURE_CORRECTION,
        )

        # For the figure-8, score the actual Cartesian trajectory error plus a
        # local tangent/normal decomposition. Do NOT use circle radius.
        def score_trial(predicted, pressure):
            position_error = float(
                np.linalg.norm(
                    target - predicted
                )
            )

            if figure8_mode:
                tangent = np.asarray(
                    tangent_dir,
                    dtype=np.float64,
                )
                normal = np.array(
                    [
                        -tangent[1],
                        tangent[0],
                        0.0,
                    ],
                    dtype=np.float64,
                )

                e = target - predicted

                normal_error = abs(
                    np.dot(e[:2], normal[:2])
                )

                tangent_error = abs(
                    np.dot(e[:2], tangent[:2])
                )

                z_error = abs(
                    target[2] - predicted[2]
                )

                return (
                    1.0 * position_error
                    + 2.0 * normal_error
                    + 0.65 * tangent_error
                    + 1.5 * z_error
                    + PRESSURE_CHANGE_WEIGHT
                    * float(
                        np.linalg.norm(
                            pressure - nominal
                        )
                    )
                )

            return self.score(
                source,
                target,
                pressure,
                center_xy,
                radius,
            )

        best = nominal.copy()
        best_prediction = baseline
        best_score = score_trial(
            best_prediction,
            best,
        )

        for scale in (
            0.25,
            0.50,
            0.75,
            1.00,
        ):
            trial = np.clip(
                nominal
                + scale * delta,
                P_MIN,
                P_MAX,
            )

            trial_prediction = (
                baseline
                + J @ (
                    trial
                    - nominal
                )
            )

            if not np.all(
                np.isfinite(
                    trial_prediction
                )
            ):
                continue

            trial_score = score_trial(
                trial_prediction,
                trial,
            )

            if trial_score < best_score:
                best = trial
                best_score = trial_score

        return np.clip(
            best,
            P_MIN,
            P_MAX,
        )


# ============================================================================
# LOGGER
# ============================================================================

class TrajectoryLogger:
    def __init__(
        self,
        path,
        trajectory,
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(
            self.file
        )

        self.trajectory = trajectory

        # Keep an in-memory copy of every row.  The plotting/export stage
        # must not depend on re-reading a CSV that may be empty, truncated,
        # buffered, or otherwise malformed after a long MuJoCo run.
        self.fieldnames = [
            "time_s",
            "phase",
            "progress",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "actual_x_m",
            "actual_y_m",
            "actual_z_m",
            "error_x_m",
            "error_y_m",
            "error_z_m",
            "error_mm",
            "percentage_error",
            "accuracy_percent",
            "radial_error_mm",
            "radial_error_abs_mm",
            "z_error_mm",
            "tangential_error_mm",
            "velocity_x_mps",
            "velocity_y_mps",
            "velocity_z_mps",
            "pressure_1_bar",
            "pressure_2_bar",
            "pressure_3_bar",
            "vla_delta_1_bar",
            "vla_delta_2_bar",
            "vla_delta_3_bar",
        ]

        self.rows = []

        self.writer.writerow(
            [
                "time_s",
                "phase",
                "progress",

                "target_x_m",
                "target_y_m",
                "target_z_m",

                "actual_x_m",
                "actual_y_m",
                "actual_z_m",

                "error_x_m",
                "error_y_m",
                "error_z_m",

                "error_mm",
                "percentage_error",
                "accuracy_percent",

                "radial_error_mm",
                "radial_error_abs_mm",
                "z_error_mm",
                "tangential_error_mm",

                "velocity_x_mps",
                "velocity_y_mps",
                "velocity_z_mps",

                "pressure_1_bar",
                "pressure_2_bar",
                "pressure_3_bar",

                "vla_delta_1_bar",
                "vla_delta_2_bar",
                "vla_delta_3_bar",
            ]
        )

    def log(
        self,
        time_s,
        phase,
        progress,
        target,
        actual,
        velocity,
        pressure,
        vla_delta,
    ):
        target = np.asarray(
            target,
            dtype=np.float64,
        )

        actual = np.asarray(
            actual,
            dtype=np.float64,
        )

        velocity = np.asarray(
            velocity,
            dtype=np.float64,
        )

        pressure = np.asarray(
            pressure,
            dtype=np.float64,
        )

        vla_delta = np.asarray(
            vla_delta,
            dtype=np.float64,
        )

        error = target - actual

        error_mm = float(
            np.linalg.norm(error)
            * 1000.0
        )

        denom = max(
            float(np.linalg.norm(target)),
            1e-9,
        )

        percentage_error = (
            100.0
            * error_mm
            / (denom * 1000.0)
        )

        accuracy = float(
            np.clip(
                100.0
                - percentage_error,
                0.0,
                100.0,
            )
        )

        # For the figure-8, the old circle radius is not meaningful.
        # Store local signed normal/cross-track error in the same CSV fields
        # so existing analysis tools remain compatible.
        tangent = self.trajectory.tangent(time_s)

        normal = np.array(
            [
                -tangent[1],
                tangent[0],
                0.0,
            ],
            dtype=np.float64,
        )

        radial_error_mm = (
            np.dot(
                target[:2] - actual[:2],
                normal[:2],
            )
            * 1000.0
        )

        tangential_error_mm = abs(
            np.dot(
                target[:2] - actual[:2],
                tangent[:2],
            )
        ) * 1000.0

        row = {
            "time_s": float(time_s),
            "phase": str(phase),
            "progress": float(progress),

            "target_x_m": float(target[0]),
            "target_y_m": float(target[1]),
            "target_z_m": float(target[2]),

            "actual_x_m": float(actual[0]),
            "actual_y_m": float(actual[1]),
            "actual_z_m": float(actual[2]),

            "error_x_m": float(error[0]),
            "error_y_m": float(error[1]),
            "error_z_m": float(error[2]),

            "error_mm": float(error_mm),
            "percentage_error": float(percentage_error),
            "accuracy_percent": float(accuracy),

            "radial_error_mm": float(radial_error_mm),
            "radial_error_abs_mm": float(abs(radial_error_mm)),
            "z_error_mm": float(error[2] * 1000.0),
            "tangential_error_mm": float(tangential_error_mm),

            "velocity_x_mps": float(velocity[0]),
            "velocity_y_mps": float(velocity[1]),
            "velocity_z_mps": float(velocity[2]),

            "pressure_1_bar": float(pressure[0]),
            "pressure_2_bar": float(pressure[1]),
            "pressure_3_bar": float(pressure[2]),

            "vla_delta_1_bar": float(vla_delta[0]),
            "vla_delta_2_bar": float(vla_delta[1]),
            "vla_delta_3_bar": float(vla_delta[2]),
        }

        # Store first, then write.  Thus the plotting data survives even if
        # the CSV stream is unavailable at the end of the simulation.
        self.rows.append(row)

        self.writer.writerow(
            [row[name] for name in self.fieldnames]
        )

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# SUMMARY
# ============================================================================

def make_summary(
    run_id,
    trajectory,
    times,
    errors,
    radial_errors,
    z_errors,
    accuracy_values,
    pressures,
    final_target,
    final_actual,
    final_pressure,
    csv_path,
):
    errors = np.asarray(errors, dtype=np.float64)
    radial_errors = np.asarray(
        radial_errors,
        dtype=np.float64,
    )
    z_errors = np.asarray(
        z_errors,
        dtype=np.float64,
    )
    accuracy_values = np.asarray(
        accuracy_values,
        dtype=np.float64,
    )
    pressures = np.asarray(
        pressures,
        dtype=np.float64,
    )

    figure8_mask = (
        np.asarray(times)
        >= APPROACH_TIME
    )

    if not np.any(figure8_mask):
        figure8_mask = np.ones(
            len(errors),
            dtype=bool,
        )

    figure8_errors = errors[figure8_mask]
    figure8_cross_track = radial_errors[figure8_mask]
    figure8_z_errors = z_errors[figure8_mask]

    figure8_accuracy = accuracy_values[figure8_mask]

    mean_accuracy = float(
        np.mean(figure8_accuracy)
    )

    return {
        "run_id": run_id,

        "benchmark": {
            "required_accuracy_percent": 92.0,
            "figure8_accuracy_percent": mean_accuracy,
            "passed": bool(
                mean_accuracy >= 92.0
            ),
        },

        "figure-8": {
            "scale_m": FIGURE8_SCALE,
            "half_width_mm": 75.0,
            "approach_time_s": APPROACH_TIME,
            "figure8_time_s": FIGURE8_TIME,
            "total_time_s": trajectory.total_time,
        },

        "whole_run": {
            "mean_error_mm": float(
                np.mean(errors)
            ),
            "median_error_mm": float(
                np.median(errors)
            ),
            "rmse_mm": float(
                np.sqrt(np.mean(errors ** 2))
            ),
            "p95_error_mm": float(
                np.percentile(errors, 95)
            ),
            "peak_error_mm": float(
                np.max(errors)
            ),
            "mean_accuracy_percent": float(
                np.mean(accuracy_values)
            ),
        },

        "figure8_only": {
            "mean_error_mm": float(
                np.mean(figure8_errors)
            ),
            "median_error_mm": float(
                np.median(figure8_errors)
            ),
            "rmse_mm": float(
                np.sqrt(
                    np.mean(
                        figure8_errors ** 2
                    )
                )
            ),
            "p95_error_mm": float(
                np.percentile(
                    figure8_errors,
                    95,
                )
            ),
            "peak_error_mm": float(
                np.max(figure8_errors)
            ),

            "mean_accuracy_percent": mean_accuracy,

            "mean_radial_abs_error_mm": float(
                np.mean(
                    np.abs(figure8_cross_track)
                )
            ),

            "p95_radial_abs_error_mm": float(
                np.percentile(
                    np.abs(figure8_cross_track),
                    95,
                )
            ),

            "mean_z_abs_error_mm": float(
                np.mean(
                    np.abs(figure8_z_errors)
                )
            ),

            "p95_z_abs_error_mm": float(
                np.percentile(
                    np.abs(figure8_z_errors),
                    95,
                )
            ),

            "samples_within_5_percent": float(
                100.0
                * np.mean(
                    figure8_errors
                    <=
                    0.05
                    * FIGURE8_SCALE
                    * 1000.0
                )
            ),
        },

        "final": {
            "target_m": np.asarray(
                final_target
            ).tolist(),

            "actual_m": np.asarray(
                final_actual
            ).tolist(),

            "error_mm": float(
                np.linalg.norm(
                    np.asarray(final_target)
                    - np.asarray(final_actual)
                )
                * 1000.0
            ),

            "pressure_bar": np.asarray(
                final_pressure
            ).tolist(),
        },

        "pressure": {
            "mean_bar": pressures.mean(
                axis=0
            ).tolist(),

            "min_bar": pressures.min(
                axis=0
            ).tolist(),

            "max_bar": pressures.max(
                axis=0
            ).tolist(),
        },

        "files": {
            "trajectory_csv": str(
                csv_path
            ),
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def find_mlp():
    for path in MLP_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find temporal inverse MLP.\n"
        "Checked:\n"
        + "\n".join(
            str(x)
            for x in MLP_CANDIDATES
        )
    )


def main():
    print()
    print("=" * 82)
    print("SOFT ROBOT — VLA + TEMPORAL MLP — 70 mm HEART")
    print("=" * 82)

    mlp_path = find_mlp()

    print()
    print("Project root :", ROOT)
    print("MuJoCo model :", MODEL_PATH)
    print("VLA          :", VLA_PATH)
    print("Temporal MLP :", mlp_path)
    print("Device       :", DEVICE)
    print("Animation    :", f"{ANIMATION_SPEEDUP:.1f}x")
    print("Figure-8       :", f"{FIGURE8_SCALE * 1000.0:.1f} mm")
   

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MuJoCo model not found:\n{MODEL_PATH}"
        )

    # Load learned components BEFORE launching the viewer.
    # This guarantees checkpoint problems are caught immediately.
    vla, vocab = load_vla()
    inverse = TemporalInverseController(
        mlp_path
    )

    mapper = PressureMapper()

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    stabilize_model(model)

    data = mujoco.MjData(model)

    if model.nu != 15:
        raise RuntimeError(
            f"Expected 15 actuators, found {model.nu}"
        )

    tip_site_id = find_site(
        model,
        (
            "end_effector",
            "tip",
            "ee",
            "end",
        ),
    )

    if tip_site_id < 0:
        raise RuntimeError(
            "Could not find end-effector site."
        )

    hide_xml_target(model)

    # ------------------------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )

    pressure = (
        INITIAL_PRESSURE.copy()
    )

    data.ctrl[:] = (
        mapper.convert(
            model,
            pressure,
        )
    )

    mujoco.mj_forward(
        model,
        data,
    )

    # One-second startup settle.
    settle_end = (
        data.time + 1.0
    )

    while (
        data.time
        <
        settle_end - 1e-12
    ):
        data.ctrl[:] = (
            mapper.convert(
                model,
                pressure,
            )
        )

        mujoco.mj_step(
            model,
            data,
        )

        if not finite_state(
            model,
            data,
        ):
            raise RuntimeError(
                "MuJoCo became unstable during startup."
            )

    initial_tip = tip_position(
        model,
        data,
        tip_site_id,
    )

    # Center the figure-8 on the robot's initial XY position so the complete
    # trajectory stays inside the same local workspace used by the circle.
    # The figure-8 remains entirely in XY and therefore stays parallel to the
    # top disc.
    figure8_center = initial_tip[:2].copy()

    figure8_z = float(
        initial_tip[2]
        + FIGURE8_HEIGHT_ABOVE_TIP
    )

    radial_start = initial_tip.copy()
    radial_start[2] = figure8_z

    trajectory = Figure8Trajectory(
        figure8_z,
        FIGURE8_SCALE,
        radial_start,
    )

    # ------------------------------------------------------------------------
    # VLA
    # ------------------------------------------------------------------------

    vla_controller = VLAResidualController(
        vla,
        vocab,
    )

    renderer = mujoco.Renderer(
        model,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )

    # ------------------------------------------------------------------------
    # LOCAL CORRECTOR
    # ------------------------------------------------------------------------

    corrector = LocalFigure8Corrector(
        model,
        mapper,
        tip_site_id,
    )

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------

    run_id = (
        "figure8_vla_mlp_v10_"
        + time.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_dir = (
        DESKTOP_ROOT
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        run_dir
        / "trajectory.csv"
    )

    summary_path = (
        run_dir
        / "summary.json"
    )

    config_path = (
        run_dir
        / "config.json"
    )

    logger = TrajectoryLogger(
        csv_path,
        trajectory,
    )

    config = {
        "architecture": {
            "vla": "original 6-state SoftRobotVLA",
            "vla_output": "normalized pressure delta",
            "temporal_mlp": "15-feature temporal inverse MLP",
            "pressure_to_actuators": "Kinematics",
            "actuators": 15,
            "local_correction": "bounded DLS",
            "pid": False,
            "integral_accumulator": False,
        },

        "paths": {
            "model": str(MODEL_PATH),
            "vla": str(VLA_PATH),
            "temporal_mlp": str(mlp_path),
        },

        "figure-8": {
            "center_xy_m": figure8_center.tolist(),
            "z_m": figure8_z,
            "scale_m": FIGURE8_SCALE,
            "half_width_mm": 75.0,
            "approach_time_s": APPROACH_TIME,
            "figure8_time_s": FIGURE8_TIME,
        },

        "control": {
            "physics_dt_s": CONTROL_DT,
            "controller_update_dt_s": CONTROL_UPDATE_DT,
            "vla_update_dt_s": VLA_UPDATE_DT,
            "lookahead_s": LOOKAHEAD_TIME,
            "prediction_horizon_s": PREDICTION_HORIZON,
            "vla_residual_weight": VLA_RESIDUAL_WEIGHT,
            "max_vla_pressure_delta_bar": MAX_VLA_PRESSURE_DELTA,
            "pressure_filter_tau_s": PRESSURE_FILTER_TAU,
            "max_pressure_rate_bar_s": MAX_PRESSURE_RATE,
        },

        "animation": {
            "speedup": ANIMATION_SPEEDUP,
            "render_every_n_physics_steps": RENDER_EVERY_N_STEPS,
        },

        "stability_profile": {
            "name": "accurate_low_vibration_figure8_v1",
            "command_hysteresis_bar": COMMAND_HYSTERESIS,
            "second_dynamic_filter": False,
            "pid": False,
            "integral": False,
        },

        "plt_figure8": {
            "archive_csv": "figure8_xy_tracking_YYYYMMDD_HHMMSS.csv",
            "project_csv": "results/figure8_xy_tracking.csv",
        },

        "initial_tip_m": initial_tip.tolist(),
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print()
    print("Initial EE :", np.round(initial_tip, 6))
    print("Figure-8 Z   :", round(figure8_z, 6))
    print("Run folder :", run_dir)
    print()
    print("Controller:")
    print("  Exact VLA checkpoint architecture : ON")
    print("  Temporal inverse MLP              : ON")
    print("  VLA residual                      :", VLA_RESIDUAL_WEIGHT)
    print("  Figure-8 trajectory correction       : ON")
    print("  Local DLS correction              : ON")
    print("  PID                                : OFF")
    print("  Integral accumulator               : OFF")
    print("  Physics timestep                   :", CONTROL_DT)
    print("  Animation                          :", f"{ANIMATION_SPEEDUP:.1f}x")
    print()

    # ------------------------------------------------------------------------
    # VIEWER
    # ------------------------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.distance = 0.34
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array(
            [
                0.0,
                0.0,
                figure8_z,
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------------------------
        # RUNTIME STATE
        # --------------------------------------------------------------------

        sim_start = float(data.time)

        pressure_target = pressure.copy()

        vla_delta = np.zeros(
            3,
            dtype=np.float64,
        )

        last_position = initial_tip.copy()

        velocity = np.zeros(
            3,
            dtype=np.float64,
        )

        next_control_update = sim_start
        next_vla_update = sim_start

        last_print_wall = (
            time.perf_counter()
        )

        wall_start = (
            time.perf_counter()
        )

        error_samples = []
        radial_samples = []
        z_samples = []
        accuracy_samples = []
        pressure_samples = []
        times = []

        physics_step_count = 0

        final_hold_start = None

        # --------------------------------------------------------------------
        # MAIN LOOP
        # --------------------------------------------------------------------

        while viewer.is_running():

            sim_time = float(
                data.time - sim_start
            )

            if sim_time <= trajectory.total_time:
                target = trajectory.position(
                    sim_time
                )
            else:
                target = trajectory.position(
                    trajectory.total_time
                )

            phase = trajectory.phase(
                sim_time
            )

            progress = float(
                np.clip(
                    sim_time
                    /
                    trajectory.total_time,
                    0.0,
                    1.0,
                )
            )

            current = tip_position(
                model,
                data,
                tip_site_id,
            )

            # Measured velocity for the temporal MLP.
            raw_velocity = (
                current - last_position
            ) / CONTROL_DT

            velocity = (
                0.65 * velocity
                + 0.35 * raw_velocity
            )

            # ----------------------------------------------------------------
            # LEARNED CONTROLLER UPDATE
            # ----------------------------------------------------------------

            if sim_time >= next_control_update - 1e-12:

                future_target = trajectory.position(
                    min(
                        sim_time
                        + LOOKAHEAD_TIME,
                        trajectory.total_time,
                    )
                )

                mlp_pressure = inverse.predict(
                    current_position=current,
                    target_position=future_target,
                    velocity=velocity,
                    previous_pressure=pressure,
                )

                # ------------------------------------------------------------
                # VLA residual
                # ------------------------------------------------------------

                if sim_time >= next_vla_update - 1e-12:

                    image = render_vla_image(
                        renderer,
                        data,
                    )

                    vla_delta = (
                        vla_controller.predict(
                            image=image,
                            current=current,
                            target=future_target,
                            velocity=velocity,
                        )
                    )

                    next_vla_update += (
                        VLA_UPDATE_DT
                    )

                    if (
                        next_vla_update
                        <
                        sim_time
                    ):
                        next_vla_update = (
                            sim_time
                            + VLA_UPDATE_DT
                        )

                # ------------------------------------------------------------
                # Combine:
                #
                # Temporal MLP = nominal absolute pressure.
                # VLA = bounded pressure residual.
                # ------------------------------------------------------------

                nominal_pressure = np.clip(
                    mlp_pressure
                    + VLA_RESIDUAL_WEIGHT
                    * vla_delta,
                    P_MIN,
                    P_MAX,
                )

                # ------------------------------------------------------------
                # Geometry correction.
                # ------------------------------------------------------------

                figure8_mode = (
                    phase == "figure8"
                )

                pressure_target = (
                    corrector.correct(
                        source=data,
                        target=future_target,
                        nominal=nominal_pressure,
                        center_xy=figure8_center,
                        radius=FIGURE8_SCALE,
                        circle_mode=False,
                        tangent_dir=trajectory.tangent(
                            min(
                                sim_time
                                + LOOKAHEAD_TIME,
                                trajectory.total_time,
                            )
                        ),
                        figure8_mode=figure8_mode,
                    )
                )

                next_control_update += (
                    CONTROL_UPDATE_DT
                )

                if (
                    next_control_update
                    <
                    sim_time
                ):
                    next_control_update = (
                        sim_time
                        + CONTROL_UPDATE_DT
                    )

                last_position = current.copy()

            # ----------------------------------------------------------------
            # PRESSURE DYNAMICS
            # ----------------------------------------------------------------

            pressure_target = hold_tiny_pressure_changes(
                pressure,
                pressure_target,
            )

            pressure = pressure_step(
                pressure,
                pressure_target,
            )

            data.ctrl[:] = (
                mapper.convert(
                    model,
                    pressure,
                )
            )

            # ----------------------------------------------------------------
            # ONE PHYSICAL STEP
            # ----------------------------------------------------------------

            mujoco.mj_step(
                model,
                data,
            )

            physics_step_count += 1

            if not finite_state(
                model,
                data,
            ):
                raise RuntimeError(
                    "MuJoCo became numerically unstable."
                )

            # ----------------------------------------------------------------
            # MEASURE
            # ----------------------------------------------------------------

            actual = tip_position(
                model,
                data,
                tip_site_id,
            )

            error = (
                target - actual
            )

            error_mm = float(
                np.linalg.norm(error)
                * 1000.0
            )

            actual_radius = float(
                np.linalg.norm(
                    actual[:2]
                    - figure8_center
                )
            )

            radial_error_mm = (
                actual_radius
                - FIGURE8_SCALE
            ) * 1000.0

            z_error_mm = (
                target[2]
                - actual[2]
            ) * 1000.0

            denom = max(
                float(np.linalg.norm(target)),
                1e-9,
            )

            percentage_error = (
                100.0
                * error_mm
                / (denom * 1000.0)
            )

            accuracy = float(
                np.clip(
                    100.0
                    - percentage_error,
                    0.0,
                    100.0,
                )
            )

            error_samples.append(
                error_mm
            )

            radial_samples.append(
                radial_error_mm
            )

            z_samples.append(
                z_error_mm
            )

            accuracy_samples.append(
                accuracy
            )

            pressure_samples.append(
                pressure.copy()
            )

            times.append(
                sim_time
            )

            logger.log(
                time_s=sim_time,
                phase=phase,
                progress=progress,
                target=target,
                actual=actual,
                velocity=velocity,
                pressure=pressure,
                vla_delta=vla_delta,
            )

            # ----------------------------------------------------------------
            # VIEWER
            # ----------------------------------------------------------------

            if (
                physics_step_count
                % RENDER_EVERY_N_STEPS
                == 0
            ):
                draw_target(
                    viewer,
                    target,
                )

                viewer.sync()

            # ----------------------------------------------------------------
            # TERMINATION
            # ----------------------------------------------------------------

            if (
                sim_time
                >=
                trajectory.total_time
            ):
                if final_hold_start is None:
                    final_hold_start = sim_time

                if (
                    sim_time
                    - final_hold_start
                    >= FINAL_HOLD
                ):
                    break

            # ----------------------------------------------------------------
            # STATUS
            # ----------------------------------------------------------------

            now_wall = time.perf_counter()

            if (
                now_wall
                - last_print_wall
                >= 0.25
            ):
                figure8_mask = (
                    np.asarray(times)
                    >= APPROACH_TIME
                )

                if np.any(figure8_mask):
                    live_accuracy = float(
                        np.mean(
                            np.asarray(
                                accuracy_samples
                            )[figure8_mask]
                        )
                    )
                else:
                    live_accuracy = float(
                        np.mean(
                            accuracy_samples
                        )
                    )

                print(
                    f"{phase.upper():8s} "
                    f"{progress * 100:6.1f}% | "
                    f"Err {error_mm:7.3f} mm | "
                    f"Cross {radial_error_mm:+7.3f} mm | "
                    f"Z {z_error_mm:+7.3f} mm | "
                    f"Figure-8Acc {live_accuracy:6.2f}% | "
                    f"P {np.round(pressure, 3)}"
                )

                last_print_wall = now_wall

            # ----------------------------------------------------------------
            # 15x WALL-CLOCK PACING
            #
            # The simulation still advances by CONTROL_DT every iteration.
            # We only make wall time pass at 1/15 of the simulated duration.
            # ----------------------------------------------------------------

            target_wall = (
                wall_start
                + sim_time / ANIMATION_SPEEDUP
            )

            sleep_for = (
                target_wall
                - time.perf_counter()
            )

            if sleep_for > 0.0:
                time.sleep(
                    min(
                        sleep_for,
                        0.003,
                    )
                )

        final_target = trajectory.position(
            trajectory.total_time
        )

        final_actual = tip_position(
            model,
            data,
            tip_site_id,
        )

    logger.flush()
    logger.close()

    summary = make_summary(
        run_id=run_id,
        trajectory=trajectory,
        times=times,
        errors=error_samples,
        radial_errors=radial_samples,
        z_errors=z_samples,
        accuracy_values=accuracy_samples,
        pressures=pressure_samples,
        final_target=final_target,
        final_actual=final_actual,
        final_pressure=pressure,
        csv_path=csv_path,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------------------
    # HEART XY MAP EXPORT
    # --------------------------------------------------------------------
    #
    # trajectory.csv remains the COMPLETE trajectory:
    #     approach -> figure-8 -> hold
    #
    # figure8_xy_tracking.csv is a dedicated plotting file containing ONLY
    # samples from the actual figure-8 phase.  This is important because the
    # approach segment is a straight radial motion and must not be included
    # when plotting the XY figure-8.
    #
    # Rows are kept in chronological order.  Nothing is sorted by X/Y.
    plt_figure8_csv = (
        DESKTOP_ROOT
        / (
            "figure8_xy_tracking_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )
    )

    project_results = (
        ROOT
        / "results"
    )
    project_results.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_figure8_csv = (
        project_results
        / "figure8_xy_tracking.csv"
    )

    # --------------------------------------------------------------------
    # FIGURE-8 XY EXPORT -- IN-MEMORY, FAIL-SAFE
    # --------------------------------------------------------------------
    #
    # IMPORTANT:
    # We DO NOT reopen trajectory.csv here.
    #
    # The old implementation reopened the CSV and tried to infer the
    # figure-8 section from a phase string.  That is exactly the wrong place
    # to fail after a successful controller run.
    #
    # TrajectoryLogger now retains every logged row in `logger.rows`.
    # Therefore this export uses the exact same samples that were generated
    # by the controller in the current run.
    #
    # Figure-8 membership is determined from the actual simulation time:
    #
    #       APPROACH_TIME <= time <= trajectory.total_time
    #
    # This is independent of CSV buffering, phase spelling, or file state.

    all_rows = list(
        logger.rows
    )

    figure8_rows = []

    for row in all_rows:
        try:
            row_time = float(
                row["time_s"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if (
            row_time >= APPROACH_TIME - 1e-9
            and row_time <= trajectory.total_time + 1e-9
        ):
            figure8_rows.append(row)

    # In a normal run this should contain essentially the entire figure-8
    # portion.  Do not crash if a user closes the viewer unusually early.
    if len(figure8_rows) < 2:
        print()
        print(
            "WARNING: fewer than two figure-8 samples were logged."
        )
        print(
            "The controller run completed, but no XY figure-8 map "
            "was exported."
        )
    else:
        for output_path in (
            plt_figure8_csv,
            project_figure8_csv,
        ):
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output_file:

                writer = csv.DictWriter(
                    output_file,
                    fieldnames=logger.fieldnames,
                    extrasaction="ignore",
                )

                writer.writeheader()

                writer.writerows(
                    figure8_rows
                )

        print()
        print(
            "FIGURE-8 XY EXPORT: SUCCESS"
        )
        print(
            f"  Samples exported : {len(figure8_rows)}"
        )
        print(
            f"  First t          : {figure8_rows[0]['time_s']:.4f} s"
        )
        print(
            f"  Last t           : {figure8_rows[-1]['time_s']:.4f} s"
        )
        print(
            f"  Plot CSV         : {plt_figure8_csv}"
        )
        print(
            f"  Project CSV      : {project_figure8_csv}"
        )

    print()
    print("=" * 82)
    print("TRACKING COMPLETE")
    print("=" * 82)

    print(
        f"Figure-8 accuracy        : "
        f"{summary['benchmark']['figure8_accuracy_percent']:.3f}%"
    )

    print(
        f"92% benchmark         : "
        f"{'PASS' if summary['benchmark']['passed'] else 'FAIL'}"
    )

    print(
        f"Figure-8 mean error      : "
        f"{summary['figure8_only']['mean_error_mm']:.3f} mm"
    )

    print(
        f"Figure-8 RMSE            : "
        f"{summary['figure8_only']['rmse_mm']:.3f} mm"
    )

    print(
        f"Mean cross-track abs error: "
        f"{summary['figure8_only']['mean_radial_abs_error_mm']:.3f} mm"
    )

    print(
        f"Mean Z abs error       : "
        f"{summary['figure8_only']['mean_z_abs_error_mm']:.3f} mm"
    )

    print()
    print("Saved:")
    print("  ", run_dir)
    print("  ", csv_path)
    print("  ", plt_figure8_csv)
    print("  ", project_figure8_csv)
    print("  ", summary_path)
    print("  ", config_path)
    print("=" * 82)


if __name__ == "__main__":
    main()
