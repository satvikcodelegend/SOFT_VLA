#!/usr/bin/env python3
"""
NYU SOFT ROBOT — UNSEEN SUPERELLIPSE GENERALIZATION TEST
==========================================================

IMPORTANT:
    This file is COMPLETELY independent of every figure-8 main file.

    It does NOT:
        - import main_figure8.py
        - import main_figure8_FINAL_FIXED.py
        - read figure-8 trajectory data
        - use figure-8-specific controller logic
        - use figure-8-specific training data

The only learned assets are the normal project checkpoints:
    model/scene.xml
    soft_robot_vla/checkpoints/best_model.pt
    models/temporal_inverse_mlp.pt
    control/kinematics.py

The test trajectory is a held-out geometric family:
    rotated superellipse / rounded-square

Controller:
    exact analytic target
        -> original 6-state VLA residual
        -> temporal inverse MLP absolute P1/P2/P3
        -> generic Frenet normal/tangent/Z predictive DLS
        -> pressure filter + slew limit
        -> Kinematics
        -> 15 actuators
        -> MuJoCo
        -> measured EE feedback

The geometry controller is generic: it uses the local tangent/normal of
whatever target curve is supplied. No circle or figure-8 equation is used
inside the correction stage.
"""

from __future__ import annotations

import csv
import json
import math
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
# EMBEDDED VLA ARCHITECTURE
# ============================================================================
#
# IMPORTANT:
#   The VLA is defined here instead of importing soft_robot_vla.model.
#   This makes this unseen-shape test independent of whichever version of
#   SoftRobotVLA happens to be installed in the local Python environment.
#
#   This is the checkpoint architecture used by the working NYU controller:
#       image [B,3,H,W]
#       instruction tokens [B,T]
#       state [B,6] = [x,y,z,vx,vy,vz]
#       target_error [B,3]
#       -> normalized pressure delta [B,3]
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
            [ResidualMLPBlock(hidden_dim) for _ in range(num_residual_blocks)]
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
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "scene.xml"
VLA_PATH = ROOT / "soft_robot_vla" / "checkpoints" / "best_model.pt"

MLP_CANDIDATES = (
    ROOT / "models" / "temporal_inverse_mlp.pt",
    ROOT / "model" / "temporal_inverse_mlp.pt",
    ROOT / "models" / "inverse_mlp.pt",
    ROOT / "model" / "inverse_mlp.pt",
)

DESKTOP_ROOT = Path.home() / "Desktop" / "soft_robot_runs"
RESULTS_ROOT = ROOT / "results"
DESKTOP_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DEVICE / TRAINED INTERFACES
# ============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

IMAGE_SIZE = 224
MAX_INSTRUCTION_LENGTH = 32

# Keep a vocabulary-safe instruction already used by the project.
# Geometry is supplied explicitly through the target/error, not through
# a new unseen language token.
INSTRUCTION = "follow a circle trajectory"

P_MIN = 0.0
P_MAX = 3.0

INITIAL_PRESSURE = np.array(
    [1.5, 1.5, 1.5],
    dtype=np.float64,
)


# ============================================================================
# CONTROL / STABILITY
# ============================================================================

CONTROL_DT = 0.010
CONTROL_UPDATE_DT = 0.020
VLA_UPDATE_DT = 0.100

LOOKAHEAD_TIME = 0.060
PREDICTION_HORIZON = 0.050

PRESSURE_FILTER_TAU = 0.040
MAX_PRESSURE_RATE = 3.80
PRESSURE_DEADBAND = 0.00010

MAX_VLA_PRESSURE_DELTA = 0.015
VLA_RESIDUAL_WEIGHT = 0.08

JACOBIAN_PRESSURE_STEP = 0.018
JACOBIAN_UPDATE_DT = 0.12
JACOBIAN_EMA = 0.65

DLS_DAMPING = 1.5e-4
MAX_PRESSURE_CORRECTION = 0.36
PRESSURE_CHANGE_WEIGHT = 0.010

# Generic trajectory-manifold gains.
POSITION_GAIN = 0.80
NORMAL_GAIN = 1.70
TANGENTIAL_GAIN = 0.75
Z_GAIN = 1.25

MAX_POSITION_CORRECTION = 0.020

CORRECTION_SCALES = (
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
)

JOINT_DAMPING_SCALE = 2.5
MIN_JOINT_DAMPING = 0.012
ACTUATOR_DAMPING = 0.030
ACTUATOR_ARMATURE = 0.003
ACTUATOR_DAMPING_RATIO = 1.25

ANIMATION_SPEEDUP = 15.0
RENDER_EVERY_N_STEPS = 2


# ============================================================================
# UNSEEN TRAJECTORY
# ============================================================================

TEST_NAME = "rotated_superellipse_unseen"

SEMI_AXIS_A = 0.035
SEMI_AXIS_B = 0.031
EXPONENT = 3.4
ROTATION_DEG = 18.0

ARC_SAMPLES = 8192

APPROACH_TIME = 6.0
TEST_TIME = 45.0
FINAL_HOLD = 1.0
HEIGHT_ABOVE_TIP = 0.025


class SuperellipseTrajectory:
    """Closed rotated superellipse with arc-length parameterization."""

    def __init__(
        self,
        z: float,
        center_xy: np.ndarray,
        a: float,
        b: float,
        exponent: float,
        rotation_deg: float,
    ):
        self.z = float(z)
        self.center_xy = np.asarray(center_xy, dtype=np.float64)
        self.a = float(a)
        self.b = float(b)
        self.exponent = float(exponent)
        self.rotation_deg = float(rotation_deg)

        if self.a <= 0 or self.b <= 0:
            raise ValueError("Superellipse axes must be positive.")
        if self.exponent <= 2:
            raise ValueError("Superellipse exponent must be > 2.")

        theta = np.linspace(
            0.0,
            2.0 * math.pi,
            ARC_SAMPLES + 1,
        )

        power = 2.0 / self.exponent

        x = (
            self.a
            * np.sign(np.cos(theta))
            * np.abs(np.cos(theta)) ** power
        )

        y = (
            self.b
            * np.sign(np.sin(theta))
            * np.abs(np.sin(theta)) ** power
        )

        angle = math.radians(self.rotation_deg)
        c = math.cos(angle)
        s = math.sin(angle)

        xr = c * x - s * y
        yr = s * x + c * y

        self.points = np.column_stack(
            [
                xr + self.center_xy[0],
                yr + self.center_xy[1],
                np.full_like(xr, self.z),
            ]
        )

        ds = np.linalg.norm(
            np.diff(self.points[:, :2], axis=0),
            axis=1,
        )

        self.arc = np.concatenate(
            [
                np.array([0.0]),
                np.cumsum(ds),
            ]
        )

        self.total_length = float(self.arc[-1])
        self.start = self.points[0].copy()

    @property
    def total_time(self):
        return APPROACH_TIME + TEST_TIME

    def _curve_position(self, fraction: float):
        fraction = float(np.clip(fraction, 0.0, 1.0))
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
            alpha = (distance - d0) / (d1 - d0)

        return (
            self.points[idx - 1]
            + alpha * (self.points[idx] - self.points[idx - 1])
        )

    def position(self, t: float):
        t = max(float(t), 0.0)

        if t < APPROACH_TIME:
            s = np.clip(t / APPROACH_TIME, 0.0, 1.0)
            s2 = s * s
            s3 = s2 * s
            smooth = 10.0 * s3 - 15.0 * s3 * s + 6.0 * s3 * s2

            return (
                self.start * smooth
                + self.radial_start * (1.0 - smooth)
            )

        fraction = np.clip(
            (t - APPROACH_TIME) / TEST_TIME,
            0.0,
            1.0,
        )
        return self._curve_position(fraction)

    def tangent(self, t: float):
        t = float(t)

        if t < APPROACH_TIME:
            h = 0.005
            p0 = self.position(max(0.0, t - h))
            p1 = self.position(min(APPROACH_TIME, t + h))
        else:
            h = max(TEST_TIME / ARC_SAMPLES * 12.0, 1e-4)
            p0 = self.position(max(APPROACH_TIME, t - h))
            p1 = self.position(min(self.total_time, t + h))

        tangent = p1 - p0
        tangent[2] = 0.0

        n = float(np.linalg.norm(tangent))
        if n < 1e-10:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        return tangent / n

    def phase(self, t: float):
        if t < APPROACH_TIME:
            return "approach"
        if t < self.total_time:
            return "superellipse"
        return "hold"


# ============================================================================
# HELPERS
# ============================================================================

def find_mlp():
    for path in MLP_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Temporal inverse MLP not found. Checked:\n"
        + "\n".join(str(p) for p in MLP_CANDIDATES)
    )


def finite_state(model, data):
    return (
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and np.all(np.isfinite(data.qacc))
        and np.all(np.isfinite(data.site_xpos))
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
        ("target_site", "target", "goal_site", "goal"),
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
        np.array([0.0045, 0.0045, 0.0045], dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    scene.ngeom = 1


def render_vla_image(renderer, data):
    renderer.update_scene(data, camera=0)
    image = np.asarray(renderer.render(), dtype=np.float32)

    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Unexpected renderer output: {image.shape}")

    image = image[:, :, :3] / 255.0
    return np.transpose(image, (2, 0, 1)).copy()


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
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or joint_id >= model.njnt:
            continue

        if int(model.actuator_biastype[actuator_id]) != int(
            mujoco.mjtBias.mjBIAS_AFFINE
        ):
            continue

        kp = float(model.actuator_gainprm[actuator_id, 0])
        if kp <= 0.0:
            continue

        dof_adr = int(model.jnt_dofadr[joint_id])
        inertia = max(float(model.dof_M0[dof_adr]), 1e-8)
        kv = (
            2.0
            * ACTUATOR_DAMPING_RATIO
            * math.sqrt(kp * inertia)
        )
        model.actuator_biasprm[actuator_id, 2] = -kv

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.tolerance = 1e-8
    model.opt.iterations = 50
    model.opt.ls_iterations = 20


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

        slide, bend_x, bend_y = k.compute_joint_targets()

        actuators = np.zeros(model.nu, dtype=np.float64)
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
# EXACT ORIGINAL 6-STATE VLA
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
        raise FileNotFoundError(f"VLA checkpoint not found:\n{VLA_PATH}")

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
        raise RuntimeError("VLA checkpoint contains no vocabulary.")

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )
    if state is None:
        raise RuntimeError("VLA checkpoint contains no model state.")

    state = clean_state_dict(state)

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

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))

    shape_errors = []
    for key in expected:
        if key in state:
            got = tuple(state[key].shape)
            wanted = tuple(expected[key].shape)
            if got != wanted:
                shape_errors.append((key, got, wanted))

    if missing or unexpected or shape_errors:
        msg = [
            "VLA checkpoint / architecture mismatch.",
            f"Missing keys: {len(missing)}",
            f"Unexpected keys: {len(unexpected)}",
            f"Shape errors: {len(shape_errors)}",
        ]
        if shape_errors:
            msg.append("First shape errors:")
            for key, got, wanted in shape_errors[:10]:
                msg.append(f"  {key}: checkpoint={got}, model={wanted}")
        raise RuntimeError("\n".join(msg))

    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    return model, vocab


class VLAResidualController:
    def __init__(self, model, vocab):
        self.model = model
        self.tokens = tokenize_instruction(INSTRUCTION, vocab)
        pad_id = vocab.get("<PAD>", 0)
        self.padding_mask = self.tokens == pad_id

    @torch.inference_mode()
    def predict(self, image, current, target, velocity):
        state = np.concatenate([current, velocity]).astype(np.float32)
        target_error = (target - current).astype(np.float32)

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
            normalized_delta.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        delta = np.clip(delta, -1.0, 1.0)
        return (delta * MAX_VLA_PRESSURE_DELTA).astype(np.float64)


# ============================================================================
# TEMPORAL INVERSE MLP
# ============================================================================

EXPECTED_FEATURE_NAMES = [
    "current_x", "current_y", "current_z",
    "target_x", "target_y", "target_z",
    "error_x", "error_y", "error_z",
    "velocity_x", "velocity_y", "velocity_z",
    "previous_pressure_1", "previous_pressure_2", "previous_pressure_3",
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
                    "Temporal MLP feature order mismatch."
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
        velocity = np.asarray(velocity, dtype=np.float64)
        previous_pressure = np.asarray(
            previous_pressure,
            dtype=np.float64,
        )

        error = target_position - current_position

        features = np.concatenate(
            [
                current_position,
                target_position,
                error,
                velocity,
                previous_pressure,
            ]
        )

        if features.shape != (15,):
            raise RuntimeError(
                f"Temporal MLP feature shape error: {features.shape}"
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
            normalized_pressure * self.target_std
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
# GENERIC FRENÉT LOCAL DLS
# ============================================================================

class GenericTrajectoryDLS:
    """
    Geometry-agnostic local pressure correction.

    The curve is represented only by:
        tangent T
        normal N = [-Ty, Tx]
        Z

    There is deliberately no circle radius term and no figure-8 term.
    """

    def __init__(self, model, mapper, site_id):
        self.model = model
        self.mapper = mapper
        self.site_id = site_id

        self.pred_data = mujoco.MjData(model)

        self.cached_J = None
        self.cached_nominal = None
        self.cached_prediction = None
        self.cached_time = -1e9

    def predict_position(self, source, pressure):
        mujoco.mj_copyData(
            self.pred_data,
            self.model,
            source,
        )

        self.pred_data.ctrl[:] = self.mapper.convert(
            self.model,
            pressure,
        )

        end_time = (
            self.pred_data.time
            + PREDICTION_HORIZON
        )

        while self.pred_data.time < end_time - 1e-12:
            mujoco.mj_step(
                self.model,
                self.pred_data,
            )

        return self.pred_data.site_xpos[self.site_id].copy()

    def finite_difference_jacobian(self, source, nominal):
        base = self.predict_position(source, nominal)

        J = np.zeros((3, 3), dtype=np.float64)

        for axis in range(3):
            plus = nominal.copy()
            minus = nominal.copy()

            plus[axis] = min(
                P_MAX,
                plus[axis] + JACOBIAN_PRESSURE_STEP,
            )
            minus[axis] = max(
                P_MIN,
                minus[axis] - JACOBIAN_PRESSURE_STEP,
            )

            denom = plus[axis] - minus[axis]
            if denom <= 1e-12:
                continue

            p_plus = self.predict_position(source, plus)
            p_minus = self.predict_position(source, minus)

            J[:, axis] = (p_plus - p_minus) / denom

        return base, J

    @staticmethod
    def frame(target, tangent):
        tangent_xy = np.asarray(
            tangent[:2],
            dtype=np.float64,
        )
        n = float(np.linalg.norm(tangent_xy))

        if n < 1e-10:
            tangent_xy = np.array([1.0, 0.0], dtype=np.float64)
            n = 1.0

        tangent_xy = tangent_xy / n
        normal_xy = np.array(
            [-tangent_xy[1], tangent_xy[0]],
            dtype=np.float64,
        )

        return tangent_xy, normal_xy

    def desired_correction(
        self,
        predicted,
        current,
        target,
        tangent,
    ):
        tangent_xy, normal_xy = self.frame(target, tangent)

        xy_error = target[:2] - predicted[:2]

        normal_error = float(
            np.dot(xy_error, normal_xy)
        )
        tangential_error = float(
            np.dot(xy_error, tangent_xy)
        )
        z_error = float(
            target[2] - predicted[2]
        )

        # Use the measured point error plus local Frenet decomposition.
        desired_xy = (
            POSITION_GAIN * (target[:2] - predicted[:2])
            + NORMAL_GAIN * normal_error * normal_xy
            + TANGENTIAL_GAIN * tangential_error * tangent_xy
        )

        desired = np.array(
            [
                desired_xy[0],
                desired_xy[1],
                Z_GAIN * z_error,
            ],
            dtype=np.float64,
        )

        norm = float(np.linalg.norm(desired))
        if norm > MAX_POSITION_CORRECTION:
            desired *= MAX_POSITION_CORRECTION / norm

        return desired

    @staticmethod
    def score(
        predicted,
        target,
        pressure,
        nominal,
        tangent,
    ):
        tangent_xy, normal_xy = GenericTrajectoryDLS.frame(
            target,
            tangent,
        )

        e = target - predicted
        point_error = float(np.linalg.norm(e))

        normal_error = abs(float(np.dot(e[:2], normal_xy)))
        tangential_error = abs(float(np.dot(e[:2], tangent_xy)))
        z_error = abs(float(e[2]))

        pressure_change = float(
            np.linalg.norm(pressure - nominal)
        )

        return (
            1.00 * point_error
            + 2.50 * normal_error
            + 0.90 * tangential_error
            + 1.50 * z_error
            + PRESSURE_CHANGE_WEIGHT * pressure_change
        )

    def correct(
        self,
        source,
        current,
        target,
        nominal,
        tangent,
        now,
    ):
        nominal = np.clip(
            np.asarray(nominal, dtype=np.float64),
            P_MIN,
            P_MAX,
        )

        refresh = (
            self.cached_J is None
            or self.cached_nominal is None
            or now - self.cached_time >= JACOBIAN_UPDATE_DT
        )

        if refresh:
            try:
                prediction, J_new = self.finite_difference_jacobian(
                    source,
                    nominal,
                )

                if np.all(np.isfinite(J_new)):
                    if self.cached_J is None:
                        J = J_new
                    else:
                        J = (
                            JACOBIAN_EMA * self.cached_J
                            + (1.0 - JACOBIAN_EMA) * J_new
                        )

                    self.cached_J = J
                    self.cached_nominal = nominal.copy()
                    self.cached_prediction = prediction.copy()
                    self.cached_time = now

            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                ValueError,
            ):
                pass

        if self.cached_J is None:
            return nominal

        J = self.cached_J

        dp = nominal - self.cached_nominal

        # Do not extrapolate a stale local model too far.
        if np.linalg.norm(dp) > 0.35:
            return nominal

        predicted_nominal = (
            self.cached_prediction
            + J @ dp
        )

        if not np.all(np.isfinite(predicted_nominal)):
            return nominal

        desired = self.desired_correction(
            predicted_nominal,
            current,
            target,
            tangent,
        )

        tangent_xy, normal_xy = self.frame(target, tangent)

        # Task Jacobian in natural curve coordinates [normal, tangent, Z].
        G = np.vstack(
            [
                normal_xy @ J[:2, :],
                tangent_xy @ J[:2, :],
                J[2, :],
            ]
        )

        if not np.all(np.isfinite(G)):
            return nominal

        W = np.diag(
            [
                2.5,
                1.25,
                1.50,
            ]
        )

        A = W @ G
        b = W @ np.array(
            [
                np.dot(desired[:2], normal_xy),
                np.dot(desired[:2], tangent_xy),
                desired[2],
            ],
            dtype=np.float64,
        )

        H = (
            A.T @ A
            + DLS_DAMPING * np.eye(3)
            + PRESSURE_CHANGE_WEIGHT * np.eye(3)
        )

        try:
            delta = np.linalg.solve(
                H,
                A.T @ b,
            )
        except np.linalg.LinAlgError:
            return nominal

        if not np.all(np.isfinite(delta)):
            return nominal

        dnorm = float(np.linalg.norm(delta))
        if dnorm > MAX_PRESSURE_CORRECTION:
            delta *= MAX_PRESSURE_CORRECTION / dnorm

        best_pressure = nominal.copy()
        best_prediction = predicted_nominal.copy()

        best_score = self.score(
            best_prediction,
            target,
            best_pressure,
            nominal,
            tangent,
        )

        # Trust-region selection using the same local Jacobian.
        for scale in CORRECTION_SCALES:
            trial = np.clip(
                nominal + scale * delta,
                P_MIN,
                P_MAX,
            )

            trial_prediction = (
                predicted_nominal
                + J @ (trial - nominal)
            )

            if not np.all(np.isfinite(trial_prediction)):
                continue

            trial_score = self.score(
                trial_prediction,
                target,
                trial,
                nominal,
                tangent,
            )

            if trial_score < best_score:
                best_score = trial_score
                best_pressure = trial

        return best_pressure


# ============================================================================
# PRESSURE DYNAMICS
# ============================================================================

def pressure_step(current, target):
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if PRESSURE_FILTER_TAU <= 1e-9:
        filtered = target.copy()
    else:
        alpha = 1.0 - math.exp(
            -CONTROL_DT / PRESSURE_FILTER_TAU
        )
        filtered = (
            current
            + alpha * (target - current)
        )

    max_change = MAX_PRESSURE_RATE * CONTROL_DT
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


def hold_tiny_pressure_changes(current, target):
    delta = target - current
    delta[np.abs(delta) < PRESSURE_DEADBAND] = 0.0
    return np.clip(
        current + delta,
        P_MIN,
        P_MAX,
    )


# ============================================================================
# LOGGER
# ============================================================================

class TrajectoryLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.file)

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
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
                "pressure_1_bar",
                "pressure_2_bar",
                "pressure_3_bar",
                "vla_d1_bar",
                "vla_d2_bar",
                "vla_d3_bar",
                "error_mm",
                "normal_error_mm",
                "tangential_error_mm",
                "z_error_mm",
                "accuracy_percent",
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
        normal_error_mm,
        tangential_error_mm,
    ):
        error = target - actual
        error_mm = float(np.linalg.norm(error) * 1000.0)

        accuracy = float(
            np.clip(
                100.0 - error_mm / 1.0,
                0.0,
                100.0,
            )
        )

        self.writer.writerow(
            [
                float(time_s),
                phase,
                float(progress),
                float(target[0]),
                float(target[1]),
                float(target[2]),
                float(actual[0]),
                float(actual[1]),
                float(actual[2]),
                float(velocity[0]),
                float(velocity[1]),
                float(velocity[2]),
                float(pressure[0]),
                float(pressure[1]),
                float(pressure[2]),
                float(vla_delta[0]),
                float(vla_delta[1]),
                float(vla_delta[2]),
                error_mm,
                float(normal_error_mm),
                float(tangential_error_mm),
                float((target[2] - actual[2]) * 1000.0),
                accuracy,
            ]
        )

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 88)
    print("NYU SOFT ROBOT — STANDALONE UNSEEN SUPERELLIPSE TEST")
    print("=" * 88)
    print("Figure-8 dependency : NONE")
    print("Figure-8 data       : NONE")
    print("Training data       : NONE")
    print("Test family         : rotated superellipse / rounded-square")
    print(
        f"Axes                : "
        f"{SEMI_AXIS_A * 1000.0:.1f} x "
        f"{SEMI_AXIS_B * 1000.0:.1f} mm"
    )
    print(f"Exponent            : {EXPONENT:.2f}")
    print(f"Rotation            : {ROTATION_DEG:.1f} deg")
    print(f"Animation           : {ANIMATION_SPEEDUP:.1f}x")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MuJoCo model not found:\n{MODEL_PATH}")

    mlp_path = find_mlp()
    vla, vocab = load_vla()
    inverse = TemporalInverseController(mlp_path)
    mapper = PressureMapper()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    stabilize_model(model)

    data = mujoco.MjData(model)

    if model.nu != 15:
        raise RuntimeError(
            f"Expected 15 actuators, found {model.nu}"
        )

    tip_site_id = find_site(
        model,
        ("end_effector", "tip", "ee", "end"),
    )
    if tip_site_id < 0:
        raise RuntimeError("Could not find end-effector site.")

    hide_xml_target(model)

    mujoco.mj_resetData(model, data)

    pressure = INITIAL_PRESSURE.copy()
    pressure_target = pressure.copy()

    data.ctrl[:] = mapper.convert(model, pressure)
    mujoco.mj_forward(model, data)

    settle_end = data.time + 1.0

    while data.time < settle_end - 1e-12:
        data.ctrl[:] = mapper.convert(model, pressure)
        mujoco.mj_step(model, data)

        if not finite_state(model, data):
            raise RuntimeError(
                "MuJoCo became unstable during startup."
            )

    initial_tip = tip_position(
        model,
        data,
        tip_site_id,
    )

    center_xy = initial_tip[:2].copy()
    test_z = float(initial_tip[2] + HEIGHT_ABOVE_TIP)

    # The curve starts at +X from its center. The approach ends exactly there.
    trajectory = SuperellipseTrajectory(
        z=test_z,
        center_xy=center_xy,
        a=SEMI_AXIS_A,
        b=SEMI_AXIS_B,
        exponent=EXPONENT,
        rotation_deg=ROTATION_DEG,
    )

    trajectory.radial_start = initial_tip.copy()
    trajectory.radial_start[2] = test_z

    # Shift the curve phase so the first curve point is the approach endpoint.
    # This changes only the test curve phase; it does not import any figure-8.
    trajectory.start = trajectory.points[0].copy()

    vla_controller = VLAResidualController(vla, vocab)

    renderer = mujoco.Renderer(
        model,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )

    corrector = GenericTrajectoryDLS(
        model,
        mapper,
        tip_site_id,
    )

    run_id = (
        "superellipse_unseen_standalone_"
        + time.strftime("%Y%m%d_%H%M%S")
    )

    run_dir = DESKTOP_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "trajectory.csv"
    xy_csv_path = run_dir / "superellipse_xy_tracking.csv"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    logger = TrajectoryLogger(csv_path)

    config = {
        "test": {
            "name": TEST_NAME,
            "family": "rotated_superellipse_rounded_square",
            "training_named_family": False,
            "figure8_dependency": False,
            "figure8_data_used": False,
            "semi_axis_a_m": SEMI_AXIS_A,
            "semi_axis_b_m": SEMI_AXIS_B,
            "exponent": EXPONENT,
            "rotation_deg": ROTATION_DEG,
        },
        "trajectory": {
            "center_xy_m": center_xy.tolist(),
            "z_m": test_z,
            "approach_time_s": APPROACH_TIME,
            "test_time_s": TEST_TIME,
            "final_hold_s": FINAL_HOLD,
            "arc_samples": ARC_SAMPLES,
        },
        "architecture": {
            "vla": "original 6-state SoftRobotVLA",
            "temporal_mlp": "15-feature temporal inverse MLP",
            "pressure_to_actuators": "Kinematics",
            "actuators": 15,
            "local_correction": "generic Frenet normal/tangent/Z bounded DLS",
            "circle_specific_correction": False,
            "figure8_specific_correction": False,
            "pid": False,
            "integral_accumulator": False,
        },
        "paths": {
            "model": str(MODEL_PATH),
            "vla": str(VLA_PATH),
            "temporal_mlp": str(mlp_path),
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
            "jacobian_pressure_step_bar": JACOBIAN_PRESSURE_STEP,
            "jacobian_update_dt_s": JACOBIAN_UPDATE_DT,
            "jacobian_ema": JACOBIAN_EMA,
            "dls_damping": DLS_DAMPING,
            "max_pressure_correction_bar": MAX_PRESSURE_CORRECTION,
        },
        "animation": {
            "speedup": ANIMATION_SPEEDUP,
            "render_every_n_steps": RENDER_EVERY_N_STEPS,
        },
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Project root :", ROOT)
    print("MuJoCo       :", MODEL_PATH)
    print("VLA          :", VLA_PATH)
    print("Temporal MLP :", mlp_path)
    print("Run folder   :", run_dir)
    print("Initial EE   :", np.round(initial_tip, 6))
    print("Test Z       :", round(test_z, 6))
    print()

    last_position = initial_tip.copy()
    velocity = np.zeros(3, dtype=np.float64)
    vla_delta = np.zeros(3, dtype=np.float64)

    next_control_update = data.time
    next_vla_update = data.time

    sim_start = float(data.time)
    wall_start = time.perf_counter()
    last_print_wall = wall_start

    target_history = []
    actual_history = []
    times = []
    errors = []
    accuracies = []

    completed = False
    final_hold_start = None
    physics_steps = 0

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        viewer.cam.distance = 0.34
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array(
            [center_xy[0], center_xy[1], test_z],
            dtype=np.float64,
        )

        while viewer.is_running():
            sim_time = float(data.time - sim_start)

            if sim_time <= trajectory.total_time:
                target = trajectory.position(sim_time)
                phase = trajectory.phase(sim_time)
                progress = float(
                    np.clip(
                        sim_time / trajectory.total_time,
                        0.0,
                        1.0,
                    )
                )
            else:
                target = trajectory.position(trajectory.total_time)
                phase = "hold"
                progress = 1.0
                completed = True

                if final_hold_start is None:
                    final_hold_start = data.time

            current = tip_position(
                model,
                data,
                tip_site_id,
            )

            raw_velocity = (
                current - last_position
            ) / CONTROL_DT

            velocity = (
                0.65 * velocity
                + 0.35 * raw_velocity
            )

            if sim_time >= next_control_update - 1e-12:
                future_time = min(
                    sim_time + LOOKAHEAD_TIME,
                    trajectory.total_time,
                )
                future_target = trajectory.position(future_time)
                tangent = trajectory.tangent(future_time)

                mlp_pressure = inverse.predict(
                    current_position=current,
                    target_position=future_target,
                    velocity=velocity,
                    previous_pressure=pressure,
                )

                if sim_time >= next_vla_update - 1e-12:
                    image = render_vla_image(renderer, data)
                    vla_delta = vla_controller.predict(
                        image=image,
                        current=current,
                        target=future_target,
                        velocity=velocity,
                    )

                    next_vla_update += VLA_UPDATE_DT
                    if next_vla_update < sim_time:
                        next_vla_update = (
                            sim_time + VLA_UPDATE_DT
                        )

                nominal_pressure = np.clip(
                    mlp_pressure
                    + VLA_RESIDUAL_WEIGHT * vla_delta,
                    P_MIN,
                    P_MAX,
                )

                pressure_target = corrector.correct(
                    source=data,
                    current=current,
                    target=future_target,
                    nominal=nominal_pressure,
                    tangent=tangent,
                    now=sim_time,
                )

                next_control_update += CONTROL_UPDATE_DT
                if next_control_update < sim_time:
                    next_control_update = (
                        sim_time + CONTROL_UPDATE_DT
                    )

                last_position = current.copy()

            pressure_target = hold_tiny_pressure_changes(
                pressure,
                pressure_target,
            )

            pressure = pressure_step(
                pressure,
                pressure_target,
            )

            data.ctrl[:] = mapper.convert(
                model,
                pressure,
            )

            mujoco.mj_step(model, data)
            physics_steps += 1

            if not finite_state(model, data):
                raise RuntimeError(
                    "MuJoCo became numerically unstable."
                )

            actual = tip_position(
                model,
                data,
                tip_site_id,
            )

            error_vec = target - actual
            error_mm = float(
                np.linalg.norm(error_vec) * 1000.0
            )

            # Tangent/normal diagnostics for the actual target curve.
            tangent = trajectory.tangent(
                min(sim_time, trajectory.total_time)
            )
            tangent_xy, normal_xy = corrector.frame(
                target,
                tangent,
            )

            xy_error = target[:2] - actual[:2]

            normal_error_mm = float(
                np.dot(xy_error, normal_xy) * 1000.0
            )

            tangential_error_mm = float(
                np.dot(xy_error, tangent_xy) * 1000.0
            )

            # Keep the historical project-style accuracy metric:
            # 100 - error_mm / 1 mm, bounded to [0,100].
            accuracy = float(
                np.clip(
                    100.0 - error_mm,
                    0.0,
                    100.0,
                )
            )

            times.append(sim_time)
            errors.append(error_mm)
            accuracies.append(accuracy)
            target_history.append(target.copy())
            actual_history.append(actual.copy())

            logger.log(
                sim_time,
                phase,
                progress,
                target,
                actual,
                velocity,
                pressure,
                vla_delta,
                normal_error_mm,
                tangential_error_mm,
            )

            if sim_time > trajectory.total_time + FINAL_HOLD:
                break

            now_wall = time.perf_counter()

            if now_wall - last_print_wall >= 0.25:
                curve_mask = np.asarray(times) >= APPROACH_TIME

                if np.any(curve_mask):
                    live_acc = float(
                        np.mean(
                            np.asarray(accuracies)[curve_mask]
                        )
                    )
                else:
                    live_acc = float(np.mean(accuracies))

                print(
                    f"\r{phase.upper():12s} "
                    f"{progress * 100:6.1f}% | "
                    f"Err {error_mm:7.3f} mm | "
                    f"Acc {live_acc:6.2f}% | "
                    f"Cross {normal_error_mm:+7.3f} mm | "
                    f"Z {(target[2]-actual[2])*1000.0:+7.3f} mm",
                    end="",
                    flush=True,
                )
                last_print_wall = now_wall

            target_wall = (
                wall_start
                + sim_time / ANIMATION_SPEEDUP
            )

            sleep_for = (
                target_wall
                - time.perf_counter()
            )

            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.003))

            if completed and final_hold_start is not None:
                if data.time - final_hold_start >= FINAL_HOLD:
                    break

            if (
                sim_time >= trajectory.total_time
                and final_hold_start is None
            ):
                final_hold_start = data.time

        final_target = trajectory.position(
            trajectory.total_time
        )
        final_actual = tip_position(
            model,
            data,
            tip_site_id,
        )

    print()

    logger.flush()
    logger.close()

    target_arr = np.asarray(target_history)
    actual_arr = np.asarray(actual_history)
    time_arr = np.asarray(times)
    error_arr = np.asarray(errors)
    acc_arr = np.asarray(accuracies)

    curve_mask = time_arr >= APPROACH_TIME

    if not np.any(curve_mask):
        raise RuntimeError(
            "No superellipse samples were recorded."
        )

    target_curve = target_arr[curve_mask]
    actual_curve = actual_arr[curve_mask]

    # Dedicated chronological XY CSV. No sorting by X or Y.
    with xy_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )

        for i in np.flatnonzero(curve_mask):
            writer.writerow(
                [
                    float(time_arr[i]),
                    float(target_arr[i, 0]),
                    float(target_arr[i, 1]),
                    float(target_arr[i, 2]),
                    float(actual_arr[i, 0]),
                    float(actual_arr[i, 1]),
                    float(actual_arr[i, 2]),
                    float(error_arr[i]),
                    float(acc_arr[i]),
                ]
            )

    target_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(target_curve[:, :2], axis=0),
                axis=1,
            )
        )
    )

    actual_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(actual_curve[:, :2], axis=0),
                axis=1,
            )
        )
    )

    final_error_mm = float(
        np.linalg.norm(final_target - final_actual) * 1000.0
    )

    summary = {
        "test_type": "unseen_named_trajectory_family",
        "trajectory_family": TEST_NAME,
        "training_named_family": False,
        "figure8_dependency": False,
        "figure8_data_used": False,
        "parameters": {
            "semi_axis_a_mm": SEMI_AXIS_A * 1000.0,
            "semi_axis_b_mm": SEMI_AXIS_B * 1000.0,
            "exponent": EXPONENT,
            "rotation_deg": ROTATION_DEG,
        },
        "samples": int(np.sum(curve_mask)),
        "tracking": {
            "mean_error_mm": float(np.mean(error_arr[curve_mask])),
            "rmse_mm": float(
                np.sqrt(np.mean(error_arr[curve_mask] ** 2))
            ),
            "p95_error_mm": float(
                np.percentile(error_arr[curve_mask], 95)
            ),
            "peak_error_mm": float(
                np.max(error_arr[curve_mask])
            ),
            "mean_accuracy_percent": float(
                np.mean(acc_arr[curve_mask])
            ),
            "samples_above_92_percent": float(
                100.0
                * np.mean(acc_arr[curve_mask] >= 92.0)
            ),
            "final_error_mm": final_error_mm,
        },
        "path_length_mm": {
            "target": target_path * 1000.0,
            "actual": actual_path * 1000.0,
            "ratio": (
                actual_path / target_path
                if target_path > 1e-12
                else None
            ),
        },
        "controller": {
            "original_6_state_vla": True,
            "temporal_inverse_mlp": True,
            "generic_frenet_dls": True,
            "circle_specific_logic": False,
            "figure8_specific_logic": False,
            "pid": False,
            "integral": False,
        },
        "physics_steps": physics_steps,
        "run_dir": str(run_dir),
        "files": {
            "trajectory_csv": str(csv_path),
            "xy_csv": str(xy_csv_path),
            "config_json": str(config_path),
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 88)
    print("STANDALONE UNSEEN SUPERELLIPSE TEST COMPLETE")
    print("=" * 88)
    print("Figure-8 imported     : NO")
    print("Figure-8 data used    : NO")
    print("Curve samples         :", int(np.sum(curve_mask)))
    print(
        "Mean curve error      :",
        f"{summary['tracking']['mean_error_mm']:.3f} mm",
    )
    print(
        "Mean curve accuracy   :",
        f"{summary['tracking']['mean_accuracy_percent']:.2f}%",
    )
    print(
        "P95 curve error       :",
        f"{summary['tracking']['p95_error_mm']:.3f} mm",
    )
    print("Run folder            :", run_dir)
    print("XY CSV                :", xy_csv_path)
    print("Summary               :", summary_path)
    print()


if __name__ == "__main__":
    main()
