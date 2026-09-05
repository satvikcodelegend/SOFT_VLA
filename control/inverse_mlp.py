"""
Temporal inverse MLP controller.

Input:
    15 features

    [ current_x, current_y, current_z,
      target_x,  target_y,  target_z,
      error_x,   error_y,   error_z,
      velocity_x, velocity_y, velocity_z,
      previous_pressure_1,
      previous_pressure_2,
      previous_pressure_3 ]

Output:
    [pressure_1, pressure_2, pressure_3]

The checkpoint is:
    models/temporal_inverse_mlp.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# TEMPORAL INVERSE MLP
# ============================================================

class TemporalInverseMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                15,
                256
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                256
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                64
            ),

            nn.GELU(),

            nn.Linear(
                64,
                3
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# INVERSE CONTROLLER
# ============================================================

class InverseController:

    FEATURE_NAMES = [
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

    def __init__(
        self,
        model_path="models/temporal_inverse_mlp.pt",
        pressure_min=0.0,
        pressure_max=3.0
    ):

        self.device = torch.device("cpu")

        self.pressure_min = float(
            pressure_min
        )

        self.pressure_max = float(
            pressure_max
        )

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                "Temporal inverse MLP not found:\n"
                f"{model_path}"
            )

        self.model_path = model_path

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        input_dim = int(
            checkpoint.get(
                "input_dim",
                15
            )
        )

        output_dim = int(
            checkpoint.get(
                "output_dim",
                3
            )
        )

        if input_dim != 15:

            raise RuntimeError(
                "Temporal inverse model must have "
                f"15 inputs, checkpoint has {input_dim}."
            )

        if output_dim != 3:

            raise RuntimeError(
                "Temporal inverse model must have "
                f"3 outputs, checkpoint has {output_dim}."
            )

        architecture = checkpoint.get(
            "architecture",
            ""
        )

        if architecture != "15-256-256-128-64-3":

            raise RuntimeError(
                "Unexpected temporal inverse architecture:\n"
                f"{architecture}"
            )

        # ----------------------------------------------------
        # Build exact network
        # ----------------------------------------------------

        self.model = (
            TemporalInverseMLP()
            .to(self.device)
        )

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        state_dict = {
            (
                key[7:]
                if key.startswith("module.")
                else key
            ): value

            for key, value in state_dict.items()
        }

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.eval()

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        required = [
            "feature_mean",
            "feature_std",
            "target_mean",
            "target_std"
        ]

        for name in required:

            if name not in checkpoint:

                raise RuntimeError(
                    "Checkpoint missing "
                    f"'{name}'."
                )

        self.feature_mean = np.asarray(
            checkpoint[
                "feature_mean"
            ],
            dtype=np.float64
        )

        self.feature_std = np.asarray(
            checkpoint[
                "feature_std"
            ],
            dtype=np.float64
        )

        self.target_mean = np.asarray(
            checkpoint[
                "target_mean"
            ],
            dtype=np.float64
        )

        self.target_std = np.asarray(
            checkpoint[
                "target_std"
            ],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Validate normalization shapes
        # ----------------------------------------------------

        if self.feature_mean.shape != (15,):

            raise RuntimeError(
                "feature_mean must have shape (15,), "
                f"got {self.feature_mean.shape}"
            )

        if self.feature_std.shape != (15,):

            raise RuntimeError(
                "feature_std must have shape (15,), "
                f"got {self.feature_std.shape}"
            )

        if self.target_mean.shape != (3,):

            raise RuntimeError(
                "target_mean must have shape (3,), "
                f"got {self.target_mean.shape}"
            )

        if self.target_std.shape != (3,):

            raise RuntimeError(
                "target_std must have shape (3,), "
                f"got {self.target_std.shape}"
            )

        self.feature_std = np.where(
            np.abs(self.feature_std) < 1e-8,
            1.0,
            self.feature_std
        )

        self.target_std = np.where(
            np.abs(self.target_std) < 1e-8,
            1.0,
            self.target_std
        )

        # ----------------------------------------------------
        # Validate feature ordering
        # ----------------------------------------------------

        checkpoint_features = checkpoint.get(
            "feature_names",
            None
        )

        if checkpoint_features is not None:

            checkpoint_features = [
                str(x)
                for x in checkpoint_features
            ]

            if (
                checkpoint_features
                != self.FEATURE_NAMES
            ):

                raise RuntimeError(
                    "Checkpoint feature ordering does not "
                    "match controller feature ordering.\n\n"
                    f"Checkpoint:\n{checkpoint_features}\n\n"
                    f"Expected:\n{self.FEATURE_NAMES}"
                )

        print()
        print("=" * 70)
        print("TEMPORAL INVERSE MLP LOADED")
        print("=" * 70)

        print(
            "Model:",
            self.model_path
        )

        print(
            "Architecture:",
            architecture
        )

        print(
            "Input features:",
            input_dim
        )

        print(
            "Output:",
            output_dim
        )

        print(
            "Temporal inputs:",
            "velocity + previous pressure"
        )

        print("=" * 70)

    # ========================================================
    # FEATURE CONSTRUCTION
    # ========================================================

    def make_features(
        self,
        current_position,
        target_position,
        velocity,
        previous_pressure
    ):

        current_position = np.asarray(
            current_position,
            dtype=np.float64
        )

        target_position = np.asarray(
            target_position,
            dtype=np.float64
        )

        velocity = np.asarray(
            velocity,
            dtype=np.float64
        )

        previous_pressure = np.asarray(
            previous_pressure,
            dtype=np.float64
        )

        if current_position.shape != (3,):

            raise ValueError(
                "current_position must have shape (3,)"
            )

        if target_position.shape != (3,):

            raise ValueError(
                "target_position must have shape (3,)"
            )

        if velocity.shape != (3,):

            raise ValueError(
                "velocity must have shape (3,)"
            )

        if previous_pressure.shape != (3,):

            raise ValueError(
                "previous_pressure must have shape (3,)"
            )

        if not np.all(
            np.isfinite(current_position)
        ):

            raise ValueError(
                "current_position contains NaN/Inf."
            )

        if not np.all(
            np.isfinite(target_position)
        ):

            raise ValueError(
                "target_position contains NaN/Inf."
            )

        if not np.all(
            np.isfinite(velocity)
        ):

            raise ValueError(
                "velocity contains NaN/Inf."
            )

        if not np.all(
            np.isfinite(previous_pressure)
        ):

            raise ValueError(
                "previous_pressure contains NaN/Inf."
            )

        error = (
            target_position
            -
            current_position
        )

        features = np.concatenate(
            [
                current_position,
                target_position,
                error,
                velocity,
                previous_pressure
            ]
        )

        if features.shape != (15,):

            raise RuntimeError(
                "Constructed feature vector does not "
                "have shape (15,)."
            )

        return features

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_pressure(
        self,
        current_position,
        target_position,
        velocity,
        previous_pressure
    ):

        features = self.make_features(
            current_position,
            target_position,
            velocity,
            previous_pressure
        )

        normalized_features = (
            features
            -
            self.feature_mean
        ) / self.feature_std

        x = torch.as_tensor(
            normalized_features,
            dtype=torch.float32,
            device=self.device
        ).reshape(
            1,
            15
        )

        with torch.inference_mode():

            normalized_pressure = (
                self.model(x)
                .cpu()
                .numpy()
                .reshape(3)
            )

        pressure = (
            normalized_pressure
            *
            self.target_std
            +
            self.target_mean
        )

        pressure = np.nan_to_num(
            pressure,
            nan=1.5,
            posinf=self.pressure_max,
            neginf=self.pressure_min
        )

        pressure = np.clip(
            pressure,
            self.pressure_min,
            self.pressure_max
        )

        return pressure.astype(
            np.float64
        )

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        features
    ):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.ndim != 2:

            raise ValueError(
                "features must be 2D."
            )

        if features.shape[1] != 15:

            raise ValueError(
                "features must have 15 columns."
            )

        normalized = (
            features
            -
            self.feature_mean.astype(
                np.float32
            )
        ) / self.feature_std.astype(
            np.float32
        )

        x = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device
        )

        with torch.inference_mode():

            output = (
                self.model(x)
                .cpu()
                .numpy()
            )

        pressure = (
            output
            *
            self.target_std
            +
            self.target_mean
        )

        pressure = np.nan_to_num(
            pressure,
            nan=1.5,
            posinf=self.pressure_max,
            neginf=self.pressure_min
        )

        return np.clip(
            pressure,
            self.pressure_min,
            self.pressure_max
        ).astype(
            np.float64
        )