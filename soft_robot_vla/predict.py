import os
import numpy as np
import torch

from PIL import Image

from model import SoftRobotVLA
from dataset import load_image


# ============================================================
# PATHS
# ============================================================

VLA_MODEL_PATH = (
    "models/soft_robot_vla/best_model.pt"
)

INVERSE_MLP_PATH = (
    "model/inverse_mlp.pt"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 224

MIN_PRESSURE = 0.0
MAX_PRESSURE = 3.0


# ============================================================
# INVERSE MLP
# ============================================================

class InverseMLP(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.network = torch.nn.Sequential(

            torch.nn.Linear(
                3,
                128
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                128,
                128
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                128,
                64
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                64,
                3
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD INVERSE MLP
# ============================================================

def load_inverse_mlp():

    checkpoint = torch.load(
        INVERSE_MLP_PATH,
        map_location=DEVICE
    )

    model = InverseMLP()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    normalization = {

        "position_mean":
            np.asarray(
                checkpoint["position_mean"],
                dtype=np.float32
            ),

        "position_std":
            np.asarray(
                checkpoint["position_std"],
                dtype=np.float32
            ),

        "pressure_mean":
            np.asarray(
                checkpoint["pressure_mean"],
                dtype=np.float32
            ),

        "pressure_std":
            np.asarray(
                checkpoint["pressure_std"],
                dtype=np.float32
            ),
    }

    return model, normalization


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    x,
    mean,
    std
):

    return (
        x - mean
    ) / np.maximum(
        std,
        1e-8
    )


def denormalize(
    x,
    mean,
    std
):

    return (
        x * std
        + mean
    )


# ============================================================
# IMAGE
# ============================================================

def prepare_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# VLA CONTROLLER
# ============================================================

class VLAPredictor:

    def __init__(
        self,
        vla_model_path=VLA_MODEL_PATH,
        inverse_mlp_path=INVERSE_MLP_PATH,
    ):

        print("=" * 70)
        print("LOADING VLA CONTROLLER")
        print("=" * 70)

        print()
        print("Device:")
        print(DEVICE)

        # ----------------------------------------------------
        # VLA
        # ----------------------------------------------------

        checkpoint = torch.load(
            vla_model_path,
            map_location=DEVICE
        )

        self.vla = SoftRobotVLA(
            checkpoint=checkpoint
        )

        self.vla.to(
            DEVICE
        )

        self.vla.eval()

        print()
        print("VLA loaded.")

        # ----------------------------------------------------
        # INVERSE MLP
        # ----------------------------------------------------

        self.inverse_mlp, self.inverse_norm = (
            load_inverse_mlp()
        )

        print(
            "Inverse MLP loaded."
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.previous_pressure = (
            np.zeros(
                3,
                dtype=np.float32
            )
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.previous_pressure = (
            np.zeros(
                3,
                dtype=np.float32
            )
        )

    # ========================================================
    # INVERSE KINEMATICS
    # ========================================================

    def position_to_pressure(
        self,
        target_position
    ):

        target_position = np.asarray(
            target_position,
            dtype=np.float32
        )

        target_normalized = normalize(
            target_position,
            self.inverse_norm[
                "position_mean"
            ],
            self.inverse_norm[
                "position_std"
            ]
        )

        x = torch.tensor(
            target_normalized,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():

            pressure_normalized = (
                self.inverse_mlp(x)
                .cpu()
                .numpy()[0]
            )

        pressure = denormalize(
            pressure_normalized,
            self.inverse_norm[
                "pressure_mean"
            ],
            self.inverse_norm[
                "pressure_std"
            ]
        )

        pressure = np.clip(
            pressure,
            MIN_PRESSURE,
            MAX_PRESSURE
        )

        return pressure.astype(
            np.float32
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        image_path,
        current_position,
        next_waypoint,
        velocity,
        curvature,
        normalized_time,
        instruction=(
            "Track the curved trajectory "
            "accurately."
        ),
    ):

        current_position = np.asarray(
            current_position,
            dtype=np.float32
        )

        next_waypoint = np.asarray(
            next_waypoint,
            dtype=np.float32
        )

        velocity = np.asarray(
            velocity,
            dtype=np.float32
        )

        target_error = (
            next_waypoint
            - current_position
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = prepare_image(
            image_path
        )

        # ----------------------------------------------------
        # TENSORS
        # ----------------------------------------------------

        position = torch.tensor(
            current_position,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        waypoint = torch.tensor(
            next_waypoint,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        error = torch.tensor(
            target_error,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        velocity_tensor = torch.tensor(
            velocity,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        previous_pressure = torch.tensor(
            self.previous_pressure,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        curvature_tensor = torch.tensor(
            [[curvature]],
            dtype=torch.float32,
            device=DEVICE
        )

        time_tensor = torch.tensor(
            [[normalized_time]],
            dtype=torch.float32,
            device=DEVICE
        )

        # ----------------------------------------------------
        # VLA
        # ----------------------------------------------------

        with torch.no_grad():

            output = self.vla(
                image=image,
                instruction=[
                    instruction
                ],
                position=position,
                waypoint=waypoint,
                target_error=error,
                velocity=velocity_tensor,
                previous_pressure=(
                    previous_pressure
                ),
                curvature=curvature_tensor,
                time=time_tensor,
            )

        # ----------------------------------------------------
        # HANDLE MODEL OUTPUT
        # ----------------------------------------------------

        if isinstance(
            output,
            dict
        ):

            if "action" in output:

                action = output["action"]

            elif "delta_position" in output:

                action = (
                    output["delta_position"]
                )

            elif "correction" in output:

                action = output["correction"]

            else:

                raise KeyError(
                    "VLA output dictionary does not "
                    "contain action/delta_position/correction."
                )

        else:

            action = output

        action = (
            action.detach()
            .cpu()
            .numpy()[0]
        )

        # ----------------------------------------------------
        # TARGET POSITION
        # ----------------------------------------------------

        predicted_position = (
            next_waypoint
            + action
        )

        # ----------------------------------------------------
        # INVERSE MLP
        # ----------------------------------------------------

        pressure = (
            self.position_to_pressure(
                predicted_position
            )
        )

        # ----------------------------------------------------
        # STATE UPDATE
        # ----------------------------------------------------

        self.previous_pressure = (
            pressure.copy()
        )

        return {

            "action": action,

            "target_error": target_error,

            "predicted_position":
                predicted_position,

            "pressure":
                pressure,

        }


# ============================================================
# SINGLE PREDICTION TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VLA PREDICTION TEST")
    print("=" * 70)

    predictor = VLAPredictor()

    # --------------------------------------------------------
    # CHANGE THIS IMAGE
    # --------------------------------------------------------

    image_path = (
        "dataset/vla_data/example.png"
    )

    if not os.path.exists(
        image_path
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            f"Image not found:\n{image_path}"
        )

        print(
            "Change image_path in main()."
        )

        return

    current_position = np.array(
        [
            0.0,
            0.0,
            0.10
        ],
        dtype=np.float32
    )

    next_waypoint = np.array(
        [
            0.01,
            0.005,
            0.105
        ],
        dtype=np.float32
    )

    velocity = np.array(
        [
            0.0,
            0.0,
            0.0
        ],
        dtype=np.float32
    )

    curvature = 0.0

    normalized_time = 0.0

    result = predictor.predict(
        image_path=image_path,
        current_position=current_position,
        next_waypoint=next_waypoint,
        velocity=velocity,
        curvature=curvature,
        normalized_time=normalized_time,
    )

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print()
    print("Current position:")
    print(current_position)

    print()
    print("Next waypoint:")
    print(next_waypoint)

    print()
    print("VLA correction:")
    print(result["action"])

    print()
    print("Predicted target position:")
    print(result["predicted_position"])

    print()
    print("Pressure:")
    print(result["pressure"])

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()