import os
import json
import math
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

POSITION_DIM = 3
PRESSURE_DIM = 3
VELOCITY_DIM = 3

# ------------------------------------------------------------
# Temporal window
# ------------------------------------------------------------

DEFAULT_WINDOW_SIZE = 8
DEFAULT_HORIZON = 1

# ------------------------------------------------------------
# Pressure action scaling
#
# VLA predicts pressure DELTA.
#
# [-1, +1] -> [-MAX_PRESSURE_DELTA, +MAX_PRESSURE_DELTA]
# ------------------------------------------------------------

MAX_PRESSURE_DELTA = 0.015

PRESSURE_MIN = 0.0
PRESSURE_MAX = 3.0


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

DEFAULT_DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "vla_data"
)

DEFAULT_NPZ_PATH = os.path.join(
    DEFAULT_DATASET_ROOT,
    "curved_trajectories.npz"
)

DEFAULT_MANIFEST_PATH = os.path.join(
    DEFAULT_DATASET_ROOT,
    "manifest.jsonl"
)

DEFAULT_NORMALIZATION_PATH = os.path.join(
    DEFAULT_DATASET_ROOT,
    "normalization.json"
)


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def safe_std(
    value,
    minimum=1e-8
):

    value = np.asarray(
        value,
        dtype=np.float32
    )

    return np.maximum(
        np.abs(value),
        minimum
    )


def normalize(
    value,
    mean,
    std
):

    value = np.asarray(
        value,
        dtype=np.float32
    )

    return (
        value - mean
    ) / safe_std(std)


# ============================================================
# TOKENIZATION
# ============================================================

PAD_ID = 0
UNK_ID = 1

MAX_INSTRUCTION_LENGTH = 32


def build_vocabulary(
    instructions
):

    vocabulary = {
        "<PAD>": PAD_ID,
        "<UNK>": UNK_ID
    }

    for instruction in instructions:

        for word in instruction.lower().split():

            if word not in vocabulary:

                vocabulary[word] = len(
                    vocabulary
                )

    return vocabulary


def encode_instruction(
    text,
    vocabulary
):

    tokens = []

    for word in text.lower().split():

        tokens.append(
            vocabulary.get(
                word,
                UNK_ID
            )
        )

    tokens = tokens[
        :MAX_INSTRUCTION_LENGTH
    ]

    while len(tokens) < MAX_INSTRUCTION_LENGTH:

        tokens.append(
            PAD_ID
        )

    return np.asarray(
        tokens,
        dtype=np.int64
    )


# ============================================================
# IMAGE
# ============================================================

def make_fallback_image(
    position,
    target,
    velocity,
    trajectory_type
):
    """
    The curved trajectory generator currently stores trajectory
    state, not camera frames.

    Therefore the dataset provides a deterministic synthetic
    observation image rather than pretending that a real camera
    image exists.

    The image contains:
        - current position
        - target position
        - direction/error information

    This keeps the image branch of the VLA usable while the
    actual MuJoCo RGB dataset can be added later.
    """

    canvas = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Convert normalized workspace coordinates to pixels.
    # --------------------------------------------------------

    workspace_min = np.array(
        [-0.30, -0.30, 0.04],
        dtype=np.float32
    )

    workspace_max = np.array(
        [0.30, 0.30, 0.22],
        dtype=np.float32
    )

    def project(point):

        point = np.asarray(
            point,
            dtype=np.float32
        )

        u = (
            (point[0] - workspace_min[0])
            /
            (
                workspace_max[0]
                - workspace_min[0]
            )
        )

        v = (
            (point[2] - workspace_min[2])
            /
            (
                workspace_max[2]
                - workspace_min[2]
            )
        )

        px = int(
            np.clip(
                u,
                0.0,
                1.0
            )
            * (IMAGE_SIZE - 1)
        )

        py = int(
            (
                1.0
                -
                np.clip(
                    v,
                    0.0,
                    1.0
                )
            )
            * (IMAGE_SIZE - 1)
        )

        return px, py

    # --------------------------------------------------------
    # Draw current position.
    # --------------------------------------------------------

    px, py = project(
        position
    )

    radius = 5

    yy, xx = np.ogrid[
        :IMAGE_SIZE,
        :IMAGE_SIZE
    ]

    mask = (
        (xx - px) ** 2
        +
        (yy - py) ** 2
        <=
        radius ** 2
    )

    canvas[mask, 0] = 255

    # --------------------------------------------------------
    # Draw target.
    # --------------------------------------------------------

    tx, ty = project(
        target
    )

    radius = 5

    mask = (
        (xx - tx) ** 2
        +
        (yy - ty) ** 2
        <=
        radius ** 2
    )

    canvas[mask, 1] = 255

    # --------------------------------------------------------
    # Draw error direction.
    # --------------------------------------------------------

    steps = 20

    for alpha in np.linspace(
        0.0,
        1.0,
        steps
    ):

        x = int(
            px
            +
            alpha
            * (tx - px)
        )

        y = int(
            py
            +
            alpha
            * (ty - py)
        )

        if (
            0 <= x < IMAGE_SIZE
            and
            0 <= y < IMAGE_SIZE
        ):

            canvas[
                max(0, y - 1):
                min(IMAGE_SIZE, y + 2),
                max(0, x - 1):
                min(IMAGE_SIZE, x + 2),
                2
            ] = 255

    return torch.from_numpy(
        canvas
    ).permute(
        2,
        0,
        1
    ).float() / 255.0


# ============================================================
# CURVATURE
# ============================================================

def compute_curvature(
    points,
    index
):

    n = len(points)

    if n < 3:

        return 0.0

    i0 = max(
        0,
        index - 1
    )

    i1 = index

    i2 = min(
        n - 1,
        index + 1
    )

    p0 = points[i0]
    p1 = points[i1]
    p2 = points[i2]

    a = np.linalg.norm(
        p1 - p0
    )

    b = np.linalg.norm(
        p2 - p1
    )

    c = np.linalg.norm(
        p2 - p0
    )

    denominator = (
        a * b * c
    )

    if denominator < 1e-10:

        return 0.0

    cross = np.linalg.norm(
        np.cross(
            p1 - p0,
            p2 - p1
        )
    )

    curvature = (
        2.0
        * cross
        / denominator
    )

    return float(
        curvature
    )


# ============================================================
# TRAJECTORY PROCESSING
# ============================================================

def compute_velocity(
    positions
):

    velocity = np.zeros_like(
        positions,
        dtype=np.float32
    )

    if len(positions) <= 1:

        return velocity

    velocity[1:] = (
        positions[1:]
        -
        positions[:-1]
    )

    velocity[0] = velocity[1]

    return velocity


def compute_pressure_proxy(
    position
):
    """
    The new curved trajectory dataset does not contain a
    pressure sequence for every trajectory point.

    We therefore keep previous pressure as a neutral state
    unless the generator provides pressure explicitly.

    This function intentionally does NOT invent a physically
    incorrect pressure mapping.
    """

    return np.zeros(
        3,
        dtype=np.float32
    )


# ============================================================
# TRAJECTORY EXTRACTION
# ============================================================

def extract_position(
    trajectory
):

    if isinstance(
        trajectory,
        dict
    ):

        if "position" in trajectory:

            return np.asarray(
                trajectory["position"],
                dtype=np.float32
            )

        if "positions" in trajectory:

            return np.asarray(
                trajectory["positions"],
                dtype=np.float32
            )

    raise KeyError(
        "Trajectory does not contain 'position'."
    )


def extract_pressure(
    trajectory,
    index
):

    if not isinstance(
        trajectory,
        dict
    ):

        return np.zeros(
            3,
            dtype=np.float32
        )

    for key in (
        "pressure",
        "pressures",
        "action",
        "actions"
    ):

        if key in trajectory:

            values = np.asarray(
                trajectory[key],
                dtype=np.float32
            )

            if values.ndim == 2:

                if (
                    0 <= index < len(values)
                ):

                    return values[index]

    return compute_pressure_proxy(
        extract_position(trajectory)[index]
    )


# ============================================================
# DATASET
# ============================================================

class CurvedTrajectoryDataset(
    Dataset
):

    """
    Dataset for continuous curved-trajectory VLA training.

    IMPORTANT:

    The new dataset contains complete trajectories:

        trajectory
            -> position[time]
            -> optional pressure[time]
            -> metadata

    It does NOT contain one manifest line per frame.

    Therefore this class:

        1. loads complete trajectories
        2. selects trajectories by split
        3. generates temporal windows
        4. predicts the next pressure delta
    """

    def __init__(
        self,
        npz_path=DEFAULT_NPZ_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
        split="train",
        window_size=DEFAULT_WINDOW_SIZE,
        horizon=DEFAULT_HORIZON,
        normalization_path=DEFAULT_NORMALIZATION_PATH,
        seed=42
    ):

        super().__init__()

        self.npz_path = npz_path
        self.manifest_path = manifest_path
        self.split = split

        self.window_size = int(
            window_size
        )

        self.horizon = int(
            horizon
        )

        self.seed = seed

        if self.window_size < 1:

            raise ValueError(
                "window_size must be >= 1"
            )

        if self.horizon < 1:

            raise ValueError(
                "horizon must be >= 1"
            )

        # ----------------------------------------------------
        # LOAD NPZ
        # ----------------------------------------------------

        if not os.path.exists(
            self.npz_path
        ):

            raise FileNotFoundError(
                f"Trajectory dataset not found:\n"
                f"{self.npz_path}"
            )

        archive = np.load(
            self.npz_path,
            allow_pickle=True
        )

        if "trajectories" not in archive:

            raise RuntimeError(
                "NPZ does not contain "
                "'trajectories'."
            )

        raw_trajectories = archive[
            "trajectories"
        ]

        self.trajectories = list(
            raw_trajectories
        )

        # ----------------------------------------------------
        # LOAD MANIFEST
        # ----------------------------------------------------

        self.manifest = {}

        if os.path.exists(
            self.manifest_path
        ):

            with open(
                self.manifest_path,
                "r"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:

                        continue

                    record = json.loads(
                        line
                    )

                    record_id = int(
                        record["id"]
                    )

                    self.manifest[
                        record_id
                    ] = record

        # ----------------------------------------------------
        # SELECT TRAJECTORIES
        # ----------------------------------------------------

        self.trajectory_ids = []

        for i in range(
            len(self.trajectories)
        ):

            record = self.manifest.get(
                i,
                {}
            )

            record_split = record.get(
                "split",
                None
            )

            # Generator uses "val", not
            # "validation".

            if self.split == "validation":

                wanted_split = "val"

            else:

                wanted_split = self.split

            if record_split == wanted_split:

                self.trajectory_ids.append(
                    i
                )

        # ----------------------------------------------------
        # FALLBACK TO NPZ SPLITS
        # ----------------------------------------------------

        if len(
            self.trajectory_ids
        ) == 0:

            if "split_indices" in archive:

                split_indices = archive[
                    "split_indices"
                ].item()

                wanted_split = (
                    "val"
                    if self.split
                    == "validation"
                    else
                    self.split
                )

                if wanted_split in split_indices:

                    self.trajectory_ids = [
                        int(x)
                        for x in
                        split_indices[
                            wanted_split
                        ]
                    ]

        if len(
            self.trajectory_ids
        ) == 0:

            raise RuntimeError(
                f"No trajectories found "
                f"for split '{self.split}'."
            )

        # ----------------------------------------------------
        # BUILD WINDOWS
        # ----------------------------------------------------

        self.windows = []

        for trajectory_id in (
            self.trajectory_ids
        ):

            trajectory = self.trajectories[
                trajectory_id
            ]

            positions = extract_position(
                trajectory
            )

            if positions.ndim != 2:

                continue

            if positions.shape[1] != 3:

                continue

            num_points = len(
                positions
            )

            # Need:
            #
            # window_size current frames
            # +
            # horizon future frame

            last_start = (
                num_points
                -
                self.window_size
                -
                self.horizon
                + 1
            )

            if last_start <= 0:

                continue

            for start in range(
                last_start
            ):

                target_index = (
                    start
                    +
                    self.window_size
                    -
                    1
                    +
                    self.horizon
                )

                self.windows.append(
                    (
                        trajectory_id,
                        start,
                        target_index
                    )
                )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        self.position_mean = np.zeros(
            3,
            dtype=np.float32
        )

        self.position_std = np.ones(
            3,
            dtype=np.float32
        )

        self.velocity_mean = np.zeros(
            3,
            dtype=np.float32
        )

        self.velocity_std = np.ones(
            3,
            dtype=np.float32
        )

        self.error_mean = np.zeros(
            3,
            dtype=np.float32
        )

        self.error_std = np.ones(
            3,
            dtype=np.float32
        )

        self.pressure_mean = np.zeros(
            3,
            dtype=np.float32
        )

        self.pressure_std = np.ones(
            3,
            dtype=np.float32
        )

        if (
            normalization_path
            is not None
            and
            os.path.exists(
                normalization_path
            )
        ):

            with open(
                normalization_path,
                "r"
            ) as f:

                normalization = json.load(
                    f
                )

            self._load_normalization(
                normalization
            )

        # ----------------------------------------------------
        # DEFAULT INSTRUCTIONS
        # ----------------------------------------------------

        self.instructions = [
            "Track the curved trajectory accurately.",
            "Follow the target trajectory smoothly.",
            "Track the continuous curved path.",
            "Follow the trajectory with high accuracy.",
            "Maintain accurate smooth trajectory tracking."
        ]

        self.vocabulary = build_vocabulary(
            self.instructions
        )

        print()
        print("=" * 70)
        print(
            "CURVED TRAJECTORY DATASET"
        )
        print("=" * 70)

        print(
            "Split:",
            self.split
        )

        print(
            "Trajectories:",
            len(
                self.trajectory_ids
            )
        )

        print(
            "Training windows:",
            len(
                self.windows
            )
        )

        print(
            "Window size:",
            self.window_size
        )

        print(
            "Prediction horizon:",
            self.horizon
        )

        if len(self.windows) == 0:

            raise RuntimeError(
                "\nZERO TRAINING WINDOWS.\n"
                "Check trajectory point count and "
                "window_size/horizon."
            )

    # ========================================================
    # NORMALIZATION LOADER
    # ========================================================

    def _load_normalization(
        self,
        norm
    ):

        mapping = {

            "position_mean":
                "position_mean",

            "position_std":
                "position_std",

            "velocity_mean":
                "velocity_mean",

            "velocity_std":
                "velocity_std",

            "pressure_mean":
                "pressure_mean",

            "pressure_std":
                "pressure_std",

            "error_mean":
                "error_mean",

            "error_std":
                "error_std"
        }

        for source, target in (
            mapping.items()
        ):

            if source in norm:

                setattr(
                    self,
                    target,
                    np.asarray(
                        norm[source],
                        dtype=np.float32
                    )
                )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.windows
        )

    # ========================================================
    # GET TRAJECTORY
    # ========================================================

    def _get_trajectory(
        self,
        trajectory_id
    ):

        return self.trajectories[
            trajectory_id
        ]

    # ========================================================
    # GET METADATA
    # ========================================================

    def _get_metadata(
        self,
        trajectory_id
    ):

        return self.manifest.get(
            trajectory_id,
            {}
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        trajectory_id, start, target_index = (
            self.windows[index]
        )

        trajectory = self._get_trajectory(
            trajectory_id
        )

        metadata = self._get_metadata(
            trajectory_id
        )

        positions = extract_position(
            trajectory
        )

        velocities = compute_velocity(
            positions
        )

        # ----------------------------------------------------
        # Current frame
        # ----------------------------------------------------

        current_index = (
            start
            +
            self.window_size
            -
            1
        )

        current_position = positions[
            current_index
        ]

        target_position = positions[
            target_index
        ]

        velocity = velocities[
            current_index
        ]

        # ----------------------------------------------------
        # Target error
        # ----------------------------------------------------

        target_error = (
            target_position
            -
            current_position
        )

        # ----------------------------------------------------
        # Curvature
        # ----------------------------------------------------

        curvature = compute_curvature(
            positions,
            current_index
        )

        # ----------------------------------------------------
        # Previous pressure
        # ----------------------------------------------------

        previous_pressure = extract_pressure(
            trajectory,
            current_index
        )

        # ----------------------------------------------------
        # Target pressure if available
        # ----------------------------------------------------

        target_pressure = extract_pressure(
            trajectory,
            target_index
        )

        pressure_delta = (
            target_pressure
            -
            previous_pressure
        )

        # ----------------------------------------------------
        # If the trajectory generator does not
        # contain pressure, action becomes zero.
        #
        # We do NOT fabricate pressure targets.
        # ----------------------------------------------------

        if (
            "pressure" not in trajectory
            and
            "pressures" not in trajectory
            and
            "action" not in trajectory
            and
            "actions" not in trajectory
        ):

            pressure_delta = np.zeros(
                3,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Normalize action.
        # ----------------------------------------------------

        action = (
            pressure_delta
            /
            MAX_PRESSURE_DELTA
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # ----------------------------------------------------
        # Normalize state.
        # ----------------------------------------------------

        position_n = normalize(
            current_position,
            self.position_mean,
            self.position_std
        )

        velocity_n = normalize(
            velocity,
            self.velocity_mean,
            self.velocity_std
        )

        target_error_n = normalize(
            target_error,
            self.error_mean,
            self.error_std
        )

        previous_pressure_n = normalize(
            previous_pressure,
            self.pressure_mean,
            self.pressure_std
        )

        # ----------------------------------------------------
        # State expected by current VLA:
        #
        # position (3)
        # pressure (3)
        #
        # = 6
        # ----------------------------------------------------

        state = np.concatenate(
            [
                position_n,
                previous_pressure_n
            ]
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Trajectory feature vector.
        #
        # Current position
        # Target position
        # Velocity
        # Target error
        # Curvature
        # Normalized progress
        # Previous pressure
        # ----------------------------------------------------

        num_points = len(
            positions
        )

        progress = (
            current_index
            /
            max(
                num_points - 1,
                1
            )
        )

        trajectory_features = np.concatenate(
            [
                position_n,
                normalize(
                    target_position,
                    self.position_mean,
                    self.position_std
                ),
                velocity_n,
                target_error_n,
                np.asarray(
                    [curvature],
                    dtype=np.float32
                ),
                np.asarray(
                    [progress],
                    dtype=np.float32
                ),
                previous_pressure_n
            ]
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Synthetic visual observation.
        # ----------------------------------------------------

        image = make_fallback_image(
            current_position,
            target_position,
            velocity,
            metadata.get(
                "trajectory_type",
                "curved"
            )
        )

        # ----------------------------------------------------
        # Language.
        # ----------------------------------------------------

        trajectory_type = metadata.get(
            "trajectory_type",
            "curved"
        )

        difficulty = metadata.get(
            "difficulty",
            "unknown"
        )

        instruction = (
            f"Track the {trajectory_type} "
            f"trajectory smoothly and accurately."
        )

        instruction_tokens = torch.from_numpy(
            encode_instruction(
                instruction,
                self.vocabulary
            )
        )

        padding_mask = (
            instruction_tokens == PAD_ID
        )

        # ----------------------------------------------------
        # Output.
        # ----------------------------------------------------

        return {

            "image":
                image,

            "instruction_tokens":
                instruction_tokens,

            "padding_mask":
                padding_mask,

            "instruction":
                instruction,

            "state":
                torch.from_numpy(
                    state
                ),

            "position":
                torch.from_numpy(
                    position_n
                ),

            "target":
                torch.from_numpy(
                    normalize(
                        target_position,
                        self.position_mean,
                        self.position_std
                    )
                ),

            "target_error":
                torch.from_numpy(
                    target_error_n
                ),

            "velocity":
                torch.from_numpy(
                    velocity_n
                ),

            "trajectory_features":
                torch.from_numpy(
                    trajectory_features
                ),

            "previous_pressure":
                torch.from_numpy(
                    previous_pressure_n
                ),

            "action":
                torch.from_numpy(
                    action.astype(
                        np.float32
                    )
                ),

            "trajectory_type":
                trajectory_type,

            "difficulty":
                difficulty,

            "trajectory_id":
                torch.tensor(
                    trajectory_id,
                    dtype=torch.long
                ),

            "frame_index":
                torch.tensor(
                    current_index,
                    dtype=torch.long
                ),

            "target_index":
                torch.tensor(
                    target_index,
                    dtype=torch.long
                )
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def curved_collate_fn(
    batch
):

    result = {

        "image":
            torch.stack(
                [
                    x["image"]
                    for x in batch
                ]
            ),

        "instruction_tokens":
            torch.stack(
                [
                    x["instruction_tokens"]
                    for x in batch
                ]
            ),

        "padding_mask":
            torch.stack(
                [
                    x["padding_mask"]
                    for x in batch
                ]
            ),

        "state":
            torch.stack(
                [
                    x["state"]
                    for x in batch
                ]
            ),

        "position":
            torch.stack(
                [
                    x["position"]
                    for x in batch
                ]
            ),

        "target":
            torch.stack(
                [
                    x["target"]
                    for x in batch
                ]
            ),

        "target_error":
            torch.stack(
                [
                    x["target_error"]
                    for x in batch
                ]
            ),

        "velocity":
            torch.stack(
                [
                    x["velocity"]
                    for x in batch
                ]
            ),

        "trajectory_features":
            torch.stack(
                [
                    x["trajectory_features"]
                    for x in batch
                ]
            ),

        "previous_pressure":
            torch.stack(
                [
                    x["previous_pressure"]
                    for x in batch
                ]
            ),

        "action":
            torch.stack(
                [
                    x["action"]
                    for x in batch
                ]
            ),

        "trajectory_type":
            [
                x["trajectory_type"]
                for x in batch
            ],

        "difficulty":
            [
                x["difficulty"]
                for x in batch
            ],

        "trajectory_id":
            torch.stack(
                [
                    x["trajectory_id"]
                    for x in batch
                ]
            ),

        "frame_index":
            torch.stack(
                [
                    x["frame_index"]
                    for x in batch
                ]
            ),

        "target_index":
            torch.stack(
                [
                    x["target_index"]
                    for x in batch
                ]
            )
    }

    result["instruction"] = [
        x["instruction"]
        for x in batch
    ]

    return result


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TESTING CURVED TRAJECTORY DATASET")
    print("=" * 70)

    dataset = CurvedTrajectoryDataset(
        npz_path=DEFAULT_NPZ_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
        split="train",
        window_size=8,
        horizon=1
    )

    print()
    print(
        "Dataset length:",
        len(dataset)
    )

    sample = dataset[0]

    print()
    print("=" * 70)
    print("FIRST SAMPLE")
    print("=" * 70)

    for key, value in sample.items():

        if torch.is_tensor(value):

            print(
                f"{key:25s}: "
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )

        else:

            print(
                f"{key:25s}: "
                f"{value}"
            )

    print()
    print("=" * 70)
    print("BATCH TEST")
    print("=" * 70)

    batch = curved_collate_fn(
        [
            dataset[0],
            dataset[1],
            dataset[2],
            dataset[3]
        ]
    )

    for key, value in batch.items():

        if torch.is_tensor(value):

            print(
                f"{key:25s}: "
                f"shape={tuple(value.shape)}"
            )

        else:

            print(
                f"{key:25s}: "
                f"type={type(value).__name__}"
            )

    print()
    print("=" * 70)
    print("DATASET TEST PASSED")
    print("=" * 70)