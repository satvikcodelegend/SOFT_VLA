import os
import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "vla_data"
)

NPZ_PATH = os.path.join(
    DATASET_DIR,
    "curved_trajectories.npz"
)

MANIFEST_PATH = os.path.join(
    DATASET_DIR,
    "manifest.jsonl"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "soft_robot_vla"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pt"
)

LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "last_model.pt"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "training_metrics.json"
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

WINDOW_SIZE = 32

STRIDE = 4

PATIENCE = 8

GRAD_CLIP = 1.0

PRINT_EVERY_BATCH = 1

# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_DIM = 6

OUTPUT_DIM = 3

HIDDEN_DIM = 256

NUM_LAYERS = 3

DROPOUT = 0.10

# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class SoftRobotVLA(nn.Module):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ):

        super().__init__()

        layers = []

        layers.append(
            nn.Linear(
                input_dim,
                hidden_dim
            )
        )

        layers.append(
            nn.LayerNorm(hidden_dim)
        )

        layers.append(
            nn.GELU()
        )

        layers.append(
            nn.Dropout(dropout)
        )

        for _ in range(num_layers - 1):

            layers.append(
                nn.Linear(
                    hidden_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.LayerNorm(hidden_dim)
            )

            layers.append(
                nn.GELU()
            )

            layers.append(
                nn.Dropout(dropout)
            )

        layers.append(
            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# WINDOW DATASET
# ============================================================

class TrajectoryWindowDataset(Dataset):

    def __init__(
        self,
        positions,
        pressures,
        trajectory_indices,
        position_mean,
        position_std,
        pressure_mean,
        pressure_std,
        window_size=32,
        stride=4
    ):

        self.samples = []

        self.positions = positions
        self.pressures = pressures

        self.position_mean = position_mean
        self.position_std = position_std

        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std

        self.window_size = window_size

        self.stride = stride

        print()
        print("=" * 70)
        print("BUILDING TEMPORAL WINDOWS")
        print("=" * 70)

        print(
            "Trajectories:",
            len(trajectory_indices)
        )

        for trajectory_id in trajectory_indices:

            pos = positions[trajectory_id]

            press = pressures[trajectory_id]

            if len(pos) != len(press):
                continue

            if len(pos) < window_size:
                continue

            max_start = (
                len(pos)
                - window_size
            )

            for start in range(
                0,
                max_start + 1,
                stride
            ):

                end = (
                    start
                    + window_size
                )

                self.samples.append(
                    (
                        trajectory_id,
                        start,
                        end
                    )
                )

        print(
            "Windows:",
            len(self.samples)
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        trajectory_id, start, end = (
            self.samples[index]
        )

        positions = self.positions[
            trajectory_id
        ][start:end]

        pressures = self.pressures[
            trajectory_id
        ][start:end]

        positions = (
            positions
            - self.position_mean
        ) / self.position_std

        pressures = (
            pressures
            - self.pressure_mean
        ) / self.pressure_std

        # ----------------------------------------------------
        # Input:
        #   position x,y,z
        #   velocity dx,dy,dz
        #
        # Output:
        #   pressure x,y,z
        # ----------------------------------------------------

        velocity = np.zeros_like(
            positions
        )

        if len(positions) > 1:

            velocity[1:] = (
                positions[1:]
                - positions[:-1]
            )

        features = np.concatenate(
            [
                positions,
                velocity
            ],
            axis=1
        )

        # Predict pressure for final point
        #
        # This makes the task causal:
        # previous trajectory information
        # -> next pressure command.

        x = features[:-1]

        y = pressures[-1]

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y


# ============================================================
# LOAD NPZ
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING CURVED TRAJECTORY DATASET")
    print("=" * 70)

    print()

    print("Dataset:")
    print(NPZ_PATH)

    if not os.path.exists(NPZ_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{NPZ_PATH}\n"
        )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True
    )

    print()
    print("NPZ keys:")

    for key in data.files:

        value = data[key]

        print(
            f"  {key:22s} "
            f"shape={value.shape} "
            f"dtype={value.dtype}"
        )

    required = [
        "positions",
        "pressures",
        "trajectory_type",
        "difficulty",
        "split"
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:

        raise RuntimeError(
            "Dataset is missing required keys:\n"
            + str(missing)
        )

    positions = data["positions"]

    pressures = data["pressures"]

    trajectory_type = data[
        "trajectory_type"
    ]

    difficulty = data[
        "difficulty"
    ]

    split = data[
        "split"
    ]

    print()
    print(
        "Total trajectories:",
        len(positions)
    )

    return (
        positions,
        pressures,
        trajectory_type,
        difficulty,
        split
    )


# ============================================================
# COMPUTE NORMALIZATION
# ============================================================

def compute_normalization(
    positions,
    pressures,
    train_indices
):

    print()
    print("=" * 70)
    print("COMPUTING NORMALIZATION")
    print("=" * 70)

    position_values = []

    pressure_values = []

    for idx in train_indices:

        position_values.append(
            positions[idx]
        )

        pressure_values.append(
            pressures[idx]
        )

    position_values = np.concatenate(
        position_values,
        axis=0
    )

    pressure_values = np.concatenate(
        pressure_values,
        axis=0
    )

    position_mean = (
        position_values.mean(
            axis=0
        ).astype(np.float32)
    )

    position_std = (
        position_values.std(
            axis=0
        ).astype(np.float32)
    )

    pressure_mean = (
        pressure_values.mean(
            axis=0
        ).astype(np.float32)
    )

    pressure_std = (
        pressure_values.std(
            axis=0
        ).astype(np.float32)
    )

    position_std = np.maximum(
        position_std,
        1e-6
    )

    pressure_std = np.maximum(
        pressure_std,
        1e-6
    )

    print()
    print("Position mean:")
    print(position_mean)

    print()
    print("Position std:")
    print(position_std)

    print()
    print("Pressure mean:")
    print(pressure_mean)

    print()
    print("Pressure std:")
    print(pressure_std)

    return (
        position_mean,
        position_std,
        pressure_mean,
        pressure_std
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch,
    total_epochs
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    total_batches = len(loader)

    epoch_start = time.time()

    for batch_idx, (
        x,
        y
    ) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # x shape:
        #
        # [batch, sequence, 6]
        # ----------------------------------------------------

        batch_size = x.shape[0]

        sequence_length = x.shape[1]

        x = x.reshape(
            batch_size * sequence_length,
            INPUT_DIM
        )

        prediction = model(x)

        # ----------------------------------------------------
        # We use the LAST temporal input.
        # ----------------------------------------------------

        prediction = prediction.reshape(
            batch_size,
            sequence_length,
            OUTPUT_DIM
        )

        prediction = prediction[:, -1, :]

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_samples = (
            y.shape[0]
        )

        running_loss += (
            loss.item()
            * batch_samples
        )

        total_samples += batch_samples

        if (
            PRINT_EVERY_BATCH > 0
            and (
                batch_idx % PRINT_EVERY_BATCH == 0
                or batch_idx == total_batches - 1
            )
        ):

            elapsed = (
                time.time()
                - epoch_start
            )

            processed = (
                batch_idx + 1
            )

            rate = (
                processed
                / max(elapsed, 1e-6)
            )

            percent = (
                100.0
                * processed
                / total_batches
            )

            print(
                f"\r"
                f"Epoch "
                f"{epoch:03d}/{total_epochs:03d} "
                f"| Batch "
                f"{processed:04d}/{total_batches:04d} "
                f"({percent:6.2f}%) "
                f"| Loss "
                f"{loss.item():.6f} "
                f"| Avg "
                f"{running_loss / total_samples:.6f} "
                f"| {rate:.2f} batch/s",
                end="",
                flush=True
            )

    print()

    return (
        running_loss
        / max(total_samples, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    predictions = []

    targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            y = y.to(DEVICE)

            batch_size = x.shape[0]

            sequence_length = x.shape[1]

            x = x.reshape(
                batch_size * sequence_length,
                INPUT_DIM
            )

            prediction = model(x)

            prediction = prediction.reshape(
                batch_size,
                sequence_length,
                OUTPUT_DIM
            )

            prediction = prediction[:, -1, :]

            loss = criterion(
                prediction,
                y
            )

            batch_samples = y.shape[0]

            running_loss += (
                loss.item()
                * batch_samples
            )

            total_samples += batch_samples

            predictions.append(
                prediction.cpu()
            )

            targets.append(
                y.cpu()
            )

    predictions = torch.cat(
        predictions,
        dim=0
    )

    targets = torch.cat(
        targets,
        dim=0
    )

    loss = (
        running_loss
        / max(total_samples, 1)
    )

    return (
        loss,
        predictions,
        targets
    )


# ============================================================
# PRESSURE METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets,
    pressure_mean,
    pressure_std
):

    predictions = (
        predictions.numpy()
        * pressure_std
        + pressure_mean
    )

    targets = (
        targets.numpy()
        * pressure_std
        + pressure_mean
    )

    predictions = np.clip(
        predictions,
        0.0,
        3.0
    )

    error = (
        predictions
        - targets
    )

    abs_error = np.abs(
        error
    )

    mae = np.mean(
        abs_error,
        axis=0
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2,
            axis=0
        )
    )

    max_error = np.max(
        abs_error,
        axis=0
    )

    mean_output_error = np.mean(
        abs_error
    )

    percentile_95 = np.percentile(
        abs_error,
        95
    )

    return {
        "mae_bar": mae.tolist(),
        "rmse_bar": rmse.tolist(),
        "max_error_bar": max_error.tolist(),
        "mean_error_bar": float(
            mean_output_error
        ),
        "percentile_95_bar": float(
            percentile_95
        )
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    position_mean,
    position_std,
    pressure_mean,
    pressure_std
):

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "epoch":
            epoch,

        "best_val_loss":
            best_val_loss,

        "position_mean":
            position_mean,

        "position_std":
            position_std,

        "pressure_mean":
            pressure_mean,

        "pressure_std":
            pressure_std,

        "architecture": {

            "input_dim":
                INPUT_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "output_dim":
                OUTPUT_DIM,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT
        },

        "window_size":
            WINDOW_SIZE,

        "stride":
            STRIDE,

        "description":
            (
                "Curved trajectory temporal "
                "pressure prediction model"
            )
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SOFT ROBOT VLA TRAINING")
    print("=" * 70)

    print()

    print("Device:")
    print(DEVICE)

    print()

    print("Dataset:")
    print(NPZ_PATH)

    print()

    print("Window size:")
    print(WINDOW_SIZE)

    print()

    print("Window stride:")
    print(STRIDE)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        positions,
        pressures,
        trajectory_type,
        difficulty,
        split
    ) = load_dataset()

    # --------------------------------------------------------
    # Identify splits
    # --------------------------------------------------------

    train_indices = np.where(
        split == "train"
    )[0]

    val_indices = np.where(
        split == "validation"
    )[0]

    test_indices = np.where(
        split == "test"
    )[0]

    print()
    print("=" * 70)
    print("DATASET SPLITS")
    print("=" * 70)

    print()
    print(
        "Training trajectories:",
        len(train_indices)
    )

    print(
        "Validation trajectories:",
        len(val_indices)
    )

    print(
        "Test trajectories:",
        len(test_indices)
    )

    if len(train_indices) == 0:

        raise RuntimeError(
            "No training trajectories found."
        )

    if len(val_indices) == 0:

        raise RuntimeError(
            "No validation trajectories found."
        )

    # --------------------------------------------------------
    # Trajectory distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING TRAJECTORY DISTRIBUTION")
    print("=" * 70)

    unique_types, counts = np.unique(
        trajectory_type[train_indices],
        return_counts=True
    )

    for name, count in zip(
        unique_types,
        counts
    ):

        print(
            f"{name:20s}: {count:7d}"
        )

    # --------------------------------------------------------
    # Difficulty distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING DIFFICULTY DISTRIBUTION")
    print("=" * 70)

    unique_diff, diff_counts = np.unique(
        difficulty[train_indices],
        return_counts=True
    )

    for name, count in zip(
        unique_diff,
        diff_counts
    ):

        print(
            f"{name:20s}: {count:7d}"
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    (
        position_mean,
        position_std,
        pressure_mean,
        pressure_std
    ) = compute_normalization(
        positions,
        pressures,
        train_indices
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = TrajectoryWindowDataset(
        positions,
        pressures,
        train_indices,
        position_mean,
        position_std,
        pressure_mean,
        pressure_std,
        WINDOW_SIZE,
        STRIDE
    )

    val_dataset = TrajectoryWindowDataset(
        positions,
        pressures,
        val_indices,
        position_mean,
        position_std,
        pressure_mean,
        pressure_std,
        WINDOW_SIZE,
        STRIDE
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset contains zero windows."
        )

    if len(val_dataset) == 0:

        raise RuntimeError(
            "Validation dataset contains zero windows."
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SoftRobotVLA().to(
        DEVICE
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(model)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Trainable parameters:",
        parameter_count
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    criterion = nn.SmoothL1Loss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    metrics_history = []

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print()

    print(
        f"Training windows:   {len(train_dataset)}"
    )

    print(
        f"Validation windows: {len(val_dataset)}"
    )

    print(
        f"Batches / epoch:    {len(train_loader)}"
    )

    print(
        f"Batch size:         {BATCH_SIZE}"
    )

    print(
        f"Epochs:             {EPOCHS}"
    )

    print()

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            EPOCHS
        )

        (
            val_loss,
            predictions,
            targets
        ) = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        val_metrics = calculate_metrics(
            predictions,
            targets,
            pressure_mean,
            pressure_std
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} "
            f"| Train Loss: {train_loss:.8f} "
            f"| Val Loss: {val_loss:.8f} "
            f"| LR: {current_lr:.3e} "
            f"| Time: {epoch_time:.1f}s"
        )

        print(
            "  Validation MAE [bar]:",
            np.array(
                val_metrics["mae_bar"]
            )
        )

        print(
            "  Validation RMSE [bar]:",
            np.array(
                val_metrics["rmse_bar"]
            )
        )

        print(
            "  Validation mean error [bar]:",
            f"{val_metrics['mean_error_bar']:.6f}"
        )

        print(
            "  Validation 95th percentile [bar]:",
            f"{val_metrics['percentile_95_bar']:.6f}"
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "validation_loss": float(
                    val_loss
                ),
                "learning_rate": float(
                    current_lr
                ),
                "epoch_time_seconds": float(
                    epoch_time
                ),
                "validation_metrics":
                    val_metrics
            }
        )

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        save_checkpoint(
            LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            position_mean,
            position_std,
            pressure_mean,
            pressure_std
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                position_mean,
                position_std,
                pressure_mean,
                pressure_std
            )

            print()
            print(
                "  *** NEW BEST MODEL SAVED ***"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 70
            )

            print(
                "Best epoch:",
                best_epoch
            )

            print(
                "Best validation loss:",
                best_val_loss
            )

            break

    total_training_time = (
        time.time()
        - training_start
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    final_metrics = {

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            float(best_val_loss),

        "total_training_time_seconds":
            float(total_training_time),

        "device":
            str(DEVICE),

        "dataset":
            NPZ_PATH,

        "train_trajectories":
            int(len(train_indices)),

        "validation_trajectories":
            int(len(val_indices)),

        "test_trajectories":
            int(len(test_indices)),

        "train_windows":
            int(len(train_dataset)),

        "validation_windows":
            int(len(val_dataset)),

        "window_size":
            WINDOW_SIZE,

        "stride":
            STRIDE,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "architecture": {

            "input_dim":
                INPUT_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "output_dim":
                OUTPUT_DIM,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT
        },

        "position_mean":
            position_mean.tolist(),

        "position_std":
            position_std.tolist(),

        "pressure_mean":
            pressure_mean.tolist(),

        "pressure_std":
            pressure_std.tolist(),

        "history":
            metrics_history
    }

    with open(
        METRICS_PATH,
        "w"
    ) as f:

        json.dump(
            final_metrics,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Last model:"
    )

    print(
        LAST_MODEL_PATH
    )

    print()

    print(
        "Metrics:"
    )

    print(
        METRICS_PATH
    )

    print()

    print(
        "Total training time:",
        f"{total_training_time / 60.0:.2f} minutes"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()