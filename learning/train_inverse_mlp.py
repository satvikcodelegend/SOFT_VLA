import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


DATASET_PATH = (
    "data/inverse_temporal_dataset.npz"
)

MODEL_PATH = (
    "models/temporal_inverse_mlp.pt"
)

METRICS_PATH = (
    "models/temporal_inverse_metrics.json"
)

EPOCHS = 150

BATCH_SIZE = 512

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-5

PATIENCE = 20

SEED = 42


torch.manual_seed(SEED)
np.random.seed(SEED)


class TemporalInverseMLP(
    nn.Module
):

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

    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


def load_dataset():

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

    features = np.asarray(
        data["features"],
        dtype=np.float32
    )

    targets = np.asarray(
        data["targets"],
        dtype=np.float32
    )

    split = np.asarray(
        data["split"]
    )

    trajectory_id = np.asarray(
        data["trajectory_id"]
    )

    if features.ndim != 2:
        raise ValueError(
            "features must be 2D."
        )

    if features.shape[1] != 15:
        raise ValueError(
            f"Expected 15 features, "
            f"got {features.shape[1]}."
        )

    if targets.shape[1] != 3:
        raise ValueError(
            "targets must have shape (N,3)."
        )

    if len(features) != len(targets):
        raise ValueError(
            "Feature/target length mismatch."
        )

    if len(split) != len(features):
        raise ValueError(
            "Split length mismatch."
        )

    if len(trajectory_id) != len(features):
        raise ValueError(
            "Trajectory ID length mismatch."
        )

    return (
        data,
        features,
        targets,
        split,
        trajectory_id
    )


def calculate_statistics(
    features,
    targets,
    train_mask
):

    train_features = (
        features[train_mask]
    )

    train_targets = (
        targets[train_mask]
    )

    feature_mean = (
        train_features.mean(
            axis=0
        )
    )

    feature_std = (
        train_features.std(
            axis=0
        )
    )

    target_mean = (
        train_targets.mean(
            axis=0
        )
    )

    target_std = (
        train_targets.std(
            axis=0
        )
    )

    feature_std = np.where(
        feature_std < 1e-8,
        1.0,
        feature_std
    )

    target_std = np.where(
        target_std < 1e-8,
        1.0,
        target_std
    )

    return (
        feature_mean,
        feature_std,
        target_mean,
        target_std
    )


def make_loader(
    x,
    y,
    shuffle
):

    dataset = TensorDataset(
        torch.tensor(
            x,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.float32
        )
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )


def evaluate(
    model,
    loader,
    device,
    target_mean,
    target_std
):

    model.eval()

    predictions = []
    actual = []

    total_loss = 0.0
    count = 0

    criterion = nn.SmoothL1Loss()

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(
                x
            )

            loss = criterion(
                output,
                y
            )

            batch_size = (
                x.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            count += batch_size

            predictions.append(
                output.cpu().numpy()
            )

            actual.append(
                y.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    actual = np.concatenate(
        actual,
        axis=0
    )

    predictions = (
        predictions
        * target_std
        + target_mean
    )

    actual = (
        actual
        * target_std
        + target_mean
    )

    error = (
        predictions
        - actual
    )

    mae = np.mean(
        np.abs(error),
        axis=0
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2,
            axis=0
        )
    )

    magnitude = np.linalg.norm(
        error,
        axis=1
    )

    mean_error = np.mean(
        magnitude
    )

    median_error = np.median(
        magnitude
    )

    p95_error = np.percentile(
        magnitude,
        95
    )

    return {
        "loss": total_loss / max(
            count,
            1
        ),
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "median_error": median_error,
        "p95_error": p95_error
    }


def main():

    start_time = time.time()

    os.makedirs(
        os.path.dirname(
            MODEL_PATH
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(
            METRICS_PATH
        ),
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 72)
    print("TEMPORAL INVERSE MLP TRAINING")
    print("=" * 72)
    print()
    print(
        "Device:",
        device
    )

    (
        data,
        features,
        targets,
        split,
        trajectory_id
    ) = load_dataset()

    train_mask = (
        split == "train"
    )

    val_mask = (
        split == "val"
    )

    test_mask = (
        split == "test"
    )

    print()
    print(
        "Dataset:",
        DATASET_PATH
    )

    print(
        "Features:",
        features.shape
    )

    print(
        "Targets:",
        targets.shape
    )

    print(
        "Train:",
        np.sum(train_mask)
    )

    print(
        "Validation:",
        np.sum(val_mask)
    )

    print(
        "Test:",
        np.sum(test_mask)
    )

    train_trajectories = np.unique(
        trajectory_id[
            train_mask
        ]
    )

    val_trajectories = np.unique(
        trajectory_id[
            val_mask
        ]
    )

    test_trajectories = np.unique(
        trajectory_id[
            test_mask
        ]
    )

    if (
        len(
            np.intersect1d(
                train_trajectories,
                val_trajectories
            )
        )
        != 0
    ):

        raise RuntimeError(
            "Trajectory leakage between "
            "train and validation."
        )

    if (
        len(
            np.intersect1d(
                train_trajectories,
                test_trajectories
            )
        )
        != 0
    ):

        raise RuntimeError(
            "Trajectory leakage between "
            "train and test."
        )

    if (
        len(
            np.intersect1d(
                val_trajectories,
                test_trajectories
            )
        )
        != 0
    ):

        raise RuntimeError(
            "Trajectory leakage between "
            "validation and test."
        )

    (
        feature_mean,
        feature_std,
        target_mean,
        target_std
    ) = calculate_statistics(
        features,
        targets,
        train_mask
    )

    normalized_features = (
        features
        - feature_mean
    ) / feature_std

    normalized_targets = (
        targets
        - target_mean
    ) / target_std

    train_loader = make_loader(
        normalized_features[
            train_mask
        ],
        normalized_targets[
            train_mask
        ],
        True
    )

    val_loader = make_loader(
        normalized_features[
            val_mask
        ],
        normalized_targets[
            val_mask
        ],
        False
    )

    test_loader = make_loader(
        normalized_features[
            test_mask
        ],
        normalized_targets[
            test_mask
        ],
        False
    )

    model = TemporalInverseMLP().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )

    criterion = nn.SmoothL1Loss(
        beta=0.5
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1

    patience_counter = 0

    history = []

    print()
    print(
        model
    )

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                x
            )

            loss = criterion(
                output,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = (
                x.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            sample_count += (
                batch_size
            )

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            target_mean,
            target_std
        )

        val_loss = (
            val_metrics["loss"]
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_pressure_error": float(
                    val_metrics[
                        "mean_error"
                    ]
                ),
                "val_median_pressure_error": float(
                    val_metrics[
                        "median_error"
                    ]
                ),
                "val_p95_pressure_error": float(
                    val_metrics[
                        "p95_error"
                    ]
                ),
                "lr": current_lr
            }
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"| Train {train_loss:.8f} "
            f"| Val {val_loss:.8f} "
            f"| Median "
            f"{val_metrics['median_error']:.6f} bar "
            f"| P95 "
            f"{val_metrics['p95_error']:.6f} bar "
            f"| LR {current_lr:.3e}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            patience_counter = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "feature_mean":
                    feature_mean,

                "feature_std":
                    feature_std,

                "target_mean":
                    target_mean,

                "target_std":
                    target_std,

                "input_dim":
                    15,

                "output_dim":
                    3,

                "architecture":
                    "15-256-256-128-64-3",

                "feature_names":
                    data[
                        "feature_names"
                    ].tolist()
                    if "feature_names"
                    in data
                    else None,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss
            }

            torch.save(
                checkpoint,
                MODEL_PATH
            )

        else:

            patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):

                print()
                print(
                    "Early stopping."
                )

                break

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        target_mean,
        target_std
    )

    validation_metrics = evaluate(
        model,
        val_loader,
        device,
        target_mean,
        target_std
    )

    elapsed = (
        time.time()
        - start_time
    )

    metrics = {
        "dataset_path":
            DATASET_PATH,

        "model_path":
            MODEL_PATH,

        "input_dim":
            15,

        "output_dim":
            3,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,

        "training_time_seconds":
            elapsed,

        "validation": {
            "loss":
                float(
                    validation_metrics[
                        "loss"
                    ]
                ),
            "mae_bar":
                validation_metrics[
                    "mae"
                ].tolist(),
            "rmse_bar":
                validation_metrics[
                    "rmse"
                ].tolist(),
            "mean_error_bar":
                float(
                    validation_metrics[
                        "mean_error"
                    ]
                ),
            "median_error_bar":
                float(
                    validation_metrics[
                        "median_error"
                    ]
                ),
            "p95_error_bar":
                float(
                    validation_metrics[
                        "p95_error"
                    ]
                )
        },

        "test": {
            "loss":
                float(
                    test_metrics[
                        "loss"
                    ]
                ),
            "mae_bar":
                test_metrics[
                    "mae"
                ].tolist(),
            "rmse_bar":
                test_metrics[
                    "rmse"
                ].tolist(),
            "mean_error_bar":
                float(
                    test_metrics[
                        "mean_error"
                    ]
                ),
            "median_error_bar":
                float(
                    test_metrics[
                        "median_error"
                    ]
                ),
            "p95_error_bar":
                float(
                    test_metrics[
                        "p95_error"
                    ]
                )
        }
    }

    with open(
        METRICS_PATH,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

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
        "Validation MAE [bar]:",
        validation_metrics[
            "mae"
        ]
    )

    print(
        "Validation RMSE [bar]:",
        validation_metrics[
            "rmse"
        ]
    )

    print(
        "Validation median pressure error [bar]:",
        validation_metrics[
            "median_error"
        ]
    )

    print(
        "Validation 95th percentile [bar]:",
        validation_metrics[
            "p95_error"
        ]
    )

    print()
    print(
        "Test MAE [bar]:",
        test_metrics[
            "mae"
        ]
    )

    print(
        "Test RMSE [bar]:",
        test_metrics[
            "rmse"
        ]
    )

    print(
        "Test median pressure error [bar]:",
        test_metrics[
            "median_error"
        ]
    )

    print(
        "Test 95th percentile [bar]:",
        test_metrics[
            "p95_error"
        ]
    )

    print()
    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Metrics:",
        METRICS_PATH
    )

    print(
        "Time:",
        elapsed,
        "seconds"
    )

    print()


if __name__ == "__main__":
    main()