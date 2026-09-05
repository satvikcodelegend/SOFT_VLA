import os
import numpy as np
import mujoco

from control.kinematics import Kinematics


MODEL_PATH = "model/scene.xml"
OUTPUT_PATH = "data/inverse_temporal_dataset.npz"

SEED = 42

NUM_TRAJECTORIES = 600
STEPS_PER_TRAJECTORY = 250

PRESSURE_MIN = 0.0
PRESSURE_MAX = 3.0

DT = 0.01

PRESSURE_STEP_MIN = 0.015
PRESSURE_STEP_MAX = 0.080

TARGET_LOOKAHEAD = 3

MIN_POSITION_CHANGE = 1e-7

WARMUP_STEPS = 20


def pressure_to_actuators(model, robot, pressure):

    pressure = np.asarray(
        pressure,
        dtype=np.float64
    )

    robot.pressure_to_strain(pressure)
    robot.compute_mean_strain()
    robot.compute_extension()
    robot.compute_curvature()

    slide_targets, bend_x_targets, bend_y_targets = (
        robot.compute_joint_targets()
    )

    actuators = np.zeros(
        model.nu,
        dtype=np.float64
    )

    n = min(
        model.nu // 3,
        len(slide_targets),
        len(bend_x_targets),
        len(bend_y_targets)
    )

    for i in range(n):

        j = 3 * i

        actuators[j] = bend_x_targets[i]
        actuators[j + 1] = bend_y_targets[i]
        actuators[j + 2] = slide_targets[i]

    return actuators


def find_control_geom(model):

    mesh_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_MESH,
        "end_mesh"
    )

    if mesh_id < 0:
        raise RuntimeError(
            "Mesh 'end_mesh' not found."
        )

    for geom_id in range(model.ngeom):

        if (
            model.geom_type[geom_id]
            == mujoco.mjtGeom.mjGEOM_MESH
            and
            model.geom_dataid[geom_id]
            == mesh_id
        ):

            return geom_id

    raise RuntimeError(
        "No geom using mesh 'end_mesh' found."
    )


def get_position(model, data, geom_id):

    return data.geom_xpos[
        geom_id
    ].copy()


def reset_simulation(model, data):

    mujoco.mj_resetData(
        model,
        data
    )

    mujoco.mj_forward(
        model,
        data
    )


def make_pressure_step(rng):

    direction = rng.normal(
        0.0,
        1.0,
        size=3
    )

    direction /= (
        np.linalg.norm(direction)
        + 1e-12
    )

    magnitude = rng.uniform(
        PRESSURE_STEP_MIN,
        PRESSURE_STEP_MAX
    )

    return direction * magnitude


def generate_pressure_sequence(rng):

    pressure = rng.uniform(
        0.25,
        2.75,
        size=3
    )

    sequence = np.zeros(
        (
            STEPS_PER_TRAJECTORY,
            3
        ),
        dtype=np.float64
    )

    sequence[0] = pressure

    for t in range(
        1,
        STEPS_PER_TRAJECTORY
    ):

        step = make_pressure_step(
            rng
        )

        pressure = (
            pressure
            + step
        )

        pressure = np.clip(
            pressure,
            PRESSURE_MIN,
            PRESSURE_MAX
        )

        sequence[t] = pressure

    return sequence


def simulate_trajectory(
    model,
    data,
    robot,
    geom_id,
    pressure_sequence
):

    positions = []

    for pressure in pressure_sequence:

        actuators = pressure_to_actuators(
            model,
            robot,
            pressure
        )

        data.ctrl[:] = actuators

        target_time = (
            data.time + DT
        )

        while data.time < target_time:

            mujoco.mj_step(
                model,
                data
            )

        positions.append(
            get_position(
                model,
                data,
                geom_id
            )
        )

    return np.asarray(
        positions,
        dtype=np.float64
    )


def build_samples(
    pressure_sequence,
    positions,
    trajectory_id
):

    features = []
    targets = []

    current_positions = []
    target_positions = []
    previous_positions = []
    errors = []
    velocities = []
    previous_pressures = []
    output_pressures = []

    max_t = (
        len(pressure_sequence)
        - TARGET_LOOKAHEAD
        - 1
    )

    for t in range(
        WARMUP_STEPS,
        max_t
    ):

        current = positions[t]

        previous = positions[t - 1]

        target = positions[
            t + TARGET_LOOKAHEAD
        ]

        velocity = (
            current - previous
        ) / DT

        error = (
            target - current
        )

        previous_pressure = (
            pressure_sequence[t - 1]
        )

        output_pressure = (
            pressure_sequence[t]
        )

        if (
            np.linalg.norm(error)
            < MIN_POSITION_CHANGE
        ):

            continue

        feature = np.concatenate(
            [
                current,
                target,
                error,
                velocity,
                previous_pressure
            ]
        )

        features.append(
            feature
        )

        targets.append(
            output_pressure
        )

        current_positions.append(
            current
        )

        target_positions.append(
            target
        )

        previous_positions.append(
            previous
        )

        errors.append(
            error
        )

        velocities.append(
            velocity
        )

        previous_pressures.append(
            previous_pressure
        )

        output_pressures.append(
            output_pressure
        )

    return {
        "features": np.asarray(
            features,
            dtype=np.float32
        ),
        "targets": np.asarray(
            targets,
            dtype=np.float32
        ),
        "current_position": np.asarray(
            current_positions,
            dtype=np.float32
        ),
        "target_position": np.asarray(
            target_positions,
            dtype=np.float32
        ),
        "previous_position": np.asarray(
            previous_positions,
            dtype=np.float32
        ),
        "position_error": np.asarray(
            errors,
            dtype=np.float32
        ),
        "velocity": np.asarray(
            velocities,
            dtype=np.float32
        ),
        "previous_pressure": np.asarray(
            previous_pressures,
            dtype=np.float32
        ),
        "pressure": np.asarray(
            output_pressures,
            dtype=np.float32
        ),
        "trajectory_id": np.full(
            len(features),
            trajectory_id,
            dtype=np.int32
        ),
        "timestep": np.arange(
            WARMUP_STEPS,
            WARMUP_STEPS + len(features),
            dtype=np.int32
        )
    }


def concatenate(results):

    keys = results[0].keys()

    output = {}

    for key in keys:

        output[key] = np.concatenate(
            [
                item[key]
                for item in results
            ],
            axis=0
        )

    return output


def main():

    rng = np.random.default_rng(
        SEED
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(
        model
    )

    robot = Kinematics()

    geom_id = find_control_geom(
        model
    )

    print()
    print("=" * 70)
    print("TEMPORAL INVERSE DATASET GENERATION")
    print("=" * 70)
    print()
    print(
        "Model:",
        MODEL_PATH
    )
    print(
        "Trajectories:",
        NUM_TRAJECTORIES
    )
    print(
        "Steps / trajectory:",
        STEPS_PER_TRAJECTORY
    )
    print(
        "MuJoCo timestep:",
        model.opt.timestep
    )
    print(
        "Control interval:",
        DT
    )
    print(
        "Input dimension:",
        15
    )
    print(
        "Output dimension:",
        3
    )
    print()

    results = []

    for trajectory_id in range(
        NUM_TRAJECTORIES
    ):

        reset_simulation(
            model,
            data
        )

        pressure_sequence = (
            generate_pressure_sequence(
                rng
            )
        )

        positions = simulate_trajectory(
            model,
            data,
            robot,
            geom_id,
            pressure_sequence
        )

        sample_data = build_samples(
            pressure_sequence,
            positions,
            trajectory_id
        )

        results.append(
            sample_data
        )

        if (
            trajectory_id + 1
        ) % 10 == 0:

            print(
                f"Trajectory "
                f"{trajectory_id + 1:4d}/"
                f"{NUM_TRAJECTORIES}"
            )

    dataset = concatenate(
        results
    )

    n = len(
        dataset["features"]
    )

    trajectory_ids = dataset[
        "trajectory_id"
    ]

    print()
    print(
        "Total samples:",
        n
    )

    print(
        "Features shape:",
        dataset["features"].shape
    )

    print(
        "Targets shape:",
        dataset["targets"].shape
    )

    print(
        "Position shape:",
        dataset["position"].shape
        if "position" in dataset
        else "not present"
    )

    dataset["position"] = (
        dataset["current_position"]
        .copy()
    )

    dataset["pressure"] = (
        dataset["targets"]
        .copy()
    )

    unique_ids = np.unique(
        trajectory_ids
    )

    rng.shuffle(
        unique_ids
    )

    n_traj = len(
        unique_ids
    )

    n_train = int(
        0.70 * n_traj
    )

    n_val = int(
        0.15 * n_traj
    )

    train_ids = unique_ids[
        :n_train
    ]

    val_ids = unique_ids[
        n_train:
        n_train + n_val
    ]

    test_ids = unique_ids[
        n_train + n_val:
    ]

    split = np.full(
        n,
        "test",
        dtype="<U5"
    )

    split[
        np.isin(
            trajectory_ids,
            train_ids
        )
    ] = "train"

    split[
        np.isin(
            trajectory_ids,
            val_ids
        )
    ] = "val"

    dataset["split"] = split

    dataset["train_trajectory_ids"] = (
        train_ids.astype(np.int32)
    )

    dataset["val_trajectory_ids"] = (
        val_ids.astype(np.int32)
    )

    dataset["test_trajectory_ids"] = (
        test_ids.astype(np.int32)
    )

    dataset["feature_names"] = np.asarray(
        [
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
            "previous_pressure_3"
        ]
    )

    np.savez_compressed(
        OUTPUT_PATH,
        **dataset
    )

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)
    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )
    print(
        "Samples:",
        n
    )
    print(
        "Train samples:",
        np.sum(
            split == "train"
        )
    )
    print(
        "Validation samples:",
        np.sum(
            split == "val"
        )
    )
    print(
        "Test samples:",
        np.sum(
            split == "test"
        )
    )
    print()
    print(
        "Train trajectories:",
        len(train_ids)
    )
    print(
        "Validation trajectories:",
        len(val_ids)
    )
    print(
        "Test trajectories:",
        len(test_ids)
    )
    print()
    print(
        "Feature names:"
    )

    for i, name in enumerate(
        dataset["feature_names"]
    ):

        print(
            f"{i:2d}: {name}"
        )

    print()


if __name__ == "__main__":
    main()