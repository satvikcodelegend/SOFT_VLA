import os
import numpy as np


# ============================================================
# GENERAL TRAJECTORY DATASET GENERATOR
# ============================================================
#
# Purpose:
#
# Generate training trajectories for a GENERAL
# dynamic / residual pressure controller.
#
# This is NOT the inverse MLP dataset.
#
# Existing inverse MLP:
#
#       position -> pressure
#
# New correction model:
#
#       current state + target state + history
#                         |
#                         v
#                    delta pressure
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_PATH = "data/trajectory_correction_dataset.npz"

SEED = 42

CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ


# ============================================================
# WORKSPACE
# ============================================================

WORKSPACE_CENTER = np.array(
    [0.0, 0.0, 0.200],
    dtype=np.float32
)

WORKSPACE_X = 0.070
WORKSPACE_Y = 0.070
WORKSPACE_Z = 0.035


# ============================================================
# TRAJECTORY SETTINGS
# ============================================================

TRAJECTORY_DURATION = 30.0

NUMBER_OF_TRAJECTORIES = 100

MIN_SPEED_SCALE = 0.50
MAX_SPEED_SCALE = 1.50


# ============================================================
# HISTORY / LOOKAHEAD
# ============================================================

LOOKAHEAD_TIME = 0.10

PREVIOUS_PRESSURE_DELAY = 0.05

LOOKAHEAD_STEPS = max(
    1,
    int(round(
        LOOKAHEAD_TIME * CONTROL_HZ
    ))
)

PRESSURE_HISTORY_STEPS = max(
    1,
    int(round(
        PREVIOUS_PRESSURE_DELAY * CONTROL_HZ
    ))
)


# ============================================================
# PRESSURE RANGE
# ============================================================

MIN_PRESSURE = 0.0
MAX_PRESSURE = 3.0


# ============================================================
# RANDOM
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# TRAJECTORY GENERATORS
# ============================================================

def smoothstep(t):
    """
    Smooth interpolation from 0 to 1.
    """

    return (
        3.0 * t ** 2
        - 2.0 * t ** 3
    )


def generate_line(
    n,
    speed_scale
):
    """
    Smooth straight-line trajectory.
    """

    start = WORKSPACE_CENTER + np.array([
        rng.uniform(
            -0.5 * WORKSPACE_X,
            0.5 * WORKSPACE_X
        ),
        rng.uniform(
            -0.5 * WORKSPACE_Y,
            0.5 * WORKSPACE_Y
        ),
        rng.uniform(
            -0.5 * WORKSPACE_Z,
            0.5 * WORKSPACE_Z
        )
    ])

    end = WORKSPACE_CENTER + np.array([
        rng.uniform(
            -0.5 * WORKSPACE_X,
            0.5 * WORKSPACE_X
        ),
        rng.uniform(
            -0.5 * WORKSPACE_Y,
            0.5 * WORKSPACE_Y
        ),
        rng.uniform(
            -0.5 * WORKSPACE_Z,
            0.5 * WORKSPACE_Z
        )
    ])

    u = np.linspace(
        0.0,
        1.0,
        n
    )

    s = smoothstep(u)

    trajectory = (
        start[None, :]
        +
        (
            end - start
        )[None, :]
        * s[:, None]
    )

    return trajectory


def generate_circle(
    n,
    speed_scale
):
    """
    General 3D circle.
    """

    center = WORKSPACE_CENTER + np.array([
        rng.uniform(
            -0.20 * WORKSPACE_X,
            0.20 * WORKSPACE_X
        ),
        rng.uniform(
            -0.20 * WORKSPACE_Y,
            0.20 * WORKSPACE_Y
        ),
        rng.uniform(
            -0.25 * WORKSPACE_Z,
            0.25 * WORKSPACE_Z
        )
    ])

    radius = rng.uniform(
        0.025,
        min(
            WORKSPACE_X,
            WORKSPACE_Y
        ) * 0.90
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    direction = rng.choice([
        -1.0,
        1.0
    ])

    angle = np.linspace(
        0.0,
        direction * 2.0 * np.pi,
        n
    )

    angle += phase

    # Randomly choose XY, XZ or YZ plane.
    plane = rng.choice([
        "xy",
        "xz",
        "yz"
    ])

    trajectory = np.tile(
        center,
        (n, 1)
    )

    if plane == "xy":

        trajectory[:, 0] += (
            radius * np.cos(angle)
        )

        trajectory[:, 1] += (
            radius * np.sin(angle)
        )

    elif plane == "xz":

        trajectory[:, 0] += (
            radius * np.cos(angle)
        )

        trajectory[:, 2] += (
            radius * np.sin(angle)
        )

    else:

        trajectory[:, 1] += (
            radius * np.cos(angle)
        )

        trajectory[:, 2] += (
            radius * np.sin(angle)
        )

    return trajectory


def generate_ellipse(
    n,
    speed_scale
):
    """
    General ellipse.
    """

    center = WORKSPACE_CENTER.copy()

    center += np.array([
        rng.uniform(
            -0.25 * WORKSPACE_X,
            0.25 * WORKSPACE_X
        ),
        rng.uniform(
            -0.25 * WORKSPACE_Y,
            0.25 * WORKSPACE_Y
        ),
        rng.uniform(
            -0.20 * WORKSPACE_Z,
            0.20 * WORKSPACE_Z
        )
    ])

    rx = rng.uniform(
        0.025,
        WORKSPACE_X * 0.85
    )

    ry = rng.uniform(
        0.020,
        WORKSPACE_Y * 0.85
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    direction = rng.choice([
        -1.0,
        1.0
    ])

    angle = np.linspace(
        0.0,
        direction * 2.0 * np.pi,
        n
    )

    angle += phase

    trajectory = np.tile(
        center,
        (n, 1)
    )

    trajectory[:, 0] += (
        rx * np.cos(angle)
    )

    trajectory[:, 1] += (
        ry * np.sin(angle)
    )

    return trajectory


def generate_figure_eight(
    n,
    speed_scale
):
    """
    Figure-eight trajectory.
    """

    center = WORKSPACE_CENTER.copy()

    scale_x = rng.uniform(
        0.025,
        WORKSPACE_X * 0.75
    )

    scale_y = rng.uniform(
        0.020,
        WORKSPACE_Y * 0.75
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    u = np.linspace(
        0.0,
        2.0 * np.pi,
        n
    )

    u += phase

    trajectory = np.tile(
        center,
        (n, 1)
    )

    trajectory[:, 0] += (
        scale_x
        * np.sin(u)
    )

    trajectory[:, 1] += (
        scale_y
        * np.sin(u)
        * np.cos(u)
    )

    return trajectory


def generate_sine(
    n,
    speed_scale
):
    """
    Smooth sinusoidal trajectory.
    """

    x_start = WORKSPACE_CENTER[0] - (
        0.65 * WORKSPACE_X
    )

    x_end = WORKSPACE_CENTER[0] + (
        0.65 * WORKSPACE_X
    )

    x = np.linspace(
        x_start,
        x_end,
        n
    )

    amplitude = rng.uniform(
        0.015,
        WORKSPACE_Y * 0.80
    )

    cycles = rng.uniform(
        0.5,
        2.5
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    trajectory = np.zeros(
        (n, 3),
        dtype=np.float32
    )

    trajectory[:, 0] = x

    trajectory[:, 1] = (
        WORKSPACE_CENTER[1]
        +
        amplitude
        * np.sin(
            cycles
            * 2.0
            * np.pi
            *
            np.linspace(
                0.0,
                1.0,
                n
            )
            + phase
        )
    )

    trajectory[:, 2] = (
        WORKSPACE_CENTER[2]
        +
        0.010
        *
        np.sin(
            np.linspace(
                0.0,
                2.0 * np.pi,
                n
            )
        )
    )

    return trajectory


def generate_random_smooth(
    n,
    speed_scale
):
    """
    Smooth random 3D trajectory.

    This is useful because it prevents the
    correction model from becoming specialized
    to circles or lines.
    """

    key_count = 12

    key_times = np.linspace(
        0.0,
        1.0,
        key_count
    )

    key_points = np.zeros(
        (key_count, 3),
        dtype=np.float32
    )

    for i in range(key_count):

        key_points[i] = (
            WORKSPACE_CENTER
            +
            np.array([
                rng.uniform(
                    -0.75 * WORKSPACE_X,
                    0.75 * WORKSPACE_X
                ),
                rng.uniform(
                    -0.75 * WORKSPACE_Y,
                    0.75 * WORKSPACE_Y
                ),
                rng.uniform(
                    -0.60 * WORKSPACE_Z,
                    0.60 * WORKSPACE_Z
                )
            ])
        )

    # Smooth interpolation using repeated
    # linear interpolation.
    t = np.linspace(
        0.0,
        1.0,
        n
    )

    trajectory = np.zeros(
        (n, 3),
        dtype=np.float32
    )

    for dim in range(3):

        trajectory[:, dim] = np.interp(
            t,
            key_times,
            key_points[:, dim]
        )

    # Smooth the trajectory several times.
    kernel = np.ones(21) / 21.0

    for dim in range(3):

        padded = np.pad(
            trajectory[:, dim],
            (
                10,
                10
            ),
            mode="edge"
        )

        trajectory[:, dim] = np.convolve(
            padded,
            kernel,
            mode="valid"
        )

    return trajectory


# ============================================================
# TRAJECTORY SELECTION
# ============================================================

def generate_trajectory(
    n,
    speed_scale
):

    trajectory_type = rng.choice([
        "line",
        "circle",
        "ellipse",
        "figure8",
        "sine",
        "random"
    ])

    if trajectory_type == "line":

        trajectory = generate_line(
            n,
            speed_scale
        )

    elif trajectory_type == "circle":

        trajectory = generate_circle(
            n,
            speed_scale
        )

    elif trajectory_type == "ellipse":

        trajectory = generate_ellipse(
            n,
            speed_scale
        )

    elif trajectory_type == "figure8":

        trajectory = generate_figure_eight(
            n,
            speed_scale
        )

    elif trajectory_type == "sine":

        trajectory = generate_sine(
            n,
            speed_scale
        )

    else:

        trajectory = generate_random_smooth(
            n,
            speed_scale
        )

    return trajectory.astype(
        np.float32
    ), trajectory_type


# ============================================================
# VELOCITY
# ============================================================

def calculate_velocity(
    positions
):
    """
    Numerical velocity.
    """

    velocity = np.gradient(
        positions,
        DT,
        axis=0
    )

    return velocity.astype(
        np.float32
    )


# ============================================================
# DATASET CONSTRUCTION
# ============================================================

def build_dataset():

    print("=" * 70)
    print("GENERAL TRAJECTORY DATASET GENERATION")
    print("=" * 70)

    samples_per_trajectory = int(
        TRAJECTORY_DURATION
        * CONTROL_HZ
    )

    print()
    print(
        "Control frequency:",
        CONTROL_HZ,
        "Hz"
    )

    print(
        "Trajectory duration:",
        TRAJECTORY_DURATION,
        "s"
    )

    print(
        "Samples per trajectory:",
        samples_per_trajectory
    )

    print(
        "Number of trajectories:",
        NUMBER_OF_TRAJECTORIES
    )

    print()

    all_target_position = []
    all_target_velocity = []

    all_lookahead_position = []
    all_lookahead_velocity = []

    all_actual_position = []
    all_actual_velocity = []

    all_position_error = []
    all_velocity_error = []

    all_current_pressure = []
    all_previous_pressure = []

    all_trajectory_id = []
    all_trajectory_type = []

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # At this stage we don't know the actual robot response.
    #
    # Therefore this file generates the TARGET side of the
    # dynamic dataset.
    #
    # The actual robot state and pressure history will be
    # filled from simulator rollouts in the next step.
    #
    # --------------------------------------------------------

    for trajectory_id in range(
        NUMBER_OF_TRAJECTORIES
    ):

        speed_scale = rng.uniform(
            MIN_SPEED_SCALE,
            MAX_SPEED_SCALE
        )

        target_position, trajectory_type = (
            generate_trajectory(
                samples_per_trajectory,
                speed_scale
            )
        )

        # ----------------------------------------------------
        # Speed scaling
        # ----------------------------------------------------

        center = np.mean(
            target_position,
            axis=0
        )

        target_position = (
            center
            +
            (
                target_position
                - center
            )
            * speed_scale
        )

        # Keep trajectory inside workspace.
        target_position[:, 0] = np.clip(
            target_position[:, 0],
            WORKSPACE_CENTER[0]
            - WORKSPACE_X,
            WORKSPACE_CENTER[0]
            + WORKSPACE_X
        )

        target_position[:, 1] = np.clip(
            target_position[:, 1],
            WORKSPACE_CENTER[1]
            - WORKSPACE_Y,
            WORKSPACE_CENTER[1]
            + WORKSPACE_Y
        )

        target_position[:, 2] = np.clip(
            target_position[:, 2],
            WORKSPACE_CENTER[2]
            - WORKSPACE_Z,
            WORKSPACE_CENTER[2]
            + WORKSPACE_Z
        )

        target_velocity = calculate_velocity(
            target_position
        )

        # ----------------------------------------------------
        # Lookahead
        # ----------------------------------------------------

        lookahead_position = np.zeros_like(
            target_position
        )

        lookahead_velocity = np.zeros_like(
            target_velocity
        )

        for i in range(
            samples_per_trajectory
        ):

            j = min(
                i + LOOKAHEAD_STEPS,
                samples_per_trajectory - 1
            )

            lookahead_position[i] = (
                target_position[j]
            )

            lookahead_velocity[i] = (
                target_velocity[j]
            )

        # ----------------------------------------------------
        # For now actual state is initialized to target.
        #
        # These values are placeholders.
        #
        # The simulator rollout generator will replace
        # these with REAL actual position/velocity/pressure
        # values.
        # ----------------------------------------------------

        actual_position = (
            target_position.copy()
        )

        actual_velocity = (
            target_velocity.copy()
        )

        current_pressure = np.zeros(
            (
                samples_per_trajectory,
                3
            ),
            dtype=np.float32
        )

        previous_pressure = np.zeros_like(
            current_pressure
        )

        position_error = (
            target_position
            - actual_position
        )

        velocity_error = (
            target_velocity
            - actual_velocity
        )

        # ----------------------------------------------------
        # Remove transient samples
        # ----------------------------------------------------

        start = max(
            LOOKAHEAD_STEPS,
            PRESSURE_HISTORY_STEPS
        )

        for i in range(
            start,
            samples_per_trajectory
        ):

            all_target_position.append(
                target_position[i]
            )

            all_target_velocity.append(
                target_velocity[i]
            )

            all_lookahead_position.append(
                lookahead_position[i]
            )

            all_lookahead_velocity.append(
                lookahead_velocity[i]
            )

            all_actual_position.append(
                actual_position[i]
            )

            all_actual_velocity.append(
                actual_velocity[i]
            )

            all_position_error.append(
                position_error[i]
            )

            all_velocity_error.append(
                velocity_error[i]
            )

            all_current_pressure.append(
                current_pressure[i]
            )

            previous_index = (
                max(
                    0,
                    i - PRESSURE_HISTORY_STEPS
                )
            )

            all_previous_pressure.append(
                current_pressure[
                    previous_index
                ]
            )

            all_trajectory_id.append(
                trajectory_id
            )

            all_trajectory_type.append(
                trajectory_type
            )

        if (
            trajectory_id == 0
            or
            (trajectory_id + 1) % 10 == 0
        ):

            print(
                f"Generated trajectory "
                f"{trajectory_id + 1:3d}/"
                f"{NUMBER_OF_TRAJECTORIES:3d} "
                f"| {trajectory_type:10s} "
                f"| speed {speed_scale:.2f}"
            )

    # ========================================================
    # CONVERT
    # ========================================================

    target_position = np.asarray(
        all_target_position,
        dtype=np.float32
    )

    target_velocity = np.asarray(
        all_target_velocity,
        dtype=np.float32
    )

    lookahead_position = np.asarray(
        all_lookahead_position,
        dtype=np.float32
    )

    lookahead_velocity = np.asarray(
        all_lookahead_velocity,
        dtype=np.float32
    )

    actual_position = np.asarray(
        all_actual_position,
        dtype=np.float32
    )

    actual_velocity = np.asarray(
        all_actual_velocity,
        dtype=np.float32
    )

    position_error = np.asarray(
        all_position_error,
        dtype=np.float32
    )

    velocity_error = np.asarray(
        all_velocity_error,
        dtype=np.float32
    )

    current_pressure = np.asarray(
        all_current_pressure,
        dtype=np.float32
    )

    previous_pressure = np.asarray(
        all_previous_pressure,
        dtype=np.float32
    )

    trajectory_id = np.asarray(
        all_trajectory_id,
        dtype=np.int32
    )

    trajectory_type = np.asarray(
        all_trajectory_type
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,

        target_position=target_position,

        target_velocity=target_velocity,

        lookahead_position=lookahead_position,

        lookahead_velocity=lookahead_velocity,

        actual_position=actual_position,

        actual_velocity=actual_velocity,

        position_error=position_error,

        velocity_error=velocity_error,

        current_pressure=current_pressure,

        previous_pressure=previous_pressure,

        trajectory_id=trajectory_id,

        trajectory_type=trajectory_type,

        control_hz=np.float32(
            CONTROL_HZ
        ),

        lookahead_time=np.float32(
            LOOKAHEAD_TIME
        ),

        pressure_history_delay=np.float32(
            PREVIOUS_PRESSURE_DELAY
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET GENERATED")
    print("=" * 70)

    print()
    print(
        "Samples:",
        len(target_position)
    )

    print(
        "Target position:",
        target_position.shape
    )

    print(
        "Target velocity:",
        target_velocity.shape
    )

    print(
        "Lookahead position:",
        lookahead_position.shape
    )

    print(
        "Lookahead velocity:",
        lookahead_velocity.shape
    )

    print(
        "Actual position:",
        actual_position.shape
    )

    print(
        "Position error:",
        position_error.shape
    )

    print(
        "Current pressure:",
        current_pressure.shape
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print()
    print(
        "This is the TARGET/STRUCTURE dataset."
    )

    print(
        "Do NOT train the correction MLP from this"
    )

    print(
        "placeholder actual-state data yet."
    )

    print()
    print(
        "Next step is simulator rollout generation,"
    )

    print(
        "which will fill actual position, velocity,"
    )

    print(
        "pressure history and correction targets."
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_dataset()