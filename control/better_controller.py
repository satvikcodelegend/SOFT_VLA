"""
Temporal closed-loop controller.

The temporal inverse MLP is responsible for learning:

    current position
    target position
    position error
    velocity
    previous pressure
            |
            v
        P1 P2 P3

This controller only handles:

1. Temporal state construction
2. Inverse MLP inference
3. Pressure limiting
4. Pressure slew-rate limiting
5. Light command smoothing
6. State reset
7. Diagnostics

No Jacobian.
No analytical inverse kinematics.
No predictive pressure search.
No second inverse model.
"""


import numpy as np

from control.inverse_mlp import (
    InverseController
)


class BetterController:

    def __init__(
        self,
        model_path="models/temporal_inverse_mlp.pt",

        pressure_min=0.0,
        pressure_max=3.0,

        max_pressure_rate=1.0,

        filter_alpha=0.25,

        velocity_filter_alpha=0.20,

        velocity_limit=2.0,

        dt=0.01
    ):

        # ====================================================
        # Parameters
        # ====================================================

        self.pressure_min = float(
            pressure_min
        )

        self.pressure_max = float(
            pressure_max
        )

        self.max_pressure_rate = float(
            max_pressure_rate
        )

        self.filter_alpha = float(
            filter_alpha
        )

        self.velocity_filter_alpha = float(
            velocity_filter_alpha
        )

        self.velocity_limit = float(
            velocity_limit
        )

        self.dt = float(
            dt
        )

        if self.pressure_max <= self.pressure_min:

            raise ValueError(
                "pressure_max must be greater "
                "than pressure_min."
            )

        if self.max_pressure_rate <= 0.0:

            raise ValueError(
                "max_pressure_rate must be positive."
            )

        if not (
            0.0
            <
            self.filter_alpha
            <=
            1.0
        ):

            raise ValueError(
                "filter_alpha must be in (0,1]."
            )

        if not (
            0.0
            <
            self.velocity_filter_alpha
            <=
            1.0
        ):

            raise ValueError(
                "velocity_filter_alpha must be in (0,1]."
            )

        if self.dt <= 0.0:

            raise ValueError(
                "dt must be positive."
            )

        # ====================================================
        # Load temporal inverse model
        # ====================================================

        self.inverse_controller = (
            InverseController(
                model_path=model_path,
                pressure_min=self.pressure_min,
                pressure_max=self.pressure_max
            )
        )

        # ====================================================
        # Temporal state
        # ====================================================

        self.previous_pressure = np.zeros(
            3,
            dtype=np.float64
        )

        self.previous_position = None

        self.filtered_velocity = np.zeros(
            3,
            dtype=np.float64
        )

        self.previous_target = None

        self.initialized = False

        # ====================================================
        # Diagnostics
        # ====================================================

        self.last_error = np.zeros(
            3,
            dtype=np.float64
        )

        self.last_velocity = np.zeros(
            3,
            dtype=np.float64
        )

        self.last_raw_pressure = np.zeros(
            3,
            dtype=np.float64
        )

        self.last_pressure = np.zeros(
            3,
            dtype=np.float64
        )

        print()
        print("=" * 70)
        print("TEMPORAL BETTER CONTROLLER")
        print("=" * 70)

        print(
            "Inverse model:",
            model_path
        )

        print(
            "Pressure range:",
            self.pressure_min,
            "->",
            self.pressure_max
        )

        print(
            "Maximum pressure rate:",
            self.max_pressure_rate,
            "bar/s"
        )

        print(
            "Pressure filter alpha:",
            self.filter_alpha
        )

        print(
            "Velocity filter alpha:",
            self.velocity_filter_alpha
        )

        print("=" * 70)

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        current_pressure=None,
        current_position=None
    ):

        if current_pressure is None:

            self.previous_pressure = np.zeros(
                3,
                dtype=np.float64
            )

        else:

            current_pressure = np.asarray(
                current_pressure,
                dtype=np.float64
            )

            if current_pressure.shape != (3,):

                raise ValueError(
                    "current_pressure must have shape (3,)."
                )

            self.previous_pressure = np.clip(
                current_pressure,
                self.pressure_min,
                self.pressure_max
            )

        if current_position is None:

            self.previous_position = None

        else:

            current_position = np.asarray(
                current_position,
                dtype=np.float64
            )

            if current_position.shape != (3,):

                raise ValueError(
                    "current_position must have shape (3,)."
                )

            self.previous_position = (
                current_position.copy()
            )

        self.filtered_velocity[:] = 0.0

        self.previous_target = None

        self.initialized = False

        self.last_error[:] = 0.0
        self.last_velocity[:] = 0.0
        self.last_raw_pressure[:] = 0.0
        self.last_pressure[:] = (
            self.previous_pressure
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_vector(
        value,
        name
    ):

        value = np.asarray(
            value,
            dtype=np.float64
        )

        if value.shape != (3,):

            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(
            np.isfinite(value)
        ):

            raise ValueError(
                f"{name} contains NaN/Inf."
            )

        return value

    # ========================================================
    # VELOCITY ESTIMATION
    # ========================================================

    def _compute_velocity(
        self,
        current_position,
        dt
    ):

        if self.previous_position is None:

            raw_velocity = np.zeros(
                3,
                dtype=np.float64
            )

        else:

            raw_velocity = (
                current_position
                -
                self.previous_position
            ) / dt

        # ----------------------------------------------------
        # Remove impossible numerical spikes.
        # ----------------------------------------------------

        speed = np.linalg.norm(
            raw_velocity
        )

        if speed > self.velocity_limit:

            if speed > 1e-12:

                raw_velocity = (
                    raw_velocity
                    *
                    (
                        self.velocity_limit
                        /
                        speed
                    )
                )

            else:

                raw_velocity[:] = 0.0

        # ----------------------------------------------------
        # Low-pass filter.
        # ----------------------------------------------------

        self.filtered_velocity = (
            (
                1.0
                -
                self.velocity_filter_alpha
            )
            *
            self.filtered_velocity
            +
            self.velocity_filter_alpha
            *
            raw_velocity
        )

        return self.filtered_velocity.copy()

    # ========================================================
    # PRESSURE SLEW LIMIT
    # ========================================================

    def _apply_pressure_rate_limit(
        self,
        desired_pressure,
        dt
    ):

        maximum_change = (
            self.max_pressure_rate
            *
            dt
        )

        change = (
            desired_pressure
            -
            self.previous_pressure
        )

        change = np.clip(
            change,
            -maximum_change,
            maximum_change
        )

        pressure = (
            self.previous_pressure
            +
            change
        )

        pressure = np.clip(
            pressure,
            self.pressure_min,
            self.pressure_max
        )

        return pressure

    # ========================================================
    # LIGHT PRESSURE FILTER
    # ========================================================

    def _filter_pressure(
        self,
        pressure
    ):

        filtered = (
            (
                1.0
                -
                self.filter_alpha
            )
            *
            self.previous_pressure
            +
            self.filter_alpha
            *
            pressure
        )

        return np.clip(
            filtered,
            self.pressure_min,
            self.pressure_max
        )

    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        target_position,
        actual_position,
        dt=None
    ):

        target_position = (
            self._validate_vector(
                target_position,
                "target_position"
            )
        )

        actual_position = (
            self._validate_vector(
                actual_position,
                "actual_position"
            )
        )

        if dt is None:

            dt = self.dt

        dt = float(dt)

        if dt <= 0.0:

            raise ValueError(
                "dt must be positive."
            )

        # ====================================================
        # Temporal velocity
        # ====================================================

        velocity = self._compute_velocity(
            actual_position,
            dt
        )

        # ====================================================
        # Error
        # ====================================================

        position_error = (
            target_position
            -
            actual_position
        )

        # ====================================================
        # IMPORTANT
        #
        # The exact features used during training are:
        #
        # current position
        # target position
        # error
        # velocity
        # previous pressure
        # ====================================================

        raw_pressure = (
            self.inverse_controller
            .predict_pressure(
                current_position=actual_position,
                target_position=target_position,
                velocity=velocity,
                previous_pressure=self.previous_pressure
            )
        )

        raw_pressure = np.clip(
            raw_pressure,
            self.pressure_min,
            self.pressure_max
        )

        # ====================================================
        # Pressure rate limiting
        # ====================================================

        slew_pressure = (
            self._apply_pressure_rate_limit(
                raw_pressure,
                dt
            )
        )

        # ====================================================
        # Light smoothing
        #
        # The temporal MLP already has previous pressure
        # as an input, so we intentionally do NOT use a huge
        # filter here.
        # ====================================================

        pressure = (
            self._filter_pressure(
                slew_pressure
            )
        )

        # ====================================================
        # Final safety clamp
        # ====================================================

        pressure = np.clip(
            pressure,
            self.pressure_min,
            self.pressure_max
        )

        # ====================================================
        # Update temporal state
        # ====================================================

        self.previous_position = (
            actual_position.copy()
        )

        self.previous_target = (
            target_position.copy()
        )

        self.previous_pressure = (
            pressure.copy()
        )

        self.last_error = (
            position_error.copy()
        )

        self.last_velocity = (
            velocity.copy()
        )

        self.last_raw_pressure = (
            raw_pressure.copy()
        )

        self.last_pressure = (
            pressure.copy()
        )

        self.initialized = True

        return pressure.copy()

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def get_tracking_error(self):

        return self.last_error.copy()

    def get_velocity(self):

        return self.last_velocity.copy()

    def get_raw_pressure(self):

        return self.last_raw_pressure.copy()

    def get_pressure(self):

        return self.last_pressure.copy()

    def get_previous_pressure(self):

        return self.previous_pressure.copy()