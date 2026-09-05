import os
import numpy as np
import mujoco

from config.parameters import (
    TOTAL_LENGTH,
    NUM_SEGMENTS,
    SEGMENT_LENGTH,
    YOUNGS_MODULUS,
    STRAIN_GAIN,
    DELTA,
    THETA,
    ACTUATOR_AREA,
    AIR_AREA,
    I1,
    I2,
    P_THRESHOLD,
)

MODEL_PATH = "model/scene.xml"
OUTPUT_PATH = "data/inverse_dataset.npz"

NUM_SAMPLES = 5000
SETTLE_TIME = 0.12
AVERAGE_TIME = 0.02
BAR_TO_PA = 1.0e5

np.random.seed(42)

print("=" * 70)
print("MUJOCO INVERSE DATASET GENERATION")
print("=" * 70)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print()
print("Model loaded")
print("Joints:", model.njnt)
print("Actuators:", model.nu)
print("Generating", NUM_SAMPLES, "pressure samples")

u = np.random.rand(NUM_SAMPLES, 3)
pressure_samples = 3.0 * u

positions = np.empty((NUM_SAMPLES, 3), dtype=np.float32)
pressures = np.empty((NUM_SAMPLES, 3), dtype=np.float32)


def pressure_to_joint_targets(pressure):
    effective_pressure = np.where(
        pressure >= P_THRESHOLD,
        pressure,
        0.0
    )

    pressure_pa = effective_pressure * BAR_TO_PA

    base_strain = (
        AIR_AREA
        / (YOUNGS_MODULUS * ACTUATOR_AREA)
    ) * pressure_pa

    strain = STRAIN_GAIN * base_strain

    extension = np.mean(strain) * TOTAL_LENGTH

    kappa_x = np.sum(
        strain
        * ACTUATOR_AREA
        * DELTA
        * np.sin(THETA)
    ) / I1

    kappa_y = -np.sum(
        strain
        * ACTUATOR_AREA
        * DELTA
        * np.cos(THETA)
    ) / I2

    segment_extension = extension / NUM_SEGMENTS
    segment_theta_x = kappa_x * SEGMENT_LENGTH
    segment_theta_y = kappa_y * SEGMENT_LENGTH

    return (
        segment_extension,
        segment_theta_x,
        segment_theta_y
    )


def set_actuators(pressure):
    extension, theta_x, theta_y = pressure_to_joint_targets(
        pressure
    )

    ctrl = np.empty(model.nu, dtype=np.float64)

    for i in range(NUM_SEGMENTS):
        base = 3 * i
        ctrl[base] = theta_x
        ctrl[base + 1] = theta_y
        ctrl[base + 2] = extension

    return ctrl


def simulate(pressure):
    mujoco.mj_resetData(model, data)

    ctrl = set_actuators(pressure)
    data.ctrl[:] = ctrl

    settle_steps = max(
        1,
        int(SETTLE_TIME / model.opt.timestep)
    )

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    average_steps = max(
        1,
        int(AVERAGE_TIME / model.opt.timestep)
    )

    position_sum = np.zeros(3)

    for _ in range(average_steps):
        mujoco.mj_step(model, data)
        position_sum += data.site_xpos[-1]

    return position_sum / average_steps


for i in range(NUM_SAMPLES):
    pressure = pressure_samples[i]
    position = simulate(pressure)

    pressures[i] = pressure
    positions[i] = position

    if i == 0 or (i + 1) % 500 == 0:
        print(f"{i + 1}/{NUM_SAMPLES}")


valid = np.all(np.isfinite(positions), axis=1)

positions = positions[valid]
pressures = pressures[valid]

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.savez(
    OUTPUT_PATH,
    position=positions,
    pressure=pressures
)

print()
print("=" * 70)
print("DATASET COMPLETE")
print("=" * 70)
print()
print("Valid samples:", len(positions))
print("Pressure shape:", pressures.shape)
print("Position shape:", positions.shape)
print()
print("Pressure minimum:")
print(pressures.min(axis=0))
print()
print("Pressure maximum:")
print(pressures.max(axis=0))
print()
print("Position minimum:")
print(positions.min(axis=0))
print()
print("Position maximum:")
print(positions.max(axis=0))
print()
print("Saved:", OUTPUT_PATH)
print()
print("=" * 70)
print("DONE")
print("=" * 70)