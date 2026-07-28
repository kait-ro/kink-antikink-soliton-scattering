import numpy as np

from .boundary import sponge_damping_coefficient
from .parameters import Params
from .physics import Vprime


def compute_acceleration(field_values, field_velocity, x, p: Params):
    # field_acceleration = field_xx - V'(field) - damping_coefficient * field_velocity
    second_spatial_derivative = np.zeros(field_values.shape)
    second_spatial_derivative[1:-1] = (
        field_values[2:] - 2 * field_values[1:-1] + field_values[:-2]
    ) / p.dx**2
    second_spatial_derivative[0] = (
        field_values[2] - 2 * field_values[1] + field_values[0]
    ) / p.dx**2
    second_spatial_derivative[-1] = (
        field_values[-1] - 2 * field_values[-2] + field_values[-3]
    ) / p.dx**2

    damping_coefficient = sponge_damping_coefficient(x, p)

    field_acceleration = (
        second_spatial_derivative
        - Vprime(field_values, p)
        - damping_coefficient * field_velocity
    )
    return field_acceleration


def advance_one_timestep(field_values, field_velocity, x, p: Params):
    """
    3 step update:
      1. half-step update to the velocity using the current acceleration
      2. full-step update to the field using the half-updated velocity
      3. half-step update to the velocity using the acceleration at the
         new field values
    Returns (new_field_values, new_field_velocity)
    """
    half_step_velocity = field_velocity + 0.5 * p.dt * compute_acceleration(
        field_values, field_velocity, x, p
    )
    new_field_values = field_values + p.dt * half_step_velocity
    new_field_velocity = half_step_velocity + 0.5 * p.dt * compute_acceleration(
        new_field_values, half_step_velocity, x, p
    )
    return new_field_values, new_field_velocity


def run_simulation(
    initial_field_values,
    initial_field_velocity,
    x,
    p: Params,
    number_of_timesteps,
    save_every_n_steps=1,
):
    """
    Runs the full time-stepping loop saving every save_every_n_steps steps so the full history is not
    kept in memory for long runs.
    """
    field_values = initial_field_values.copy()
    field_velocity = initial_field_velocity.copy()

    time_values = [0.0]
    field_history = [field_values.copy()]

    current_time = 0.0
    for step_number in range(1, number_of_timesteps + 1):
        field_values, field_velocity = advance_one_timestep(
            field_values, field_velocity, x, p
        )
        current_time += p.dt

        if step_number % save_every_n_steps == 0:
            time_values.append(current_time)
            field_history.append(field_values.copy())

    return np.array(time_values), np.array(field_history)