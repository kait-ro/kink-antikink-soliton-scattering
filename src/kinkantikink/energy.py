import numpy as np
from scipy.integrate import trapezoid

from .parameters import Params
from .physics import V


def energy_density(field_values, field_velocity, x, p: Params):
    # epsilon = 0.5 * field_t^2 + 0.5 * field_x^2 + V(field)
    field_x = np.zeros(field_values.shape)
    field_x[1:-1] = (field_values[2:] - field_values[:-2]) / (2 * p.dx)
    field_x[0] = (field_values[1] - field_values[0]) / p.dx
    field_x[-1] = (field_values[-1] - field_values[-2]) / p.dx

    return 0.5 * field_velocity**2 + 0.5 * field_x**2 + V(field_values, p)


def total_energy(field_values, field_velocity, x, p: Params):
    epsilon = energy_density(field_values, field_velocity, x, p)
    return trapezoid(epsilon, x)