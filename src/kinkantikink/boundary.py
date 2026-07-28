import numpy as np

from .parameters import Params


def sponge_damping_coefficient(x, p: Params):
    distance_from_left_edge = x - (-p.L)
    distance_from_right_edge = p.L - x

    left_ramp = np.clip(1.0 - distance_from_left_edge / p.sponge_width, 0.0, 1.0)
    right_ramp = np.clip(1.0 - distance_from_right_edge / p.sponge_width, 0.0, 1.0)

    return p.damping * (left_ramp + right_ramp)