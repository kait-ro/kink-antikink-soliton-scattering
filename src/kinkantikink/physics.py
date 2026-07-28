import numpy as np

from .parameters import Params


def V(phi, p: Params):
    return p.lam / 4 * (phi**2 - p.v**2) ** 2


def Vprime(phi, p: Params):
    return p.lam * phi * (phi**2 - p.v**2)


def phi_kink(x, x0, p: Params, sign=1):
    k = (p.lam / 2) ** 0.5 * p.v
    u = k * (x - x0)
    return sign * p.v * np.tanh(u)

# wrt x
def phi_x(x, x0, p: Params, sign=1):
    k = (p.lam / 2) ** 0.5 * p.v
    u = k * (x - x0)
    sech_u = 1 / np.cosh(u)
    return sign * p.v * k * sech_u**2

# wrt x
def phi_xx(x, x0, p: Params, sign=1):
    k = (p.lam / 2) ** 0.5 * p.v
    u = k * (x - x0)
    sech_u = 1 / np.cosh(u)
    return -p.lam * (p.v**3) * sech_u**2 * np.tanh(u) * sign
