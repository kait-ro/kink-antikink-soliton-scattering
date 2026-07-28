import numpy as np

from .parameters import Params


def spatialGrid(p: Params):
    x = np.arange(-p.L, p.L, p.dx)
    return x

def gridToPhi(x, phi_func, x0, p: Params, sign=1):
    return phi_func(x, x0, p, sign)

def laplacian(phiArr, dx, boundary="zero"):
    lap = np.zeros(len(phiArr))
    lap[1:-1] = (phiArr[2:] - 2 * phiArr[1:-1] + phiArr[:-2]) / dx**2

    if boundary == "shift":
        lap[0] = (phiArr[2] - 2 * phiArr[1] + phiArr[0]) / dx**2
        lap[-1] = (phiArr[-1] - 2 * phiArr[-2] + phiArr[-3]) / dx**2

    return lap

def checkCFL(p: Params):
    if p.dt > p.dx:
        raise ValueError(f"CFL violated: dt={p.dt} must be <= dx={p.dx}")