from .parameters import Params
from .physics import phi_kink, phi_x


def positionKinkBoostInitial(x, x0, v_in, p: Params, sign):
    gamma = 1 / (1 - v_in**2) ** 0.5
    return phi_kink(gamma * x, gamma * x0, p, sign)


def movingKinkInitial(x, x0, v_in, p: Params, sign):
    gamma = 1 / (1 - v_in**2) ** 0.5
    return -v_in * gamma * phi_x(gamma * x, gamma * x0, p, sign)


# Two soliton initial condition
# Centered at 0, d0 apart
def TwoSolitonIC(x, d0, v_in, p):
    kink_part = positionKinkBoostInitial(x, -d0, v_in, p, sign=1)
    antikink_part = positionKinkBoostInitial(x, d0, v_in, p, sign=-1)
    phi0 = kink_part + antikink_part - p.v

    kink_vel = movingKinkInitial(x, -d0, v_in, p, sign=1)
    antikink_vel = movingKinkInitial(x, d0, v_in, p, sign=-1)
    phi_dot0 = kink_vel - antikink_vel
    return phi0, phi_dot0