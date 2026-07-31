import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kinkantikink.boost import TwoSolitonIC
from kinkantikink.parameters import Params
from kinkantikink.physics import Vprime

# test array
p = Params()
d0 = 20.0
x = np.arange(-80, 80, 0.05)


# Test 1: Vacuum asymptotics far left and far right must BOTH equal -v
@pytest.mark.parametrize("v_in", [0.1, 0.3, 0.6])
def test_far_field_matches_vacuum(v_in):
    phi0, _ = TwoSolitonIC(x, d0, v_in, p)

    left_value = phi0[0]
    right_value = phi0[-1]

    assert left_value == pytest.approx(-p.v, abs=1e-3)
    assert right_value == pytest.approx(-p.v, abs=1e-3)


# 2. Interior plateau: between the two solitons, field should sit near +v
def test_interior_plateau_matches_positive_vacuum():
    v_in = 0.2
    phi0, _ = TwoSolitonIC(x, d0, v_in, p)

    mid_index = np.argmin(np.abs(x))
    assert phi0[mid_index] == pytest.approx(p.v, abs=1e-2)


# 3. Eqn Of Motion consistency, bas does phi0 still satisfy eqn after being boosted
def test_eom_residual_small_away_from_cores():
    v_in = 0.2
    phi0, _ = TwoSolitonIC(x, d0, v_in, p)
    dx = x[1] - x[0]

    lap = np.empty_like(phi0)
    lap[1:-1] = (phi0[2:] - 2 * phi0[1:-1] + phi0[:-2]) / dx**2

    residual = lap[1:-1] - Vprime(phi0[1:-1], p)
    core_width = 5.0
    mask = (np.abs(x[1:-1] + d0) > core_width) & (np.abs(x[1:-1] - d0) > core_width)

    assert np.max(np.abs(residual[mask])) < 0.05


# 4. Propagation check, if boosted kink is moving at v_in or not
def test_kink_moves_at_v_in():
    v_in = 0.25
    phi0, phi_dot0 = TwoSolitonIC(x, d0, v_in, p)

    def find_left_kink_zero_crossing(phi):
        window = (x > -d0 - 10) & (x < -d0 + 10)
        xi = x[window]
        phii = phi[window]
        sign_changes = np.where(np.diff(np.sign(phii)))[0]
        idx = sign_changes[0]
        x0_, x1_ = xi[idx], xi[idx + 1]
        y0_, y1_ = phii[idx], phii[idx + 1]
        return x0_ - y0_ * (x1_ - x0_) / (y1_ - y0_)

    dt = 0.01
    crossing_before = find_left_kink_zero_crossing(phi0)
    phi_after = phi0 + dt * phi_dot0
    crossing_after = find_left_kink_zero_crossing(phi_after)

    measured_speed = (crossing_after - crossing_before) / dt
    assert measured_speed == pytest.approx(v_in, rel=0.15)

# 5. Propagation check, if boosted antikink is moving at v_in or not
def test_antikink_moves_at_v_in():
    v_in = 0.25
    phi0, phi_dot0 = TwoSolitonIC(x, d0, v_in, p)

    def find_right_antikink_zero_crossing(phi):
        window = (x > d0 - 10) & (x < d0 + 10)
        xi = x[window]
        phii = phi[window]
        sign_changes = np.where(np.diff(np.sign(phii)))[0]
        idx = sign_changes[0]
        x0_, x1_ = xi[idx], xi[idx + 1]
        y0_, y1_ = phii[idx], phii[idx + 1]
        return x0_ - y0_ * (x1_ - x0_) / (y1_ - y0_)

    dt = 0.01
    crossing_before = find_right_antikink_zero_crossing(phi0)
    phi_after = phi0 + dt * phi_dot0
    crossing_after = find_right_antikink_zero_crossing(phi_after)

    measured_speed = (crossing_after - crossing_before) / dt
    # antikink at +d0 should move toward the kink, i.e. leftward: speed ~ -v_in
    assert measured_speed == pytest.approx(-v_in, rel=0.15)


# python -m pytest -v
