import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.integrate import trapezoid

from kinkantikink.boost import TwoSolitonIC
from kinkantikink.energy import energy_density
from kinkantikink.parameters import Params
from kinkantikink.solver import run_simulation

p = Params()
d0 = 20.0
x = np.arange(-p.L, p.L, p.dx)


def test_interior_energy_roughly_conserved():
    v_in = 0.3
    phi0, phi_dot0 = TwoSolitonIC(x, d0, v_in, p)

    number_of_timesteps = 400
    save_every_n_steps = 40
    t, phi_history = run_simulation(
        phi0, phi_dot0, x, p, number_of_timesteps, save_every_n_steps
    )

    interior_mask = (x > -p.L + p.sponge_width) & (x < p.L - p.sponge_width)
    x_interior = x[interior_mask]

    interior_energy_over_time = []
    for i, current_phi in enumerate(phi_history):
        if i == 0:
            phi_dot_snapshot = phi_dot0
        else:
            dt_between_snapshots = t[i] - t[i - 1]
            phi_dot_snapshot = (current_phi - phi_history[i - 1]) / dt_between_snapshots

        epsilon = energy_density(current_phi, phi_dot_snapshot, p)
        interior_energy = trapezoid(epsilon[interior_mask], x_interior)
        interior_energy_over_time.append(interior_energy)

    interior_energy_over_time = np.array(interior_energy_over_time)

    initial_energy = interior_energy_over_time[0]
    max_relative_deviation = np.max(
        np.abs(interior_energy_over_time - initial_energy) / initial_energy
    )

    assert max_relative_deviation < 0.15

# python -m pytest -v