"""
run single collision at given v_in and save the output

Usage:
    python scripts/run_collision.py --v_in 0.18
    python scripts/run_collision.py --v_in 0.35 --d0 15 --t_final 80 --output_dir data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinkantikink.boost import TwoSolitonIC
from kinkantikink.discretize import checkCFL, spatialGrid
from kinkantikink.energy import total_energy
from kinkantikink.io_utils import save_run
from kinkantikink.parameters import Params
from kinkantikink.solver import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="run one collision and save result.")
    parser.add_argument(
        "--v_in",
        type=float,
        required=True,
        help="incoming speed of each soliton in natural units",
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=15.0,
        help="separation of the kink/antikink centers at t=0",
    )
    parser.add_argument(
        "--t_final",
        type=float,
        default=120.0,
        help="total simulated time",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="save the field every N timesteps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="directory to write the .npz output into",
    )
    return parser.parse_args()


def run_one_collision(
    v_in, d0=15.0, t_final=80.0, save_every=10, output_dir="data", p=None, verbose=True
):
    if p is None:
        p = Params()
    checkCFL(p)

    x = spatialGrid(p)

    phi0, phi_dot0 = TwoSolitonIC(x, d0, v_in, p)

    e_initial = total_energy(phi0, phi_dot0, x, p)
    if verbose:
        print(f"Initial total energy: {e_initial:.6f}")

    number_of_timesteps = int(t_final / p.dt)

    if verbose:
        print(
            f"Running collision: v_in={v_in}, d0={d0}, "
            f"t_final={t_final} ({number_of_timesteps} steps), "
            f"save_every={save_every}"
        )

    t, phi_history = run_simulation(
        phi0,
        phi_dot0,
        x,
        p,
        number_of_timesteps,
        save_every_n_steps=save_every,
    )
    if verbose and len(t) >= 2:
        dt_saved = t[-1] - t[-2]
        phi_dot_final_est = (phi_history[-1] - phi_history[-2]) / dt_saved
        e_final = total_energy(phi_history[-1], phi_dot_final_est, x, p)
        print(f"Final total energy (approx): {e_final:.6f}")
        print(f"Energy drift: {e_final - e_initial:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    filename = save_run(x, t, phi_history, p, v_in, output_dir=output_dir)
    if verbose:
        print(f"Saved run to: {filename}")

    return filename


def main():
    args = parse_args()
    run_one_collision(
        v_in=args.v_in,
        d0=args.d0,
        t_final=args.t_final,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
