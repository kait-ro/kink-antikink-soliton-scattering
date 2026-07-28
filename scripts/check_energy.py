"""
load .npz file to check energy

Usage:
    python check_energy.py --input data/something.npz
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinkantikink.energy import total_energy
from kinkantikink.io_utils import load_run
from kinkantikink.parameters import Params


def parse_args():
    parser = argparse.ArgumentParser(
        description="plot total energy vs. time for a saved collision run."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="path to the saved .npz run."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="output image path"
    )
    parser.add_argument(
        "--show", action="store_true", help="whether to open an interactive window."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = load_run(args.input)

    x = run["x"]
    t = run["t"]
    phi_history = run["phi_history"]
    v_in = float(run["v_in"])

    # rebuild the Params used for this run so V(phi) etc. match
    p = Params(
        lam=float(run["lam"]),
        v=float(run["v"]),
        L=float(run["L"]),
        dx=float(run["dx"]),
        dt=float(run["dt"]),
        damping=float(run["damping"]),
        sponge_width=float(run["sponge_width"]),
    )

    dt_saved = np.diff(t)
    phi_dot = np.zeros_like(phi_history)
    phi_dot[1:-1] = (phi_history[2:] - phi_history[:-2]) / (
        dt_saved[1:, None] + dt_saved[:-1, None]
    )
    phi_dot[0] = (phi_history[1] - phi_history[0]) / dt_saved[0]
    phi_dot[-1] = (phi_history[-1] - phi_history[-2]) / dt_saved[-1]

    energies = np.array([
        total_energy(phi_history[i], phi_dot[i], x, p)
        for i in range(len(t))
    ])

    max_abs_phi = np.max(np.abs(phi_history), axis=1)

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.input)
        output_path = base + "_energy.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(t, energies, color="tab:red")
    ax1.set_ylabel("Total energy")
    ax1.set_title(f"Energy & max |phi| vs time, v_in = {v_in:.3f}")
    ax1.axhline(energies[0], color="gray", ls="--", lw=1, label="initial energy")
    ax1.legend()

    ax2.plot(t, max_abs_phi, color="tab:blue")
    ax2.axhline(p.v, color="gray", ls="--", lw=1, label=f"vacuum |phi| = {p.v}")
    ax2.set_ylabel("max |phi(x,t)|")
    ax2.set_xlabel("t")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved energy plot to {output_path}")

    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy:   {energies[-1]:.6f}")
    print(f"Energy drift:   {energies[-1] - energies[0]:.6f}")
    print(f"Max |phi| overall: {np.max(max_abs_phi):.4f} (vacuum = {p.v})")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()



