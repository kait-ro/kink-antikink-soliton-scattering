"""
load .npz file and make 2d gif

Usage:
    python post/animate_2d.py --input data/something.npz --output out.gif --fps 30 --stride 1
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinkantikink.io_utils import load_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate a saved collision run as a 2D line-plot GIF."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the saved .npz run."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GIF path (default: alongside input, with .gif extension).",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the GIF (default: 30)."
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="to stride, use every Nth frame in gif.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = load_run(args.input)

    x = run["x"]
    t = run["t"][:: args.stride]
    phi_history = run["phi_history"][:: args.stride]
    v_in = float(run["v_in"])

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.input)
        output_path = base + ".gif"

    fig, ax = plt.subplots(figsize=(8, 4))
    (line,) = ax.plot(x, phi_history[0], color="tab:blue", lw=2)
    ax.set_xlim(x[0], x[-1])

    y_pad = 0.2 * (np.max(np.abs(phi_history)) + 1e-9)
    ax.set_ylim(np.min(phi_history) - y_pad, np.max(phi_history) + y_pad)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x,t)$")
    title = ax.set_title(f"v_in = {v_in:.3f}, t = {t[0]:.2f}")

    def update(frame_idx):
        line.set_ydata(phi_history[frame_idx])
        title.set_text(f"v_in = {v_in:.3f}, t = {t[frame_idx]:.2f}")
        return line, title

    anim = FuncAnimation(
        fig, update, frames=len(t), interval=1000 / args.fps, blit=False
    )

    print(f"Saving animation to {output_path} ({len(t)} frames)")
    anim.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print("Done")


if __name__ == "__main__":
    main()