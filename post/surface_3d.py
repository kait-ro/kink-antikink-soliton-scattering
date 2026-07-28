"""
Load a saved kink-antikink collision run and plot the full (x, t, phi)
array as a 3D space-time surface - a "whole collision at a glance" figure,
good for the README. Can render either a static PNG of the final surface,
or a GIF that grows the surface forward in time, frame by frame, so you
watch the kink/antikink actually move and collide in 3D.

Usage:
    python post/surface_3d.py --input data/run_vin0.18_....npz
    python post/surface_3d.py --input data/run_vin0.18_....npz --output surface.png
    python post/surface_3d.py --input data/run_vin0.18_....npz --show

    # animated GIF that grows over time (t=0 -> t_final):
    python post/surface_3d.py --input data/run_vin0.18_....npz --gif
    python post/surface_3d.py --input data/run_vin0.18_....npz --gif --fps 20 --stride 2
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinkantikink.io_utils import load_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a saved collision run as a 3D space-time surface."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the saved .npz run."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: alongside input, with _surface.png or "
             "_surface.gif suffix depending on --gif).",
    )
    parser.add_argument(
        "--x_stride", type=int, default=2,
        help="Downsample the spatial grid by this stride, for render speed (default: 2).",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open an interactive rotatable window."
    )
    parser.add_argument(
        "--elev", type=float, default=25.0, help="Camera elevation angle (default: 25)."
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0, help="Camera azimuth angle (default: -60)."
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Render a GIF that grows the surface forward in time, "
             "frame by frame, instead of a static PNG of the finished surface.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Use every Nth saved time frame for the GIF, to shorten/thin "
             "the animation (default: 1, i.e. every saved frame).",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second for the GIF (default: 20)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = load_run(args.input)

    x = run["x"][:: args.x_stride]
    t_full = run["t"][:: args.stride]
    phi_full = run["phi_history"][:: args.stride, :: args.x_stride]
    v_in = float(run["v_in"])

    base, _ = os.path.splitext(args.input)

    # fixed z-limits (and colorbar range) across all frames so the GIF
    # doesn't rescale/jump as more of the surface is revealed
    z_min, z_max = np.min(phi_full), np.max(phi_full)
    z_pad = 0.1 * (z_max - z_min + 1e-9)
    z_min, z_max = z_min - z_pad, z_max + z_pad

    if args.gif:
        output_path = args.output
        if output_path is None:
            output_path = base + "_surface.gif"

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        n_frames = len(t_full)

        def update(frame_idx):
            ax.clear()
            # reveal everything from t=0 up through the current frame
            x_sub = x
            t_sub = t_full[: frame_idx + 1]
            phi_sub = phi_full[: frame_idx + 1]

            if len(t_sub) < 2:
                # plot_surface needs at least a 2x2 grid; pad with a
                # duplicate row for the very first frame
                t_sub = np.array([t_sub[0], t_sub[0] + 1e-6])
                phi_sub = np.vstack([phi_sub, phi_sub])

            X, T = np.meshgrid(x_sub, t_sub)
            ax.plot_surface(
                X, T, phi_sub, cmap="coolwarm", vmin=z_min, vmax=z_max,
                linewidth=0, antialiased=True,
            )

            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(t_full[0], t_full[-1])
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel(r"$\phi(x,t)$")
            ax.set_title(
                f"Kink-antikink collision, v_in = {v_in:.3f}, "
                f"t = {t_full[frame_idx]:.2f}"
            )
            ax.view_init(elev=args.elev, azim=args.azim)
            return ()

        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=1000 / args.fps, blit=False
        )

        print(f"Saving growing surface GIF to {output_path} ({n_frames} frames)...")
        anim.save(output_path, writer=PillowWriter(fps=args.fps))
        print("Done.")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

    else:
        output_path = args.output
        if output_path is None:
            output_path = base + "_surface.png"

        X, T = np.meshgrid(x, t_full)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X, T, phi_full, cmap="coolwarm", linewidth=0, antialiased=True
        )

        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel(r"$\phi(x,t)$")
        ax.set_title(f"Kink-antikink collision, v_in = {v_in:.3f}")
        ax.view_init(elev=args.elev, azim=args.azim)
        fig.colorbar(surf, shrink=0.6, aspect=12, label=r"$\phi$")

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved surface plot to {output_path}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()