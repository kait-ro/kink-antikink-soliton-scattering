import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from run_collision import run_one_collision

SWEEP_VALUES = {
    "0.10": 0.10,
    "0.20": 0.20,
    "0.35": 0.35,
    "0.45": 0.45,
    "0.8": 0.8,
}


def parse_args():
    parser = argparse.ArgumentParser(description=("Run the hand-picked v_in sweep."))
    parser.add_argument(
        "--d0",
        type=float,
        default=30.0,
        help="incoming speed of each soliton in natural units",
    )
    parser.add_argument(
        "--t_final",
        type=float,
        default=None,
        help="total simulated time",
    )
    parser.add_argument(
        "--t_buffer",
        type=float,
        default=40.0,
        help="Extra time added after the expected collision time",
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


def main():
    args = parse_args()

    results = {}
    for label, v_in in SWEEP_VALUES.items():
        t_final = args.t_final
        if t_final is None:
            t_final = 2 * args.d0 / v_in + args.t_buffer

        print("=" * 60)
        print(f"sweep: {label}  (v_in={v_in}, t_final={t_final:.1f})")
        print("=" * 60)

        filename = run_one_collision(
            v_in=v_in,
            d0=args.d0,
            t_final=t_final,
            save_every=args.save_every,
            output_dir=args.output_dir,
        )
        results[label] = filename

    print("Sweep complete. Saved runs:")
    for label, filename in results.items():
        print(f"{label:32s} -> {filename}")


if __name__ == "__main__":
    main()
