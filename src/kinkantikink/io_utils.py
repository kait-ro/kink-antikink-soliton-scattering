import numpy as np

from .parameters import Params


def save_run(x, t, phi_history, p: Params, v_in, epsilon_history=None, output_dir="data"):
    filename = (
        f"{output_dir}/run_vin{v_in}_lam{p.lam}_v{p.v}_L{p.L}"
        f"_dx{p.dx}_dt{p.dt}_damping{p.damping}_sw{p.sponge_width}.npz"
    )

    save_kwargs = {
        "x": x,
        "t": t,
        "phi_history": phi_history,
        "v_in": v_in,
        "lam": p.lam,
        "v": p.v,
        "L": p.L,
        "dx": p.dx,
        "dt": p.dt,
        "damping": p.damping,
        "sponge_width": p.sponge_width,
    }
    if epsilon_history is not None:
        save_kwargs["epsilon_history"] = epsilon_history

    np.savez(filename, **save_kwargs)
    return filename


def load_run(filepath):
    data = np.load(filepath)

    run = {
        "x": data["x"],
        "t": data["t"],
        "phi_history": data["phi_history"],
        "v_in": data["v_in"],
        "lam": data["lam"],
        "v": data["v"],
        "L": data["L"],
        "dx": data["dx"],
        "dt": data["dt"],
        "damping": data["damping"],
        "sponge_width": data["sponge_width"],
    }
    if "epsilon_history" in data:
        run["epsilon_history"] = data["epsilon_history"]

    return run