from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    lam: float = 2.0
    v: float = 1.0
    L: float = 60.0
    dx: float = 0.08
    dt: float = 0.04
    damping: float = 0.35 
    sponge_width: float = 8.0