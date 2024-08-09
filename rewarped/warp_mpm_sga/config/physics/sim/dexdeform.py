from dataclasses import dataclass
from .base import BaseSimConfig

@dataclass(kw_only=True)
class DexDeformSimConfig(BaseSimConfig, name="dexdeform"):
    num_steps: int = 1000
    gravity: tuple[float, float, float] = (0.0, -90.0, 0.0)
    bc: str = 'dexdeform'
    num_grids: int = 64
    dt: float = 0.5e-4
    bound: int = 3
    eps: float = 1e-15
    skip_frames: int = 1

    body_friction: float = 0.5
    body_softness: float = 666.
