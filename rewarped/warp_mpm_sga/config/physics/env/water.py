from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import WaterPhysicsConfig
from .shape import CubeHDShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class WaterEnvConfig(BaseEnvConfig, name='water'):
    rho: float = 3e3
    clip_bound: float = 0.5

    physics: WaterPhysicsConfig = field(default_factory=WaterPhysicsConfig)
    shape: CubeHDShapeConfig = field(default_factory=CubeHDShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)
