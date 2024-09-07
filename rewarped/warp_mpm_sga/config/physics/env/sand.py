from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import SandPhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class SandEnvConfig(BaseEnvConfig, name='sand'):
    rho: float = 1e3
    clip_bound: float = 0.5

    physics: SandPhysicsConfig = field(default_factory=SandPhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)
