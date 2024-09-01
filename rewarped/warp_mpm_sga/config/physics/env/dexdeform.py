from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import DexDeformPhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class DexDeformEnvConfig(BaseEnvConfig, name='dexdeform'):
    rho: float = 1000.0
    clip_bound: float = 0.5

    physics: DexDeformPhysicsConfig = field(default_factory=DexDeformPhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)
