from dataclasses import dataclass, field
from .base import BaseEnvConfig
from .physics import PlasticinePhysicsConfig
from .shape import CubeShapeConfig
from .vel import ZeroVelConfig

@dataclass(kw_only=True)
class PlasticineEnvConfig(BaseEnvConfig, name='plasticine'):
    rho: float = 1000.0
    clip_bound: float = 0.5

    physics: PlasticinePhysicsConfig = field(default_factory=PlasticinePhysicsConfig)
    shape: CubeShapeConfig = field(default_factory=CubeShapeConfig)
    vel: ZeroVelConfig = field(default_factory=ZeroVelConfig)
