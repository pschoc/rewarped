from dataclasses import dataclass
from pathlib import Path
from .base import BasePhysicsConfig


@dataclass(kw_only=True)
class DexDeformPhysicsConfig(BasePhysicsConfig, name='dexdeform'):
    path: str = str(Path(__file__).parent.resolve() / 'templates' / 'plasticine.py')
    elasticity: str = 'sigma'

    material: str = 'plasticine'
    # E: float = 4.0e3
    # nu: float = 0.2
    # yield_stress: float = 130.0

    youngs_modulus_log: float = 13.0
    poissons_ratio: float = 0.25
    yield_stress: float = 3e4
