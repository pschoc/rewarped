import warp as wp
import warp.sim

from .warp.import_mjcf import parse_mjcf

wp.sim.parse_mjcf = parse_mjcf

from .warp.import_urdf import parse_urdf

wp.sim.parse_urdf = parse_urdf

from .warp.model import ModelBuilder

wp.sim.ModelBuilder = ModelBuilder

wp.init()
