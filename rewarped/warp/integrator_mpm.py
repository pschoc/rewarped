from warp.sim.integrator import Integrator
from warp.sim.model import Control, Model, State


class MPMIntegrator(Integrator):
    def __init__(self):
        pass

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        pass
