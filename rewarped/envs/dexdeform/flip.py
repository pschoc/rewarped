from ...environment import run_env
from .hand import Hand


class Flip(Hand):
    sim_name = "Flip" + "DexDeform"

    TRAJ_OPT = False
    LOAD_DEMO = True

    def __init__(self, task_name="flip", episode_length=300, **kwargs):
        super().__init__(task_name=task_name, episode_length=episode_length, **kwargs)


if __name__ == "__main__":
    run_env(Flip, no_grad=False)
