import os

import torch
from omegaconf import OmegaConf

import warp as wp

from ...environment import IntegratorType, run_env
from ...warp_env import WarpEnv
from .utils.interface import DEFAULT_INITIAL_QPOS

TASK_LENGTHS = {
    # "folding": 250,
    # "rope": 250,
    # "bun": 250,
    # "dumpling": 250,
    # "wrap": 500,
    "flip": 500,
    "lift": 100,
}


class Hand(WarpEnv):
    sim_name = "Hand" + "DexDeform"
    env_offset = (5.0, 5.0, 0.0)

    eval_fk = True
    eval_ik = False

    # integrator_type = IntegratorType.EULER
    # sim_substeps_euler = 16
    # euler_settings = dict(angular_damping=0.0)

    integrator_type = IntegratorType.FEATHERSTONE
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.0, update_mass_matrix_every=sim_substeps_featherstone)

    up_axis = "Z"
    ground_plane = True

    state_tensors_names = ("joint_q", "joint_qd")
    control_tensors_names = ("joint_act",)

    def __init__(self, num_envs=8, episode_length=1000, early_termination=True, **kwargs):
        num_obs = 0
        num_act = 0
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.action_scale = 1.0

        self.dexdeform_cfg = self.create_cfg_dexdeform()
        print(self.dexdeform_cfg)

    def create_cfg_dexdeform(self):
        cfg_file = os.path.join(os.path.dirname(__file__), f"env_cfgs/{self.task_name}.yml")
        dexdeform_cfg = OmegaConf.load(open(cfg_file, "r"))

        dexdeform_cfg.SIMULATOR.gravity = eval(dexdeform_cfg.SIMULATOR.gravity)
        for i, shape in enumerate(dexdeform_cfg.SHAPES):
            dexdeform_cfg.SHAPES[i].init_pos = eval(shape.init_pos)
            if "width" in shape:
                dexdeform_cfg.SHAPES[i].width = eval(shape.width)
        for i, manipulator in enumerate(dexdeform_cfg.MANIPULATORS):
            dexdeform_cfg.MANIPULATORS[i].init_pos = eval(manipulator.init_pos)
            dexdeform_cfg.MANIPULATORS[i].init_rot = eval(manipulator.init_rot)

        return dexdeform_cfg

    def create_modelbuilder(self):
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.05
        return builder

    # def create_env(self, builder):
    #     self.create_articulation(builder)

    def create_articulation(self, builder):
        raise NotImplementedError

    def init_sim(self):
        super().init_sim()
        # self.print_model_info()

        with torch.no_grad():
            self.joint_act = wp.to_torch(self.model.joint_act).view(self.num_envs, -1).clone()
            self.joint_act_indices = ...

    def reset_idx(self, env_ids):
        if self.early_termination:
            raise NotImplementedError
        else:
            super().reset_idx(env_ids)

    def randomize_init(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clip(actions, -1.0, 1.0)
        self.actions = actions
        acts = self.action_scale * actions

        if self.joint_act_indices is ...:
            self.control.assign("joint_act", acts.flatten())
        else:
            joint_act = self.scatter_actions(self.joint_act, self.joint_act_indices, acts)
            self.control.assign("joint_act", joint_act.flatten())
        # self.control.assign("joint_act", wp.to_torch(self.model.joint_act).clone().flatten())

    def compute_observations(self):
        self.obs_buf = {}
        raise NotImplementedError

    def compute_reward(self):
        rew = None
        raise NotImplementedError

        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination
        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)
        if early_termination:
            raise NotImplementedError
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)
        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated


if __name__ == "__main__":
    run_env(Hand)
