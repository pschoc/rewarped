import os

import numpy as np
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

    def create_env(self, builder):
        mode = self.dexdeform_cfg.SIMULATOR.mode
        if mode == "dual":
            raise NotImplementedError

        for hand_cfg in self.dexdeform_cfg.MANIPULATORS:
            scale = self.dexdeform_cfg.SIMULATOR.scale
            fixed_base = self.dexdeform_cfg.SIMULATOR.get("fixed_base", False)
            self.create_hand(builder, mode, hand_cfg, scale, fixed_base)

    def create_hand(self, builder, mode, hand_cfg, scale, fixed_base):
        init_pos, init_rot, init_qpos = hand_cfg.init_pos, hand_cfg.init_rot, hand_cfg.init_qpos
        init_rot[0] += -np.pi / 2
        if mode == "lh":
            asset_file = "left_hand.xml"
        elif mode == "rh":
            asset_file = "right_hand.xml"
        else:
            raise ValueError

        xform = wp.transform(init_pos, wp.quat_rpy(*init_rot))
        wp.sim.parse_mjcf(
            os.path.join(self.asset_dir, f"shadow/{asset_file}"),
            builder,
            xform=xform,
            floating=not fixed_base,
            stiffness=1000.0,
            damping=0.0,
            # parse_meshes=False,
            # ignore_names=["C_forearm"],
            visual_classes=["D_Vizual"],
            collider_classes=["DC_Hand"],
            enable_self_collisions=False,
            scale=scale,
            reduce_capsule_height_by_radius=True,
            collapse_fixed_joints=True,
            # force_show_colliders=True,
            # hide_visuals=True,
        )

        if init_qpos == "default":
            for i, joint_name in enumerate(builder.joint_name):
                builder.joint_q[i] = DEFAULT_INITIAL_QPOS[f"robot0:{joint_name}"]
        elif init_qpos == "zero":
            for i, joint_name in enumerate(builder.joint_name):
                builder.joint_q[i] = 0.0
        else:
            raise ValueError

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
