import os

import numpy as np
import torch
from omegaconf import OmegaConf

import warp as wp

from ...environment import IntegratorType, run_env
from ...mpm_warp_env_mixin import MPMWarpEnvMixin
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


class Hand(MPMWarpEnvMixin, WarpEnv):
    sim_name = "Hand" + "DexDeform"
    env_offset = (1.0, 0.0, 0.0)
    env_offset_correction = False

    eval_fk = True
    kinematic_fk = True
    eval_ik = False

    integrator_type = IntegratorType.MPM
    sim_substeps_mpm = 40

    up_axis = "Y"
    ground_plane = True

    state_tensors_names = ("joint_q", "body_q") + ("mpm_x", "mpm_v", "mpm_C", "mpm_F_trial", "mpm_F", "mpm_stress")
    control_tensors_names = ("joint_act",)

    def __init__(self, task_name="flip", num_envs=2, episode_length=300, early_termination=False, **kwargs):
        num_obs = 0
        num_act = 24
        if episode_length == -1:
            episode_length = TASK_LENGTHS[task_name]
        super().__init__(num_envs, num_obs, num_act, episode_length, early_termination, **kwargs)

        self.task_name = task_name
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

    def create_builder(self):
        builder = super().create_builder()
        self.create_builder_mpm(builder)
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

    def create_cfg_mpm(self, mpm_cfg):
        mpm_cfg.update(["--physics.sim", "dexdeform"])
        mpm_cfg.update(["--physics.env", "dexdeform"])

        # mpm_cfg.update(["--physics.sim.gravity", str(tuple(self.dexdeform_cfg.SIMULATOR.gravity))])
        mpm_cfg.update(["--physics.sim.body_friction", str(self.dexdeform_cfg.SIMULATOR.hand_friction)])

        if self.task_name == "lift":
            shape_cfg = self.dexdeform_cfg.SHAPES[0]
            init_pos, width = shape_cfg.init_pos, shape_cfg.width
            mpm_cfg.update(["--physics.env.shape", "cube"])
            mpm_cfg.update(["--physics.env.shape.center", str(tuple(init_pos))])
            mpm_cfg.update(["--physics.env.shape.size", str(tuple(width))])
            mpm_cfg.update(["--physics.env.shape.resolution", str(20)])
            # TODO: change resolution based on num_particles
        elif self.task_name == "flip":
            mpm_cfg.update(["--physics.env.shape", "cylinder_dexdeform"])
            # mpm_cfg.update(["--physics.env.shape.num_particles", str(self.dexdeform_cfg.SIMULATOR.n_particles)])

            # lowering to decrease memory requirements for parallel envs
            mpm_cfg.update(["--physics.sim.num_grids", str(48)])
            mpm_cfg.update(["--physics.env.shape.num_particles", str(2500)])
        else:
            raise NotImplementedError

        print(mpm_cfg)
        return mpm_cfg

    def create_model(self):
        model = super().create_model()
        self.create_model_mpm(model)
        return model

    def init_sim(self):
        self.init_sim_mpm()
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

        if self.kinematic_fk:
            # ensure joint limit
            joint_q = self.state.joint_q.clone().view(self.num_envs, -1)
            joint_q = joint_q.detach()
            acts = torch.clip(acts, self.joint_limit_lower - joint_q, self.joint_limit_upper - joint_q)

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

    def render(self, state=None):
        if self.renderer is not None:
            with wp.ScopedTimer("render", False):
                self.render_time += self.frame_dt
                self.renderer.begin_frame(self.render_time)
                # render state 1 (swapped with state 0 just before)
                self.renderer.render(state or self.state_1)

                # render mpm particles
                particle_q = self.state.mpm_x
                if isinstance(particle_q, torch.Tensor):
                    particle_q = particle_q.detach().cpu().numpy()
                else:
                    particle_q = particle_q.numpy()
                particle_radius = 7.5e-3
                particle_color = (0.875, 0.451, 1.0)  # 0xdf73ff
                self.renderer.render_points("particle_q", particle_q, radius=particle_radius, colors=particle_color)

                self.renderer.end_frame()


if __name__ == "__main__":
    run_env(Hand)
