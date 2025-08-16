import os
import math
import torch
import warp as wp

from rewarped.warp_env import WarpEnv
from rewarped.environment import IntegratorType
from .utils.torch_utils import normalize, quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate
from warp.sim.collide import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, mesh_sdf, plane_sdf, sphere_sdf


# @wp.kernel
# def collision_cost(
#     body_q: wp.array(dtype=wp.transform),
#     obstacle_ids: wp.array(dtype=int, ndim=2),
#     shape_X_bs: wp.array(dtype=wp.transform),
#     geo: wp.sim.ModelShapeGeometry,
#     margin: float,
#     weighting: float,
#     cost: wp.array(dtype=wp.float32),
# ):
#     env_id, obs_id = wp.tid()
#     shape_index = obstacle_ids[env_id, obs_id]

#     px = wp.transform_get_translation(body_q[env_id])

#     X_bs = shape_X_bs[shape_index]

#     # transform particle position to shape local space
#     x_local = wp.transform_point(wp.transform_inverse(X_bs), px)

#     # geo description
#     geo_type = geo.type[shape_index]
#     geo_scale = geo.scale[shape_index]

#     # evaluate shape sdf
#     d = 1e6

#     if geo_type == wp.sim.GEO_SPHERE:
#         d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
#     elif geo_type == wp.sim.GEO_BOX:
#         d = box_sdf(geo_scale, x_local)
#     elif geo_type == wp.sim.GEO_CAPSULE:
#         d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
#     elif geo_type == wp.sim.GEO_CYLINDER:
#         d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
#     elif geo_type == wp.sim.GEO_CONE:
#         d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
#     elif geo_type == wp.sim.GEO_MESH:
#         mesh = geo.source[shape_index]
#         min_scale = wp.min(geo_scale)
#         max_dist = margin / min_scale
#         d = mesh_sdf(mesh, wp.cw_div(x_local, geo_scale), max_dist)
#         d *= min_scale  # TODO fix this, mesh scaling needs to be handled properly
#     elif geo_type == wp.sim.GEO_SDF:
#         volume = geo.source[shape_index]
#         xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
#         nn = wp.vec3(0.0, 0.0, 0.0)
#         d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
#     elif geo_type == wp.sim.GEO_PLANE:
#         d = plane_sdf(geo_scale[0], geo_scale[1], x_local)

#     d = wp.max(d, 0.0)
#     if d < margin:
#         c = margin - d
#         wp.atomic_add(cost, env_id, weighting * c)


class Propeller:
    """Simple propeller data structure"""
    def __init__(self):
        self.body = 0
        self.pos = (0.0, 0.0, 0.0)
        self.dir = (0.0, 1.0, 0.0)
        self.thrust = 0.0
        self.power = 0.0
        self.diameter = 0.0
        self.height = 0.0
        self.max_rpm = 0.0
        self.max_thrust = 0.0
        self.max_torque = 0.0
        self.turning_direction = 0.0
        self.max_speed_square = 0.0


def define_propeller(
    drone: int,
    pos: tuple,
    fps: float,
    thrust: float = 10.9919,
    power: float = 0.040164,
    diameter: float = 0.2286,
    height: float = 0.01,
    max_rpm: float = 6396.667,
    turning_direction: float = 1.0,
):
    # Air density at sea level.
    air_density = 1.225  # kg / m^3

    rps = max_rpm / fps
    max_speed = rps * 2 * 3.14159  # radians / sec (using pi constant)
    rps_square = rps**2

    prop = Propeller()
    prop.body = drone
    # Use simple tuple assignment for compatibility
    prop.pos = (pos[0], pos[1], pos[2])
    prop.dir = (0.0, 1.0, 0.0)  
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / (2 * 3.14159)
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2

    return prop

class DroneParcour(WarpEnv):
    sim_name = "DroneParcour"
    env_offset = (5.0, 0.0, 5.0)

    integrator_type = IntegratorType.EULER
    sim_substeps_featherstone = 16
    featherstone_settings = dict(angular_damping=0.05, update_mass_matrix_every=sim_substeps_featherstone)

    state_tensors_names = ("body_q", "body_qd", "body_f")
    control_tensors_names = ()  # Empty - drone uses external forces, not joint actuators

    def __init__(self, num_envs=16, episode_length=1000, early_termination=True, **kwargs):
        # Extract environment parameters before calling super().__init__()
        self.device = kwargs.pop("device")
        self.drone_size = torch.tensor(kwargs.pop("drone_size", 0.2), device=self.device)
        self.arena_shape = torch.tensor(kwargs.pop("arena_shape", [4.0, 4.0, 4.0]), device=self.device)
        self.num_obstacles = kwargs.pop("num_obstacles", 5)
        self.obstacle_min_size = kwargs.pop("obstacle_min_size", 0.2)
        self.obstacle_max_size = kwargs.pop("obstacle_max_size", 0.5)
        self.termination_distance = kwargs.pop("termination_distance", 0.1)
        self.reaction_torque_ratio = kwargs.pop("reaction_torque_ratio", 0.01)

        # Extract render settings from config
        render = kwargs.pop("render", False)
        render_mode = kwargs.pop("render_mode", "none")

        # Simple observation like ant: position + velocity + target + actions = 16
        num_obs = 20
        num_act = 4  # Four thrust controls

        self.control_limits = torch.tensor([[0.1, 1.0]] * 4)

        # Now call super with only the parameters it expects
        super().__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            early_termination,
            render=render,
            render_mode=render_mode,
            **kwargs,
        )

        # Set additional attributes needed for reward computation
        self.termination_height = 0.1  # Ground collision threshold

    def create_modelbuilder(self):
        """Create the model builder with drone-specific settings"""
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.02
        return builder

    def create_articulation(self, builder):        
        props = []
        colliders = []
        crossbar_length = self.arm_specs[2]
        crossbar_height = self.drone_size * 0.05
        crossbar_width = self.drone_size * 0.05
        carbon_fiber_density = 1750.0  # kg / m^3

        # Register the drone as a rigid body in the simulation model.
        body = builder.add_body(name="drone")

        # Define the shapes making up the drone's rigid body.
        builder.add_shape_box(
            body,
            hx=crossbar_length,
            hy=crossbar_height,
            hz=crossbar_width,
            density=carbon_fiber_density,
        )
        builder.add_shape_box(
            body,
            hx=crossbar_width,
            hy=crossbar_height,
            hz=crossbar_length,
            density=carbon_fiber_density,
        )

        # Initialize the propellers.
        props.extend(
            (
                define_propeller(
                    body,
                    (crossbar_length, 0.0, 0.0),
                    1.0 / self.frame_dt,
                    turning_direction=-1.0,
                ),
                define_propeller(
                    body,
                    (-crossbar_length, 0.0, 0.0),
                    1.0 / self.frame_dt,
                    turning_direction=1.0,
                ),
                define_propeller(
                    body,
                    (0.0, 0.0, crossbar_length),
                    1.0 / self.frame_dt,
                    turning_direction=1.0,
                ),
                define_propeller(
                    body,
                    (0.0, 0.0, -crossbar_length),
                    1.0 / self.frame_dt,
                    turning_direction=-1.0,
                ),
            ),
        )

        # Initialize the colliders.
        colliders.append(
            (
                builder.add_shape_capsule(
                    -1,
                    pos=(0.5, 2.0, 0.5),
                    radius=0.15,
                    half_height=2.0,
                ),
            ),
        )
        self.props = props  # Keep as Python list for easy indexing

        # Initial position: higher up for drone flight
        builder.body_q[0] = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Start at 2m height

    def init_sim(self):
        super().init_sim()

        # self.print_model_info()

        with torch.no_grad():
            # these are used for resetting drone later
            self.start_body_q = self.state.body_q.view(self.num_envs, -1).clone()
            self.start_body_qd = self.state.body_qd.view(self.num_envs, -1).clone()

            self.start_pos = self.start_body_q[:, :3]
            self.start_rot = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion for simplicity
            self.start_rotation = torch.tensor(self.start_rot, device=self.device)

            self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
            self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device)
            self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)

            self.x_unit_tensor = self.x_unit_tensor.repeat((self.num_envs, 1))
            self.y_unit_tensor = self.y_unit_tensor.repeat((self.num_envs, 1))
            self.z_unit_tensor = self.z_unit_tensor.repeat((self.num_envs, 1))

            self.up_vec = self.y_unit_tensor.clone()
            self.heading_vec = self.x_unit_tensor.clone()
            self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

            self.basis_vec0 = self.heading_vec.clone()
            self.basis_vec1 = self.up_vec.clone()

            self.prop_controls = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

            # Initialize target positions for each environment
            self.targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            self.reset_targets()
            
        # Add the external forces hook to the control object after init
        # Store the method reference on the control object's underlying data
        # The warp_utils.py hook checks for this method and calls it if present
        if hasattr(self.control, 'data'):
            self.control.data.apply_external_forces = self.apply_drone_forces
        else:
            # Fallback: store on self and let warp_utils check the env's control
            self._apply_external_forces = self.apply_drone_forces
            # Monkey patch the control object by accessing its __dict__ directly
            try:
                self.control.__dict__['apply_external_forces'] = self.apply_drone_forces
            except (AttributeError, TypeError):
                # If monkey patching fails, we'll need to handle it in the physics step
                print("Warning: Could not add apply_external_forces to control object. Using fallback method.")
        
    def apply_drone_forces(self, model, state):
        """Apply drone thrust forces and torques to the physics state.
        This method is called by the simulation loop after clear_forces() and collisions.
        """
        # Apply forces for each propeller
        total_force = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

        for i, prop in enumerate(self.props):
            control_val = self.prop_controls[:, i]

            # Get body transform (position and rotation)
            body_pos = state.body_q[:, :3]  # Position
            body_rot = state.body_q[:, 3:]  # Quaternion

            # Convert propeller position to world coordinates
            prop_pos_tensor = torch.tensor([prop.pos[0], prop.pos[1], prop.pos[2]],
                                         device=self.device, dtype=torch.float32)

            # Apply rotation using quaternion
            qw, qx, qy, qz = body_rot[:, 3], body_rot[:, 0], body_rot[:, 1], body_rot[:, 2]

            # Rotate propeller position
            prop_world_pos = body_pos + self._quat_rotate_vector(
                torch.stack([qx, qy, qz, qw], dim=1), prop_pos_tensor
            )

            # Propeller thrust direction (always up in body frame, then rotated to world)
            thrust_dir_body = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
            thrust_dir_world = self._quat_rotate_vector(
                torch.stack([qx, qy, qz, qw], dim=1), thrust_dir_body
            )

            # Compute thrust magnitude
            thrust_magnitude = control_val * prop.max_thrust
            thrust_force = thrust_magnitude[:, None] * thrust_dir_world

            # Compute torque due to propeller position relative to COM
            r = prop_world_pos - body_pos  # Position relative to COM (assuming COM at origin)
            torque = torch.cross(r, thrust_force, dim=1)

            # Add propeller reaction torque
            turning_mask = torch.tensor([0.0, prop.turning_direction, 0.0],
                                      device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
            reaction_torque_body = turning_mask * thrust_magnitude[:, None] * self.reaction_torque_ratio
            reaction_torque_world = self._quat_rotate_vector(
                torch.stack([qx, qy, qz, qw], dim=1), reaction_torque_body
            )
            torque += reaction_torque_world

            # Accumulate forces and torques (spatial force format: [torque, force])
            total_force[:, :3] += torque
            total_force[:, 3:] += thrust_force

        # Apply to state (assuming drone is body 0 in each environment)
        state.body_f[:, :6] = total_force

    def reset_targets(self):
        """Reset target positions to random locations above termination height"""
        with torch.no_grad():
            self.targets[:, 0:3] = (
                torch.rand([self.num_envs, 3], device=self.device) * self.arena_shape - self.arena_shape / 2
            )

    # def reset_idx(self, env_ids):
    #     """Reset specified environments"""
    #     super().reset_idx(env_ids)

    #     with torch.no_grad():
    #         # Update targets for reset environments
    #         N = len(env_ids)
    #         self.targets[env_ids, 0] = torch.rand(N, device=self.device) * self.arena_size - self.arena_size / 2
    #         self.targets[env_ids, 1] = (
    #             torch.rand(N, device=self.device) * 2.0 + self.termination_height
    #         )  # [termination_height, termination_height+2.0]
    #         self.targets[env_ids, 2] = torch.rand(N, device=self.device) * self.arena_size - self.arena_size / 2

    @torch.no_grad()
    def randomize_init(self, env_ids):
        """Randomize drone spawn positions following ant pattern"""
        # For rigid body drone, use body_q instead of joint_q
        body_q = self.state.body_q.view(self.num_envs, -1)
        body_qd = self.state.body_qd.view(self.num_envs, -1)

        N = len(env_ids)

        # Set identity quaternion first
        body_q[env_ids, 3:7] = self.start_rotation.clone()

        # Add random position offset
        body_q[env_ids, 0:3] += (torch.rand(size=(N, 3), device=self.device) - 0.5) * self.arena_shape

        # Small random orientation around random axis
        angle = (torch.rand(N, device=self.device) - 0.5) * math.pi / 12.0  # ±15 degrees
        axis = torch.nn.functional.normalize(torch.rand((N, 3), device=self.device) - 0.5, dim=1)
        body_q[env_ids, 3:7] = quat_mul(body_q[env_ids, 3:7], quat_from_angle_axis(angle, axis))

        # Set small random velocities
        body_qd[env_ids, :] = 0.1 * (torch.rand(size=(N, body_qd.shape[1]), device=self.device) - 0.5)

    def pre_physics_step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clamp(actions, -1.0, 1.0)  # clamp policy outputs
        self.prop_controls = (actions + 1.0) * 0.5

    def _quat_rotate_vector(self, quat, vec):
        """Rotate vector by quaternion using PyTorch operations
        Args:
            quat: [batch_size, 4] quaternions in (x,y,z,w) format
            vec: [3] or [batch_size, 3] vector(s) to rotate
        """
        if vec.dim() == 1:
            vec = vec.unsqueeze(0).expand(quat.shape[0], -1)
        
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        q_xyz = torch.stack([qx, qy, qz], dim=1)
        qw = qw.unsqueeze(1)
        
        cross1 = torch.cross(q_xyz, vec, dim=1) + qw * vec
        cross2 = torch.cross(q_xyz, cross1, dim=1)
        return vec + 2.0 * cross2

    def compute_observations(self):
        """Extract observations following ant pattern"""
        body_q = self.state.body_q.clone()
        body_qd = self.state.body_qd.clone()

        # Extract drone position and velocity (first 6 DOF)
        drone_pos = body_q[:, :3] - self.env_offsets  # Remove environment offset
        drone_rot = body_q[:, 3:7]

        # Velocities
        lin_vel = body_qd[:, 3:6] if body_qd.shape[1] > 6 else body_qd[:, :3]
        ang_vel = body_qd[:, :3]

        # Target relative position
        target_rel = self.targets - (body_q[:, :3] - self.env_offsets)

        obs_buf = [
            drone_pos,  # 0:3   - position
            drone_rot,  # 3:7   - full quaternion orientation
            lin_vel,  # 7:10  - linear velocity
            ang_vel,  # 10:13 - angular velocity
            target_rel,  # 13:16 - target relative position
            self.actions.clone(),  # 16:20 - previous actions
        ]
        self.obs_buf = torch.cat(obs_buf, dim=-1)

    def compute_reward(self):
        """Compute rewards following ant pattern"""
        body_q = self.state.body_q.clone().view(self.num_envs, -1)
        body_qd = self.state.body_qd.clone().view(self.num_envs, -1)

        # Extract position
        drone_pos = body_q[:, :3] - self.env_offsets

        # Debug: Check if body_q is being reset to zeros
        # if torch.any(body_q[:, 1] < 0.5):  # If y position is too low
        #     print(f"ERROR: body_q is being reset to zeros every step: {body_q[:, :7]}")
        #     print(f"  This means the physics simulation is not working properly!")
        #     print(f"  The propeller forces are not being integrated into the physics state.")
            # Don't fix it - let it fail so we can see the real problem

        # Distance to target reward
        target_dist = torch.norm(drone_pos - self.targets, dim=1)
        progress_reward = -target_dist * 0.1

        # Height reward (stay above ground)
        height_reward = torch.clamp(drone_pos[:, 1] - self.termination_height, min=0.0) * 0.1

        # Control penalty
        control_penalty = torch.sum(self.actions**2, dim=-1) * 0.01

        # Target reached bonus
        target_bonus = torch.where(target_dist < 0.5, torch.ones_like(target_dist) * 2.0, torch.zeros_like(target_dist))

        rew = progress_reward + height_reward - control_penalty + target_bonus

        # Handle termination following ant pattern
        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination

        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)

        if early_termination:
            # Ground collision -> define later using collision detection
            ground_collision = torch.zeros_like(reset_buf, dtype=torch.bool)
            # Out of bounds
            out_of_bounds = torch.any(torch.abs(drone_pos) > self.arena_shape / 2, dim=1)
            # Target reached
            target_reached = target_dist < self.termination_distance

            # Debug output to understand why episodes are ending
            # if torch.any(ground_collision | out_of_bounds | target_reached):
            #     print(f"Termination triggered:")
            #     print(f"  drone_pos: {drone_pos}")
            #     print(f"  termination_height: {self.termination_height}")
            #     print(f"  ground_collision: {ground_collision}")
            #     print(f"  out_of_bounds: {out_of_bounds} (arena_shape/2: {self.arena_shape / 2})")
            #     print(f"  target_reached: {target_reached} (target_dist: {target_dist})")                

            terminated = ground_collision | out_of_bounds | target_reached
            reset = torch.where(terminated, torch.ones_like(reset), reset)
        else:
            terminated = torch.zeros_like(reset, dtype=torch.bool)

        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = rew, reset, terminated, truncated
