# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
import os
import math
import torch
import warp as wp
import warp.sim  # ensure wp.sim types are loaded
import numpy as np
from gym import spaces

from rewarped.warp_env import WarpEnv
from rewarped.warp.model_monkeypatch import Model_control
from rewarped.environment import IntegratorType
from .utils.torch_utils import normalize, quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate, euler_to_rotation_matrix, quat_from_euler_ypr, quat_to_yaw, quat_error_to_axis_angle
from warp.sim.collide import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, mesh_sdf, plane_sdf, sphere_sdf

import random

class SimulationView:
    """View class to access drone and obstacle bodies from the full simulation state"""
    
    def __init__(self, model, num_envs):
        self.model = model
        self.device = model.device
        self.num_envs = num_envs

        # Body indices: [drone, object, object, ..., object]
        self.drone_indices = list(range(0,1))

        self.num_mobile_obstacles = model.body_count - 1
        self.obstacle_indices = list(range(1, 1 + self.num_mobile_obstacles))

        # Create warp arrays for efficient GPU access
        self._obstacle_indices_wp = wp.array(self.obstacle_indices, dtype=int, device=self.device) if self.num_mobile_obstacles > 0 else None
        self._drone_indices_wp = wp.array(self.drone_indices, dtype=int, device=self.device)        
        
    def get_drone_positions(self, state):
        """Get drone positions from full state"""
        return state.body_q[self.drone_indices, :3]

    def get_drone_orientations(self, state):
        """Get drone orientations from full state"""
        return state.body_q[self.drone_indices, 3:7]

    def get_drone_velocities(self, state):
        """Get drone linear velocities from full state"""
        return state.body_qd[self.drone_indices, 3:6]

    def get_drone_angular_velocities(self, state):
        """Get drone angular velocities from full state"""
        return state.body_qd[self.drone_indices, :3]

    def get_drone_states(self, state):
        """Get full drone body_q and body_qd"""
        body_q = state.body_q[self.drone_indices]
        body_qd = state.body_qd[self.drone_indices]
        return body_q, body_qd
    
    def get_drone_force(self, state):
        """Get full drone body_q and body_qd"""
        body_f = state.body_f[self.drone_indices]
        return body_f
    
    def set_drone_positions(self, state, positions):
        """Set drone positions in full state"""
        state.body_q[self.drone_indices, :3] = positions
    
    def set_drone_orientations(self, state, orientations):
        """Set drone orientations in full state"""
        state.body_q[self.drone_indices, 3:7] = orientations

    def set_drone_velocities(self, state, velocities):
        """Set drone linear velocities in full state"""
        state.body_qd[self.drone_indices, 3:6] = velocities

    def set_drone_angular_velocities(self, state, angular_velocities):
        """Set drone angular velocities in full state"""
        state.body_qd[self.drone_indices, :3] = angular_velocities

    def get_obstacle_positions(self, state):
        """Get obstacle positions from full state"""
        if self.num_mobile_obstacles == 0:
            return None
        return state.body_q[self.obstacle_indices, :3]

    def get_obstacle_orientations(self, state):
        """Get obstacle orientations from full state"""
        if self.num_mobile_obstacles == 0:
            return None
        return state.body_q[self.obstacle_indices, 3:7]


class Propeller:
    """Physics-based propeller model using thrust and drag coefficients"""
    def __init__(self):
        self.body = 0        
        self.dir = (0.0, 1.0, 0.0)
        self.diameter = 0.0
        self.k_f = 0.0  # thrust coefficient
        self.k_d = 0.0  # drag coefficient
        self.turning_direction = 0.0
        self.moment_of_inertia = 0.0  # propeller moment of inertia around spin axis


def define_propeller(
    drone: int,    
    diameter: float = 0.2286,  # diameter in meters
    turning_direction: float = 1.0,
):
    """
    Define propeller using first-principles aerodynamics.
    
    Thrust: T = k_f * omega²
    Drag torque: Q = k_d * omega²
    
    where k_f and k_d are aerodynamic coefficients based on:
    - Air density ρ
    - Propeller diameter D
    - Propeller pitch P (affects thrust efficiency)
    - Thrust coefficient CT and power coefficient CP from literature
    """
    # Air density at sea level
    rho = 1.225  # kg/m³
    
    # For typical quadcopter propellers, use reasonable CT and CP values
    # CT ≈ 0.1-0.15, CP ≈ 0.05-0.08 for efficient props
    # Pitch affects thrust efficiency: higher pitch = higher thrust per RPM
    C_T = 0.15  # thrust coefficient
    C_P = 0.05  # power coefficient

    C_Q = C_P / (2.0 * 3.14159)

    k_f = C_T * rho * (diameter ** 4) / (4.0 * 3.14159**2)
    k_d = C_Q * rho * (diameter ** 5) / (4.0 * 3.14159**2)
   
    prop = Propeller()
    prop.body = drone    
    prop.dir = (0.0, 1.0, 0.0)
    prop.diameter = diameter
    prop.k_f = k_f
    prop.k_d = k_d
    prop.turning_direction = turning_direction    
    
    return prop

@wp.func
def sdf_shape_distance_world(
    x_world: wp.vec3,
    shape_index: int,
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: wp.sim.ModelShapeGeometry,
) -> float:
    X_bs = shape_X_bs[shape_index]
    x_local = wp.transform_point(wp.transform_inverse(X_bs), x_world)

    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]

    d = 1.0e6
    if geo_type == wp.sim.GEO_SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
    elif geo_type == wp.sim.GEO_BOX:
        d = box_sdf(geo_scale, x_local)
    elif geo_type == wp.sim.GEO_CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
    elif geo_type == wp.sim.GEO_MESH:
        mesh = geo.source[shape_index]
        min_scale = wp.min(geo_scale)
        max_dist = 0.25 / min_scale
        d = mesh_sdf(mesh, wp.cw_div(x_local, geo_scale), max_dist) * min_scale
    elif geo_type == wp.sim.GEO_SDF:
        volume = geo.source[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
    elif geo_type == wp.sim.GEO_PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)

    return d


@wp.kernel
def collision_distance_kernel(
    body_q: wp.array(dtype=wp.transform),
    indices_shape_with_collision: wp.array(dtype=int),    
    drone_body_index: wp.array(dtype=int),
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: wp.sim.ModelShapeGeometry,      
    shape_body: wp.array(dtype=int),
    depths: wp.array(dtype=float, ndim=2),    
):
    env_id, obs_id = wp.tid()
    shape_index = indices_shape_with_collision[obs_id]
    drone_idx = drone_body_index[env_id]

    if shape_body[shape_index] == drone_idx:
        d = float(1.0e6) # max val                
    else:
        px = wp.transform_get_translation(body_q[drone_idx])
        d = sdf_shape_distance_world(px, shape_index, shape_X_bs, geo)

    d = wp.max(d, 0.0)
    
    depths[env_id, obs_id] = d


@wp.kernel
def render_depth_kernel(
    body_q: wp.array(dtype=wp.transform),
    drone_body_index: wp.array(dtype=int),
    shape_X_bs: wp.array(dtype=wp.transform),
    geo: wp.sim.ModelShapeGeometry,
    obstacle_ids: wp.array(dtype=int),    
    cam_dirs_local: wp.array(dtype=wp.vec3),
    cam_pos_local: wp.vec3,        
    min_depth: float,
    max_depth: float,
    depths: wp.array(dtype=float, ndim=2),  # [num_envs, width*height]
):
    env_id, pix = wp.tid()

    # fetch body transform for this env
    body_idx = drone_body_index[env_id]
    tf_b = body_q[body_idx]

    # camera origin and ray direction in world
    ro = wp.transform_point(tf_b, cam_pos_local)
    rd = wp.normalize(wp.transform_vector(tf_b, cam_dirs_local[pix]))

    # sphere tracing
    t = min_depth
    hit_t = max_depth

    # loop bounds
    MAX_STEPS = 64
    EPS_HIT = 1.0e-3

    # number of obstacle shapes for this env
    # obstacle_ids is padded with -1 for invalid slots
    for step in range(MAX_STEPS):
        if t > max_depth:
            break

        p = ro + rd * t

        # evaluate min signed distance over all shapes (must be dynamic in a dynamic loop)
        d_min = float(1.0e6)

        # iterate shapes
        for s in range(len(obstacle_ids)):
            shape_index = obstacle_ids[s]
            if shape_index < 0:
                continue
            d = sdf_shape_distance_world(p, shape_index, shape_X_bs, geo)            
            d = wp.max(d, 0.0)
            d_min = wp.min(d_min, d)

        if d_min < EPS_HIT:
            hit_t = t
            break

        # conservative step
        if d_min > 0.0:
            t = t + d_min
        else:
            # guard against zero step
            t = t + 1.0e-3

    depths[env_id, pix] = wp.clamp(hit_t, min_depth, max_depth)

@wp.kernel
def apply_drone_forces_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    drone_body_index: wp.array(dtype=int),
    drone_force_torque_mapping: wp.array(dtype=float, ndim=2),  # shape [6, 6]
    body_force: wp.array(dtype=float, ndim=2),  # shape [num_envs, 6]
    max_rpm: float,
):
    env_id = wp.tid()
    body_idx = drone_body_index[env_id]

    tf = body_q[body_idx]
    
    rpm_scale = max_rpm * 2.0 * 3.14159 / 60.0

    # simple drag coefficients
    cd_linear = 0.001
    cd_angular = 0.001
    
    force_x = 0.0
    force_y = 0.0  
    force_z = 0.0
    torque_x = 0.0
    torque_y = 0.0
    torque_z = 0.0
    
    # calculate propulsive forces
    for j in range(6):  # 6 motors
        rpm_cmd = body_force[env_id, j]              
        squared_omega = (rpm_cmd * rpm_scale) * (rpm_cmd * rpm_scale)        

        if rpm_cmd < 0.0:  
            squared_omega *= -1.0

        # Matrix multiplication: result[i] = sum(mapping[i,j] * squared_omega[j])
        force_x += drone_force_torque_mapping[0, j] * squared_omega
        force_y += drone_force_torque_mapping[1, j] * squared_omega
        force_z += drone_force_torque_mapping[2, j] * squared_omega
        torque_x += drone_force_torque_mapping[3, j] * squared_omega
        torque_y += drone_force_torque_mapping[4, j] * squared_omega
        torque_z += drone_force_torque_mapping[5, j] * squared_omega
    
    force_body_prop = wp.vec3(force_x, force_y, force_z)    
    torque_body_prop = wp.vec3(torque_x, torque_y, torque_z)        
    force_world_prop = wp.transform_vector(tf, force_body_prop)    
    # torque_world_prop = wp.transform_vector(tf, torque_body_prop)    
    torque_world_prop = torque_body_prop
    # aerodynamic forces
    body_qd_val = body_qd[body_idx]
    drag_force_x = -cd_linear * body_qd_val[3] * abs(body_qd_val[3])
    drag_force_y = -cd_linear * body_qd_val[4] * abs(body_qd_val[4])
    drag_force_z = -cd_linear * body_qd_val[5] * abs(body_qd_val[5])
    drag_torque_x = -cd_angular * body_qd_val[0] * abs(body_qd_val[0])
    drag_torque_y = -cd_angular * body_qd_val[1] * abs(body_qd_val[1])
    drag_torque_z = -cd_angular * body_qd_val[2] * abs(body_qd_val[2])

    # add to body forces
    sf = body_f[body_idx]
    updated_forces_torques_world = wp.spatial_vector(
        sf[0] + torque_world_prop[0] + drag_torque_x,
        sf[1] + torque_world_prop[1] + drag_torque_y,
        sf[2] + torque_world_prop[2] + drag_torque_z,
        sf[3] + force_world_prop[0] + drag_force_x,
        sf[4] + force_world_prop[1] + drag_force_y,
        sf[5] + force_world_prop[2] + drag_force_z,
    )
    
    body_f[body_idx] = updated_forces_torques_world

class DroneParcour(WarpEnv):
    sim_name = "DroneParcour"    

    state_tensors_names = ("body_q", "body_qd")    
    control_tensors_names = ("body_force",)

    def __init__(self, num_envs=16, episode_length=1000, early_termination=True, **kwargs):

        # warp to body transformation (going from a stupid y up to a proper z down coordinate system)        
        self.quat_BBtilde = wp.quat_rpy(math.pi/2, 0.0, 0.0)
        
        # Extract environment parameters before calling super().__init__()
        device = kwargs.get("device", "cuda:0")       
        max_episode_length = kwargs.pop("max_episode_length", episode_length)
        early_termination = kwargs.pop("early_termination", early_termination)
        self.render_fps = kwargs.pop("render_fps", 10)  # Frames per second for rendering
        self.integrator_type = IntegratorType(kwargs.pop("integrator", IntegratorType.XPBD))  # Default to XPBD integrator
        self.separate_collision_group_per_env = kwargs.pop("separate_collision_group_per_env", False)
        self.use_graph_capture = kwargs.pop("use_graph_capture", True)
        self.use_depth_observations = kwargs.pop("use_depth_observations", False)
        self.use_collision_distances = kwargs.pop("use_collision_distances", False)
        self.debug_pov = kwargs.pop("debug_pov", False)
        self.ground_plane = kwargs.pop("ground_plane", True)
        self.has_static_obstacles = kwargs.pop("has_static_obstacles", False)
        self.no_env_offset = kwargs.pop("no_env_offset", False)

        frame_fps = kwargs.pop("frame_fps", 4)  # Simulation steps per second
        self.frame_dt = 1.0 / frame_fps  # seconds per frame
        solver_fps = kwargs.pop("solver_fps", 400)  # Solver steps per second        
        self.sim_substeps = round(solver_fps / frame_fps)
        self.episode_duration = kwargs.pop("max_episode_duration", max_episode_length)  # seconds

        self.gravity = kwargs.pop("gravity", -9.81)  # m/s²
        self.max_target_lin_vel = kwargs.pop("max_target_lin_vel", 1.0)  # m/s
        self.max_target_ang_vel = kwargs.pop("max_target_ang_vel", 1.0)  # rad/s

        # Define bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.spawn_bounds = torch.tensor(kwargs.pop("spawn_bounds", [[-2.0, 2.0], [1.0, 3.0], [-2.0, 2.0]]), device=device)
        self.arena_bounds = torch.tensor(kwargs.pop("arena_bounds", [[-8.0, 8.0], [0.0, 6.0], [-8.0, 8.0]]), device=device)
        self.target_bounds = torch.tensor(kwargs.pop("target_bounds", [[-6.0, 6.0], [1.0, 5.0], [-6.0, 6.0]]), device=device)
        self.room_bounds = torch.tensor(kwargs.pop("room_bounds", [[-10.0, 10.0], [0.0, 4.0], [-10.0, 10.0]]), device=device)

        self.num_static_obstacles = kwargs.pop("num_static_obstacles", 5)
        self.obstacle_min_size = kwargs.pop("obstacle_min_size", 0.2)
        self.obstacle_max_size = kwargs.pop("obstacle_max_size", 0.5)
        self.termination_distance = kwargs.pop("termination_distance", 0.2)

        self.is_render_targets = kwargs.pop("is_render_targets", False)  # Whether to render target coordinate frames

        # spawning parameters
        self.initial_drone_attitude_euler_deg = torch.tensor(kwargs.pop("initial_drone_attitude_euler_deg", [0.0, 0.0, 0.0]), device=device)  # [roll, pitch, yaw] in degrees
        self.initial_drone_lin_vel = torch.tensor(kwargs.pop("initial_drone_lin_vel", [0.0, 0.0, 0.0]), device=device)  # [vx, vy, vz] in m/s
        self.initial_drone_ang_vel = torch.tensor(kwargs.pop("initial_drone_ang_vel", [0.0, 0.0, 0.0]), device=device)  # [wx, wy, wz] in rad/s

        self.three_sigma_initial_attitude_deg = torch.tensor(kwargs.pop("three_sigma_initial_attitude_deg", [10.0, 10.0, 10.0]), device=device)  # 10 degrees
        self.three_sigma_initial_lin_vel = torch.tensor(kwargs.pop("three_sigma_initial_lin_vel", [0.5, 0.5, 0.5]), device=device)  # [vx, vy, vz] in m/s
        self.three_sigma_initial_ang_vel = torch.tensor(kwargs.pop("three_sigma_initial_ang_vel", [0.1, 0.1, 0.1]), device=device)  # [wx, wy, wz] in rad/s

        # target parameters
        self.target_attitude_euler_deg = torch.tensor(kwargs.pop("target_attitude_euler_deg", [0.0, 0.0, 0.0]), device=device)  # [roll, pitch, yaw] in degrees
        self.target_lin_vel = torch.tensor(kwargs.pop("target_lin_vel", [0.0, 0.0, 0.0]), device=device)  # [vx, vy, vz] in m/s
        self.target_ang_vel = torch.tensor(kwargs.pop("target_ang_vel", [0.0, 0.0, 0.0]), device=device)  # [wx, wy, wz] in rad/s

        self.three_sigma_target_attitude_deg = torch.tensor(kwargs.pop("three_sigma_target_attitude_deg", [0.0, 0.0, 0.0]), device=device)  # 0 degrees
        self.three_sigma_target_lin_vel = torch.tensor(kwargs.pop("three_sigma_target_lin_vel", [0.0, 0.0, 0.0]), device=device)  # [vx, vy, vz] in m/s
        self.three_sigma_target_ang_vel = torch.tensor(kwargs.pop("three_sigma_target_ang_vel", [0.0, 0.0, 0.0]), device=device)  # [wx, wy, wz] in rad/s

        # Drone construction parameters
        self.density_carbon = kwargs.pop("density_carbon", 1600.0)  # kg/m³
        self.density_aluminum = kwargs.pop("density_aluminum", 2700.0)  # kg/m³
        self.arm_specs = kwargs.pop("arm_specs", [0.020, 0.002, 0.400])  # [diameter, thickness, span] in meters
        self.body_dimensions = kwargs.pop("body_dimensions", [0.10, 0.10, 0.05])  # [length, width, height] in meters
        self.body_mass = kwargs.pop("body_mass", 0.600)    # 600g
        
        # motor specs
        self.motor_specs = kwargs.pop("motor_specs", [0.07, 0.02, 0.095])  # [diameter, thickness, weight in kg]
        self.tilt_angle_deg = kwargs.pop("tilt_angle_deg", 20.0)        

        # Propeller specifications (now in inches)
        self.prop_diameter_inch = kwargs.pop("prop_diameter_inch", 12)    # 12 inch diameter
        self.prop_pitch_inch = kwargs.pop("prop_pitch_inch", 4)          # 4 inch pitch
        self.prop_thickness = kwargs.pop("prop_thickness", 0.003)        # 3mm thick 
        self.prop_weight = kwargs.pop("prop_weight", 0.030)            # 30g per propeller       
        
        # Battery and motor specifications
        self.lipo_cells = kwargs.pop("lipo_cells", 6)                    # 6S battery
        self.motor_kv = kwargs.pop("motor_kv", 170)                      # 170 KV
        self.nominal_cell_voltage = kwargs.pop("nominal_cell_voltage", 3.7)  # 3.7V per cell
      
        # Evaluation parameters (used by agent, not environment)
        self.num_eval_episodes = kwargs.pop("num_eval_episodes", 10)  # Remove from kwargs but don't need to store
        
        # Convert to metric for internal calculations
        self.prop_diameter = self.prop_diameter_inch * 0.0254  # inches to meters
        self.prop_pitch = self.prop_pitch_inch * 0.0254       # inches to meters

        # Extract render settings from config
        render = kwargs.pop("render", False)
        render_mode = kwargs.pop("render_mode", "none")

        # Depth camera parameters
        self.cam_width = kwargs.pop("cam_width", 48)
        self.cam_height = kwargs.pop("cam_height", 32)
        self.cam_fov_x_deg = kwargs.pop("cam_fov_x_deg", 90.0)
        self.cam_fov_y_deg = kwargs.pop("cam_fov_y_deg", 60.0)
        self.cam_min_depth = kwargs.pop("cam_min_depth", 1.0)
        self.cam_max_depth = kwargs.pop("cam_max_depth", 8.0)
        # Mounted slightly in front of body center along +X (forward)
        self.cam_offset_local = kwargs.pop(
            "cam_offset_local",
            [self.body_dimensions[0] / 2.0 + 0.02, 0.0, 0.0],
        )
        
        # Simple observation like ant: position + velocity + target + actions = 16
        if self.use_depth_observations:
            num_obs = {
                'obs': 13,
                'obs_depth':  (self.cam_width, self.cam_height) if self.use_depth_observations else None
            }            
        else:
            num_obs = {
                'obs': 13,                
            }  
            
        self.obs_space = self.create_obs_space(num_obs)

        num_act_policy = 4
        self.num_act_controller = 6  # motor RPM commands

        # collision group indices
        self.collision_group_index_obstacles = -1  # -1 means collide with everything
        self.collision_group_index_agent = -1      # -1 means collide with everything
  
        self.last_actor_cmd = torch.zeros((num_envs, self.num_act_controller), device=device, requires_grad=True)
        self.last_power_draw = torch.zeros(num_envs, device=device, requires_grad=True)
        self.spawn_positions = torch.zeros((num_envs, 3), device=device)
        
        # initialize imu readings
        self.imu_accel_bias_std = kwargs.pop("imu_accel_bias_std", 0.01)
        self.imu_gyro_bias_std = kwargs.pop("imu_gyro_bias_std", 0.02)
        self.imu_accel_bias = torch.randn((num_envs, 3), device=device) * self.imu_accel_bias_std
        self.imu_gyro_bias = torch.randn((num_envs, 3), device=device) * self.imu_gyro_bias_std
        self.imu_accel_noise_std = kwargs.pop("imu_accel_noise_std", 0.03)
        self.imu_gyro_noise_std = kwargs.pop("imu_gyro_noise_std", 0.02)    

        # initialize env offsets
        if not self.no_env_offset:
            x_min, x_max = self.arena_bounds[0, 0].item(), self.arena_bounds[0, 1].item()
            y_min, y_max = self.arena_bounds[1, 0].item(), self.arena_bounds[1, 1].item()
            z_min, z_max = self.arena_bounds[2, 0].item(), self.arena_bounds[2, 1].item()
            self.env_offset = (x_max - x_min, y_max - y_min, z_max - z_min)

        # Now call super with only the parameters it expects
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act_policy,
            episode_length=max_episode_length,
            early_termination=early_termination,
            render=render,
            render_mode=render_mode,
            use_graph_capture=True,
            **kwargs,
        )        
       
    def create_modelbuilder(self):
        """Create the model builder with drone-specific settings"""
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.02
        
        return builder

    def add_static_shared_obstacles(self, builder):
        """Add static obstacles including arena walls and maze-like structures"""
        if not self.has_static_obstacles:
            return
        
        # # Get arena bounds
        # x_min, x_max = self.room_bounds[0, 0].item(), self.room_bounds[0, 1].item()
        # y_min, y_max = self.room_bounds[1, 0].item(), self.room_bounds[1, 1].item()
        # z_min, z_max = self.room_bounds[2, 0].item(), self.room_bounds[2, 1].item()

        # wall_thickness = 0.1 
        # wall_height = y_max - y_min
        # ground_altiude = 0.0

        # # Add 4 perimeter walls
        # # North wall (positive Z)
        # builder.add_shape_box(
        #     -1,
        #     pos=(0.0, ground_altiude + wall_height/2, z_max + wall_thickness/2),
        #     hx=(x_max - x_min)/2,
        #     hy=wall_height/2,
        #     hz=wall_thickness/2,
        #     is_solid=True,
        #     has_shape_collision=True,            
        #     is_visible=True,
        # )
        
        # # South wall (negative Z)
        # builder.add_shape_box(
        #     -1,
        #     pos=(0.0, ground_altiude + wall_height/2, z_min - wall_thickness/2),
        #     hx=(x_max - x_min)/2,
        #     hy=wall_height/2,
        #     hz=wall_thickness/2,
        #     is_solid=True,
        #     has_shape_collision=True,            
        #     is_visible=True,
        # )
        
        # # East wall (positive X)
        # builder.add_shape_box(
        #     -1,
        #     pos=(x_max + wall_thickness/2, ground_altiude + wall_height/2, 0.0),
        #     hx=wall_thickness/2,
        #     hy=wall_height/2,
        #     hz=(z_max - z_min)/2,
        #     is_solid=True,
        #     has_shape_collision=True,            
        #     is_visible=True,
        # )
        
        # # West wall (negative X)
        # builder.add_shape_box(
        #     -1,
        #     pos=(x_min - wall_thickness/2, ground_altiude + wall_height/2, 0.0),
        #     hx=wall_thickness/2,
        #     hy=wall_height/2,
        #     hz=(z_max - z_min)/2,
        #     is_solid=True,
        #     has_shape_collision=True,            
        #     is_visible=True,
        # )
        
        # # Generate structured maze with guaranteed connectivity
        # walls = self.generate_connected_maze(x_min, x_max, z_min, z_max, wall_thickness)
        
        # # Add maze walls to the builder
        # for wall in walls:
        #     builder.add_shape_box(
        #         -1,
        #         pos=(wall['pos'][0], ground_altiude + wall_height/2, wall['pos'][2]),
        #         hx=wall['size'][0]/2,
        #         hy=wall_height/2,
        #         hz=wall['size'][2]/2,
        #         is_solid=True,
        #         has_shape_collision=True,                    
        #         is_visible=True,
        #     )

        # self.num_static_obstacles = 4 + len(walls)  # boundary walls + maze walls
        self.spawn_racing_parcour(builder)

    def spawn_racing_parcour(self, builder):
        """Spawns a racing parcour with random obstacles."""
        x_min, x_max = self.room_bounds[0, 0].item(), self.room_bounds[0, 1].item()
        y_min, y_max = self.room_bounds[1, 0].item(), self.room_bounds[1, 1].item()
        z_min, z_max = self.room_bounds[2, 0].item(), self.room_bounds[2, 1].item()
        
        num_obstacles = self.num_static_obstacles

        for _ in range(num_obstacles):
            # Choose obstacle type
            obstacle_type = random.choice(['box', 'capsule'])
            
            # Random position within the arena
            pos_x = random.uniform(x_min, x_max)
            pos_y = random.uniform(y_min, y_max)
            pos_z = random.uniform(z_min, z_max)
            
            # Random orientation (yaw rotation around Y axis)
            rot_angle = random.uniform(0, 2 * math.pi)
            rot = wp.quat_rpy(0.0, rot_angle, 0.0)

            if obstacle_type == 'box':
                # Random size
                hx = random.uniform(self.obstacle_min_size, self.obstacle_max_size)
                hy = random.uniform(self.obstacle_min_size, self.obstacle_max_size)
                hz = random.uniform(self.obstacle_min_size, self.obstacle_max_size)
                builder.add_shape_box(
                    -1,
                    pos=(pos_x, pos_y, pos_z),
                    rot=rot,
                    hx=hx,
                    hy=hy,
                    hz=hz,
                    is_solid=True,
                    has_shape_collision=True,
                    is_visible=True,
                )
            elif obstacle_type == 'capsule':
                # Random size
                radius = random.uniform(self.obstacle_min_size / 2, self.obstacle_max_size / 2)
                half_height = random.uniform(self.obstacle_min_size, self.obstacle_max_size)
                builder.add_shape_capsule(
                    -1,
                    pos=(pos_x, pos_y, pos_z),
                    rot=rot,
                    radius=radius,
                    half_height=half_height,
                    is_solid=True,
                    has_shape_collision=True,
                    is_visible=True,
                )
        self.num_static_obstacles = num_obstacles
        self.has_static_obstacles = True  # Ensure the flag is set to True

    def generate_connected_maze(self, x_min, x_max, z_min, z_max, wall_thickness):
        """Generate a maze with guaranteed connectivity using one of several algorithms"""        
        
        # Choose maze generation method
        # maze_type = random.choice(['rooms_and_corridors', 'recursive_division', 'voronoi_rooms'])
        maze_type = 'rooms_and_corridors'  # for now, just use rooms and corridors

        if maze_type == 'rooms_and_corridors':
            return self._generate_rooms_and_corridors(x_min, x_max, z_min, z_max, wall_thickness)
        elif maze_type == 'recursive_division':
            return self._generate_recursive_division_maze(x_min, x_max, z_min, z_max, wall_thickness)
        else:  # voronoi_rooms
            return self._generate_voronoi_rooms(x_min, x_max, z_min, z_max, wall_thickness)

    def _generate_rooms_and_corridors(self, x_min, x_max, z_min, z_max, wall_thickness):
        """Generate rooms connected by corridors - guarantees connectivity"""        
        walls = []
        
        # Parameters
        min_room_size = 2.0
        max_room_size = 4.0
        corridor_width = 1.0
        margin = 1.0
        
        # Generate 3-6 rooms
        num_rooms = random.randint(3, 6)
        rooms = []
        
        # Place rooms with non-overlapping positions
        for i in range(num_rooms):
            attempts = 0
            while attempts < 50:  # Prevent infinite loops
                room_w = random.uniform(min_room_size, max_room_size)
                room_h = random.uniform(min_room_size, max_room_size)
                
                room_x = random.uniform(x_min + margin + room_w/2, x_max - margin - room_w/2)
                room_z = random.uniform(z_min + margin + room_h/2, z_max - margin - room_h/2)
                
                new_room = {
                    'center': (room_x, room_z),
                    'size': (room_w, room_h),
                    'bounds': (room_x - room_w/2, room_x + room_w/2, room_z - room_h/2, room_z + room_h/2)
                }
                
                # Check if room overlaps with existing rooms
                overlap = False
                for existing_room in rooms:
                    if self._rooms_overlap(new_room, existing_room, corridor_width):
                        overlap = True
                        break
                
                if not overlap:
                    rooms.append(new_room)
                    break
                attempts += 1
        
        # Connect rooms with L-shaped corridors
        for i in range(len(rooms) - 1):
            room_a = rooms[i]
            room_b = rooms[i + 1]
            
            # Create L-shaped corridor from room_a to room_b
            cx_a, cz_a = room_a['center']
            cx_b, cz_b = room_b['center']
            
            # Horizontal segment from room_a to intermediate point
            intermediate_x = cx_b
            corridor_h1 = {
                'center': ((cx_a + intermediate_x)/2, cz_a),
                'size': (abs(intermediate_x - cx_a), corridor_width),
            }
            
            # Vertical segment from intermediate point to room_b
            corridor_v = {
                'center': (intermediate_x, (cz_a + cz_b)/2),
                'size': (corridor_width, abs(cz_b - cz_a)),
            }
            
            # Add corridor walls (walls around the corridor, not blocking it)
            corridors = [corridor_h1, corridor_v]
            
            for corridor in corridors:
                cx, cz = corridor['center']
                cw, ch = corridor['size']
                
                # Add walls on both sides of corridor
                if cw > ch:  # Horizontal corridor
                    # Top and bottom walls
                    walls.append({
                        'pos': (cx, 0, cz + ch/2 + wall_thickness/2),
                        'size': (cw + wall_thickness, 0, wall_thickness)
                    })
                    walls.append({
                        'pos': (cx, 0, cz - ch/2 - wall_thickness/2),
                        'size': (cw + wall_thickness, 0, wall_thickness)
                    })
                else:  # Vertical corridor
                    # Left and right walls
                    walls.append({
                        'pos': (cx + cw/2 + wall_thickness/2, 0, cz),
                        'size': (wall_thickness, 0, ch + wall_thickness)
                    })
                    walls.append({
                        'pos': (cx - cw/2 - wall_thickness/2, 0, cz),
                        'size': (wall_thickness, 0, ch + wall_thickness)
                    })
        
        # Add room boundary walls (with openings for corridors)
        for room in rooms:
            cx, cz = room['center']
            rw, rh = room['size']
            
            # Add walls around each room (simplified - you may want to add corridor openings)
            room_walls = [
                {'pos': (cx, 0, cz + rh/2 + wall_thickness/2), 'size': (rw, 0, wall_thickness)},  # Top
                {'pos': (cx, 0, cz - rh/2 - wall_thickness/2), 'size': (rw, 0, wall_thickness)},  # Bottom  
                {'pos': (cx + rw/2 + wall_thickness/2, 0, cz), 'size': (wall_thickness, 0, rh)},  # Right
                {'pos': (cx - rw/2 - wall_thickness/2, 0, cz), 'size': (wall_thickness, 0, rh)},  # Left
            ]
            
            walls.extend(room_walls)
        
        return walls

    def _generate_recursive_division_maze(self, x_min, x_max, z_min, z_max, wall_thickness):
        """Generate maze using recursive division algorithm"""
        walls = []
        
        def divide_space(x1, x2, z1, z2, min_size=6.0, max_depth=3, current_depth=0):
            width = x2 - x1
            height = z2 - z1
            
            # Stop recursion if space is too small or we've reached max depth
            if width < min_size or height < min_size or current_depth >= max_depth:
                return
                
            # Choose division direction (prefer longer dimension)
            divide_vertically = width > height
            
            if divide_vertically:
                # Divide with vertical wall that extends FULLY to boundaries
                div_x = random.uniform(x1 + min_size/3, x2 - min_size/3)  # More conservative positioning
                
                # Add wall with gap (wider hallways for drone navigation)
                gap_z = random.uniform(z1 + 3.0, z2 - 3.0)  # More buffer from boundaries
                gap_size = 4.0
                
                # Wall segments - extend EXACTLY to the space boundaries (no margin)
                if gap_z - gap_size/2 > z1:
                    walls.append({
                        'pos': (div_x, 0, (z1 + gap_z - gap_size/2)/2),
                        'size': (wall_thickness, 0, gap_z - gap_size/2 - z1)
                    })
                
                if gap_z + gap_size/2 < z2:
                    walls.append({
                        'pos': (div_x, 0, (gap_z + gap_size/2 + z2)/2),
                        'size': (wall_thickness, 0, z2 - (gap_z + gap_size/2))
                    })
                
                # Recursively divide both sides
                divide_space(x1, div_x, z1, z2, min_size, max_depth, current_depth + 1)
                divide_space(div_x, x2, z1, z2, min_size, max_depth, current_depth + 1)
                
            else:
                # Divide with horizontal wall that extends FULLY to boundaries
                div_z = random.uniform(z1 + min_size/3, z2 - min_size/3)  # More conservative positioning
                
                # Add wall with gap (wider hallways for drone navigation)
                gap_x = random.uniform(x1 + 3.0, x2 - 3.0)  # More buffer from boundaries
                gap_size = 4.0
                
                # Wall segments - extend EXACTLY to the space boundaries (no margin)
                if gap_x - gap_size/2 > x1:
                    walls.append({
                        'pos': ((x1 + gap_x - gap_size/2)/2, 0, div_z),
                        'size': (gap_x - gap_size/2 - x1, 0, wall_thickness)
                    })
                
                if gap_x + gap_size/2 < x2:
                    walls.append({
                        'pos': ((gap_x + gap_size/2 + x2)/2, 0, div_z),
                        'size': (x2 - (gap_x + gap_size/2), 0, wall_thickness)
                    })
                
                # Recursively divide both sides
                divide_space(x1, x2, z1, div_z, min_size, max_depth, current_depth + 1)
                divide_space(x1, x2, div_z, z2, min_size, max_depth, current_depth + 1)
        
        # Start recursive division with NO MARGIN - use exact boundary coordinates
        divide_space(x_min, x_max, z_min, z_max)
        
        return walls

    def _generate_voronoi_rooms(self, x_min, x_max, z_min, z_max, wall_thickness):
        """Generate rooms using Voronoi-like cell division"""        
        walls = []
        
        # Generate seed points for rooms
        num_seeds = random.randint(4, 8)
        seeds = []
        margin = 2.0
        
        for _ in range(num_seeds):
            seed_x = random.uniform(x_min + margin, x_max - margin)
            seed_z = random.uniform(z_min + margin, z_max - margin)
            seeds.append((seed_x, seed_z))
        
        # Create walls between regions (simplified Voronoi)
        grid_res = 0.5
        x_coords = torch.arange(x_min, x_max, grid_res)
        z_coords = torch.arange(z_min, z_max, grid_res)
        
        boundary_points = []
        
        for x in x_coords:
            for z in z_coords:
                # Find two closest seeds
                distances = [(math.sqrt((x-sx)**2 + (z-sz)**2), i) for i, (sx, sz) in enumerate(seeds)]
                distances.sort()
                
                # If close to boundary between regions, mark as wall candidate
                if len(distances) >= 2 and distances[1][0] - distances[0][0] < 0.3:
                    boundary_points.append((x.item(), z.item()))
        
        # Group boundary points into wall segments
        # (This is a simplified approach - you might want more sophisticated wall building)
        if boundary_points:
            # Sample some boundary points to create walls
            num_walls = min(len(boundary_points) // 3, self.num_static_obstacles)
            selected_points = random.sample(boundary_points, min(num_walls * 2, len(boundary_points)))
            
            for i in range(0, len(selected_points) - 1, 2):
                p1 = selected_points[i]
                p2 = selected_points[i + 1]
                
                # Create wall between these points
                center_x = (p1[0] + p2[0]) / 2
                center_z = (p1[1] + p2[1]) / 2
                
                length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                length = max(0.5, min(length, 3.0))  # Clamp wall length
                
                # Random orientation
                if random.random() > 0.5:
                    walls.append({
                        'pos': (center_x, 0, center_z),
                        'size': (wall_thickness, 0, length)
                    })
                else:
                    walls.append({
                        'pos': (center_x, 0, center_z),
                        'size': (length, 0, wall_thickness)
                    })
        
        return walls

    def _rooms_overlap(self, room1, room2, buffer=0.5):
        """Check if two rooms overlap with buffer zone"""
        x1_min, x1_max, z1_min, z1_max = room1['bounds']
        x2_min, x2_max, z2_min, z2_max = room2['bounds']
        
        # Add buffer
        x1_min -= buffer
        x1_max += buffer
        z1_min -= buffer
        z1_max += buffer
        
        return not (x1_max < x2_min or x2_max < x1_min or z1_max < z2_min or z2_max < z1_min)


    def create_model(self):
        """Create model and ensure control() allocates a differentiable user_act tensor."""
        model = super().create_model()
            
        num_envs = self.num_envs
        requires_grad = self.requires_grad
        device = model.device

        # Wrap the existing control constructor to attach user_act
        original_control_fn = Model_control.__get__(model, model.__class__)

        def control_with_user_act(self_model, requires_grad_arg=None, clone_variables=True, copy="clone"):
            c = original_control_fn(requires_grad=requires_grad_arg, clone_variables=clone_variables, copy=copy)
            if not hasattr(c, "body_force"):
                c.body_force = wp.zeros(  # type: ignore[attr-defined]
                    (num_envs, self.num_act_controller), dtype=float, device=device, requires_grad=requires_grad
                )
            return c

        model.control = control_with_user_act.__get__(model, model.__class__)
        return model
    
    def create_scene_interactive_elements(self, builder):        
        warp.sim.import_usd.parse_usd(
            "assets/manipulation_scene.usd", 
            builder, 
            verbose=True,
            invert_rotations=True, # why the fuck do we need this? It appears like the tfs are reversed (either from omniverse usd composer or applied in reverse in the warp mesh generator)         
            contact_ke=1e4,
            contact_kd=1e3,
            contact_kf=1e2,
            contact_ka=0.0,
            contact_mu=0.6,
            contact_restitution=1e-2,
            contact_thickness=0.0,
            joint_limit_ke=100.0,
            joint_limit_kd=10.0,
        )

    def create_articulation(self, builder):
        # """Create the drone as a rigid body with automatic mass calculation."""                

        # Create the drone as a single rigid body
        body = builder.add_body(name="drone")
        
        # Calculate virtual density to achieve target body mass
        body_volume = self.body_dimensions[0] * self.body_dimensions[1] * self.body_dimensions[2]
        virtual_density = self.body_mass / body_volume

        # arm length
        arm_length = self.arm_specs[2] / 2
        print(f"arm_length: {arm_length}")

        # collision box
        builder.add_shape_box(
            body,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            hx=arm_length*1.3,
            hy=self.body_dimensions[2]/2,
            hz=arm_length*1.3,
            density=0.0,
            is_visible=False,
            has_shape_collision=True,
            has_ground_collision=True,
        )
        
        # main body (battery, fc, ...)
        builder.add_shape_box(
            body,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            hx=self.body_dimensions[0]/2,  # length/2
            hy=self.body_dimensions[2]/2,  # height/2
            hz=self.body_dimensions[1]/2,  # width/2
            density=virtual_density,
            has_shape_collision=False,
            has_ground_collision=False,
        )

        # marker to indicate forward direction
        builder.add_shape_cylinder(
            body,           
            pos=(0.25, 0.1, 0.0),
            up_axis=0,
            radius=0.01, 
            half_height=0.25, # half the arm-lenght is quarter of span
            density=0.0,
            is_solid=True,            
            has_shape_collision=False,
            has_ground_collision=False,
        )
       
        # build props and arms
        props = []        
        
        thetas = [-30, 30, 90, 150, 210, 270]
        thetas = [theta * math.pi/180 for theta in thetas]
        polarity = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

        prop_positions_b = []

        # Initialize as numpy array first for easier manipulation
        drone_force_torque_mapping = np.zeros([6, 6], dtype=np.float32)
 
        for i in range(6):
            theta = thetas[i]
            turning_dir = polarity[i]
            alpha = -polarity[i] * math.radians(self.tilt_angle_deg)

            quat_BtildeMi = wp.quat_rpy(alpha, 0.0, theta)  

            p_armCenter_B = -wp.quat_rotate(self.quat_BBtilde * quat_BtildeMi, wp.vec3(-arm_length, 0.0, 0.0)) / 2.0 # center of arm in body frame          
            o_Mi_B = -wp.quat_rotate(self.quat_BBtilde * quat_BtildeMi, wp.vec3(-arm_length, 0.0, self.arm_specs[0]))  # origin of motor frame in body frame            
            p_prop_B = -wp.quat_rotate(self.quat_BBtilde * quat_BtildeMi, wp.vec3(-arm_length, 0.0, self.arm_specs[0] + self.motor_specs[1] / 2 + self.prop_thickness / 2)) # center of prop in body frame
            
            prop_radius = self.prop_diameter_inch / 2 * 0.0254

            # Air density at sea level
            rho = 1.225  # kg/m³
            
            # For typical quadcopter propellers, use reasonable CT and CP values
            # CT ≈ 0.1-0.15, CP ≈ 0.05-0.08 for efficient props
            # Pitch affects thrust efficiency: higher pitch = higher thrust per RPM
            C_T = 0.15  # thrust coefficient
            C_P = 0.05  # power coefficient

            C_Q = C_P / (2.0 * 3.14159)

            k_f = C_T * rho * ((prop_radius*2) ** 4) / (4.0 * 3.14159**2)
            k_d = C_Q * rho * ((prop_radius*2) ** 5) / (4.0 * 3.14159**2)

            r_3_i = wp.quat_rotate(self.quat_BBtilde * quat_BtildeMi, wp.vec3(0.0, 0.0, -1.0))
          
            # Convert warp vectors to numpy and assign
            drone_force_torque_mapping[:3, i] = np.array(r_3_i * k_f)
            drone_force_torque_mapping[3:6, i] = np.array(r_3_i * (k_d * polarity[i]) + wp.cross(o_Mi_B, r_3_i * k_f))                

            # add arm            
            builder.add_shape_cylinder(
                body,
                pos=p_armCenter_B,
                rot=self.quat_BBtilde * quat_BtildeMi,
                up_axis=0,
                radius=self.arm_specs[0]/2, 
                half_height=self.arm_specs[2]/4, # half the arm-lenght is quarter of span
                density=self.density_carbon,
                is_solid=False,
                thickness=self.arm_specs[1],            
                has_shape_collision=False,
                has_ground_collision=False,
            )

            motor_volume = math.pi * (self.motor_specs[0]/2)**2 * self.motor_specs[1]
            motor_equivalent_density = self.motor_specs[2] / motor_volume
            
            # Motors
            builder.add_shape_cylinder(
                body,
                pos=o_Mi_B,
                rot=self.quat_BBtilde * quat_BtildeMi,
                up_axis=2,
                radius=self.motor_specs[0]/2,            
                half_height=self.motor_specs[1]/2, 
                density=motor_equivalent_density,
                is_solid=True,                      
                has_shape_collision=False,
                has_ground_collision=False,
            )
            
            prop_disk_volume = math.pi * prop_radius**2 * self.prop_thickness
            prop_disk_equivalent_density = self.prop_weight / prop_disk_volume

            # Props
            builder.add_shape_cylinder(
                body,
                pos=p_prop_B,
                rot=self.quat_BBtilde * quat_BtildeMi,
                up_axis=2,
                radius=prop_radius,
                half_height=self.prop_thickness / 2,
                density=prop_disk_equivalent_density,
                is_solid=True,                      
                has_shape_collision=False,
                has_ground_collision=False,
            )
                        
            prop_positions_b.append((p_prop_B[0], p_prop_B[1], p_prop_B[2]))  
        
        # Convert to warp array after all assignments are done
        # drone_force_torque_mapping[drone_force_torque_mapping < 1e-11] = 0.0 
        self._drone_force_torque_mapping = wp.array(drone_force_torque_mapping, dtype=wp.float32)

        # Store prop data for force calculations
        self.props = props
        self.turning_directions = polarity
        self.prop_positions = prop_positions_b

    def init_sim(self):
        super().init_sim()

        # Create simulation view for clean body access
        self.sim_view = SimulationView(
            model=self.model,
            num_envs=self.num_envs            
        )

        with torch.no_grad():
            self.start_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

            # Initialize target positions for each environment
            self.target_body_q = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
            self.target_body_qd = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
            self.reset_targets(torch.arange(self.num_envs, device=self.device))

        # Provide an external-forces callback via the model so newly-created
        # Control objects inherit it each step.
        self.model.apply_external_forces = self.apply_drone_forces
        self.model.compute_collision_distances = self.compute_collision_distances
        self.model.compute_depth_observations = self.compute_depth_observations
        
        # Display drone specifications
        self.print_drone_specs()
        
        # Setup drone parameters and cache computed values
        self.setup_drone()

        # ================= Depth camera setup =================
        # Precompute local ray directions (unit)
        fovx = math.radians(self.cam_fov_x_deg)
        fovy = math.radians(self.cam_fov_y_deg)
        tan_x = math.tan(fovx * 0.5)
        tan_y = math.tan(fovy * 0.5)
        W = int(self.cam_width)
        H = int(self.cam_height)
        dirs = np.zeros((W * H, 3), dtype=np.float32)
        idx = 0
        for j in range(H):
            v = (float(j) + 0.5) / float(H) * 2.0 - 1.0
            for i in range(W):
                u = (float(i) + 0.5) / float(W) * 2.0 - 1.0
                dx = 1.0
                dy = -v * tan_y
                dz = u * tan_x
                inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
                dirs[idx, 0] = dx * inv
                dirs[idx, 1] = dy * inv
                dirs[idx, 2] = dz * inv
                idx += 1

        self._cam_dirs_local_wp = wp.array(dirs, dtype=wp.vec3, device=self.model.device)  # type: ignore[attr-defined]
        self._cam_pos_local_wp = wp.vec3(float(self.cam_offset_local[0]), float(self.cam_offset_local[1]), float(self.cam_offset_local[2]))  # type: ignore[attr-defined]
        self._cam_depths_wp = wp.full(
            (self.num_envs, W * H),
            float(self.cam_max_depth),
            dtype=float,
            device=self.model.device,
            requires_grad=self.requires_grad,
        )

        # container for collision distances        
        self.indices_shape_with_collision = wp.array([i for i, x in enumerate(self.model.shape_shape_collision) if x], dtype=int, device=self.model.device)        
        self._collision_distances = wp.zeros([self.num_envs, len(self.indices_shape_with_collision)], dtype=wp.float32, device=self.model.device, requires_grad=True)
        
        # random spawn points
        self.randomize_init(torch.arange(self.num_envs, device=self.device))
            
    def setup_drone(self):
        """Setup drone parameters and cache computed values. Can be called to reset/randomize parameters."""
        # Calculate and cache max RPM from motor specifications
        self.max_rpm = self.motor_kv * self.nominal_cell_voltage * self.lipo_cells
        self.max_omega = self.max_rpm * 2.0 * 3.14159 / 60.0  # rad/s        

    def print_drone_specs(self):
        """Calculate and display drone specifications including thrust-to-weight ratio."""
        # Calculate max RPM from battery and motor specs
        total_voltage = self.lipo_cells * self.nominal_cell_voltage
        max_rpm = self.motor_kv * total_voltage
        
        # Calculate max thrust per motor at max RPM
        # Thrust = k_f * omega² where omega = 2π * RPM / 60
        max_omega = 2 * 3.14159 * max_rpm / 60.0  # rad/s

        res = wp.to_torch(self._drone_force_torque_mapping).cpu() @ torch.ones((6,1))* max_omega**2 
        total_max_thrust = res[1].item()   
        
        # Calculate thrust-to-weight ratio
        total_mass = wp.to_torch(self.model.body_mass)[0].item()
        total_weight = total_mass * abs(self.gravity)
        thrust_to_weight_ratio = total_max_thrust / total_weight
        
        # Display specifications
        print("\n" + "="*60)
        print("🚁 DRONE SPECIFICATIONS")
        print("="*60)
        print(f"📏 Dimensions:")
        print(f"   Arm span: {self.arm_specs[2]*100:.1f} cm")
        print(f"   Body: {self.body_dimensions[0]*100:.1f} × {self.body_dimensions[1]*100:.1f} × {self.body_dimensions[2]*100:.1f} cm")
        print(f"   Propeller: {self.prop_diameter_inch}×{self.prop_pitch_inch} inch")
        
        print(f"\n🔋 Battery & Motor:")
        print(f"   Battery: {self.lipo_cells}S ({total_voltage:.1f}V)")
        print(f"   Motor KV: {self.motor_kv}")
        print(f"   Max RPM: {max_rpm:.0f}")
        
        print(f"\n⚖️ Mass & Inertia:")        
        print(f"   Total mass: {total_mass:.3f} kg")
        print(f"   Body inertia (kg·m²):")

        print(f"\n⚡ Thrust & Performance:")                
        print(f"   Total max straight hover thrust: {total_max_thrust:.2f} N")        
        print(f"   Total weight: {total_weight:.2f} N ({total_mass:.3f} kg)")
        print(f"   Thrust-to-weight ratio: {thrust_to_weight_ratio:.2f}")
        
        # Overall performance based on thrust-to-weight ratio
        if thrust_to_weight_ratio > 2.0:
            print(f"   🚀 Performance: Excellent (>2.0)")
        elif thrust_to_weight_ratio > 1.5:
            print(f"   ✈️ Performance: Good (1.5-2.0)")
        elif thrust_to_weight_ratio > 1.0:
            print(f"   🛸 Performance: Adequate (1.0-1.5)")
        else:
            print(f"   ⚠️ Performance: Poor (<1.0) - may not hover!")

    def apply_drone_forces(self, model, state, control=None):        
        wp.launch(
            kernel=apply_drone_forces_kernel,
            dim=self.num_envs,
            inputs=(
                state.body_q,
                model.body_com,
                state.body_qd,
                state.body_f,
                self.sim_view._drone_indices_wp,                                
                self._drone_force_torque_mapping,
                control.body_force,
                float(self.max_rpm),
            ),
            device=self.model.device,
        )        

    def render(self, state=None):
        """Override render to show dynamic target spheres"""        
        if self.render_time % (1.0 / self.render_fps) < self.frame_dt:
            if self.renderer is not None:
                self.renderer.begin_frame(self.render_time)
                
                # Add custom target visualization if targets exist
                if hasattr(self, 'target_body_q') and self.is_render_targets:
                    # Draw target coordinate frames for each environment
                    for i in range(self.num_envs):
                        target_pos = self.target_body_q[i, :3].cpu().numpy()
                        target_quat = self.target_body_q[i, 3:7].cpu().numpy()  # [x, y, z, w]
                                
                        # Arrow dimensions
                        base_radius = 0.05
                        base_height = 0.3
                        cap_radius = 0.08
                        cap_height = 0.1
                        
                        try:
                            # stateX-axis arrow (red)
                            self.renderer.render_arrow(
                                name=f"target_x_{i}",
                                pos=tuple(target_pos),
                                rot=target_quat,
                                base_radius=base_radius,
                                base_height=base_height,
                                cap_radius=cap_radius,
                                cap_height=cap_height,
                                up_axis=0,  # X-axis
                                color=(1.0, 0.0, 0.0),  # Red
                                visible=True
                            )
                            
                            # Y-axis arrow (green) - points downward in warp coordinate system
                            self.renderer.render_arrow(
                                name=f"target_y_{i}",
                                pos=tuple(target_pos),
                                rot=target_quat,
                                base_radius=base_radius,
                                base_height=base_height,
                                cap_radius=cap_radius,
                                cap_height=cap_height,
                                up_axis=1,  # Y-axis (downward)
                                color=(0.0, 1.0, 0.0),  # Green
                                visible=True
                            )
                            
                            # Z-axis arrow (blue)
                            self.renderer.render_arrow(
                                name=f"target_z_{i}",
                                pos=tuple(target_pos),
                                rot=target_quat,
                                base_radius=base_radius,
                                base_height=base_height,
                                cap_radius=cap_radius,
                                cap_height=cap_height,
                                up_axis=2,  # Z-axis
                                color=(0.0, 0.0, 1.0),  # Blue
                                visible=True
                            )
                        except Exception as e:
                            # If rendering fails, skip this target
                            pass
                
                # Render the simulation state
                self.renderer.render(state or self.state_0)
                self.renderer.end_frame()

            if self.debug_pov:
                # Visualize depth observations with tiled plot
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Initialize figure and axes once
                if not hasattr(self, '_debug_fig'):
                    num_to_show = min(16, self.num_envs)
                    n_row_col = math.ceil(math.sqrt(num_to_show))
                    self._debug_fig, axes = plt.subplots(n_row_col, n_row_col, figsize=(6, 6))
                    if num_to_show == 1:
                        self._debug_axes = [axes]  # Convert single Axes to list
                    else:
                        self._debug_axes = axes.flatten()  # Multiple axes - flatten as before
                        
                    self._debug_images = []
                    
                    # Initialize image objects                    
                    for i in range(num_to_show):
                        im = self._debug_axes[i].imshow(np.zeros((self.cam_height, self.cam_width)), 
                                                       cmap='viridis', vmin=self.cam_min_depth, vmax=self.cam_max_depth)
                        self._debug_axes[i].set_title(f'Env {i}')
                        self._debug_axes[i].axis('off')
                        self._debug_images.append(im)
                    
                    # Hide unused subplots
                    for i in range(num_to_show):
                        self._debug_axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.show(block=False)
                
                # Update existing figure with new data
                visual_obs = wp.to_torch(self._cam_depths_wp)
                depth_images = visual_obs.reshape(self.num_envs, self.cam_height, self.cam_width).detach().cpu().numpy()
                
                num_to_show = min(16, self.num_envs)
                for i in range(num_to_show):
                    self._debug_images[i].set_data(depth_images[i])
                
                self._debug_fig.canvas.draw()
                self._debug_fig.canvas.flush_events()

        self.render_time += self.frame_dt

    def reset_targets(self, env_ids):
        """Reset target positions to random locations within target bounds"""
        with torch.no_grad():            
            # Generate random positions within scaled target bounds
            random_factors = torch.rand([len(env_ids), 3], device=self.device)
            mins = self.target_bounds[:, 0]  # [x_min, y_min, z_min]
            maxs = self.target_bounds[:, 1]  # [x_max, y_max, z_max]
            ranges = maxs - mins        # [x_range, y_range, z_range]

            target_pos_rand = mins + random_factors * ranges

            indices_shape_with_collision = wp.array([i for i, x in enumerate(self.model.shape_shape_collision) if x], dtype=int, device=self.model.device)        
            _collision_distances = wp.zeros([len(env_ids), len(indices_shape_with_collision)], dtype=wp.float32, device=self.model.device, requires_grad=True)
            wp.launch(
                collision_distance_kernel,
                dim=(len(env_ids), len(indices_shape_with_collision)),
                inputs=[
                    wp.array(torch.cat((target_pos_rand, torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(len(env_ids), -1)), dim=-1), dtype=wp.transform, device=self.model.device),  # target transforms
                    indices_shape_with_collision,
                    self.sim_view._drone_indices_wp,
                    self.model.shape_transform,
                    self.model.shape_geo,  
                    self.model.shape_body,                                            
                ],
                outputs=[_collision_distances],
                device=self.model.device
            )               
            min_distances, _ = torch.min(wp.to_torch(_collision_distances), dim=1)

            mask_need_respawn = min_distances < self.arm_specs[2] / 2.0 + 0.2

            num_trials = 0
            while torch.any(mask_need_respawn) and num_trials < 10:
                # update spawn positions of problematic poses
                num_to_respawn = torch.sum(mask_need_respawn).item()
                # print(f"Respawning {num_to_respawn} targets...")     
                random_factors = torch.rand([num_to_respawn, 3], device=self.device)
                target_pos_rand[torch.where(mask_need_respawn)[0], :] = mins + random_factors * ranges
                
                candidate_transforms = wp.array(torch.cat((target_pos_rand, torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(len(env_ids), -1)), dim=-1), dtype=wp.transform, device=self.model.device)
                
                wp.launch(
                    collision_distance_kernel,
                    dim=(len(env_ids), len(indices_shape_with_collision)),
                    inputs=[
                        candidate_transforms,  # target transforms
                        indices_shape_with_collision,
                        self.sim_view._drone_indices_wp,
                        self.model.shape_transform,
                        self.model.shape_geo,  
                        self.model.shape_body,                                            
                    ],
                    outputs=[_collision_distances],
                    device=self.model.device
                )               
                min_distances_respawn, _ = torch.min(wp.to_torch(_collision_distances), dim=1)           

                # Update mask for next iteration                
                mask_need_respawn = min_distances_respawn < self.arm_specs[2]  

                num_trials += 1
            # generate random target attitudes            
            randomized_target_orientation = self.target_attitude_euler_deg.repeat(len(env_ids), 1) / 180.0 * math.pi
            randomized_target_orientation += torch.randn_like(randomized_target_orientation) * self.three_sigma_target_attitude_deg / 3.0 / 180.0 * math.pi

            # Create quaternions for each rotation axis
            quat_roll = quat_from_angle_axis(randomized_target_orientation[:, 0], torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(env_ids), -1))
            quat_pitch = quat_from_angle_axis(randomized_target_orientation[:, 1] , torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(len(env_ids), -1))
            quat_yaw = quat_from_angle_axis(randomized_target_orientation[:, 2], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(env_ids), -1))
            
            # Combine rotations: yaw * pitch * roll (Z-Y-X convention)
            target_attitudes = quat_mul(quat_yaw, quat_mul(quat_pitch, quat_roll))
            
            self.target_body_q[env_ids, 0:3] = target_pos_rand
            self.target_body_q[env_ids, 3:7] = target_attitudes

            # Set target velocities
            self.target_body_qd[env_ids, 3:6] = self.target_lin_vel.repeat(len(env_ids), 1) + torch.randn_like(self.target_lin_vel) * self.three_sigma_target_lin_vel / 3.0
            self.target_body_qd[env_ids, :3] = self.target_ang_vel.repeat(len(env_ids), 1) + torch.randn_like(self.target_ang_vel) * self.three_sigma_target_ang_vel / 3.0

    # def reset_static_obstacles(self):
    #     with torch.no_grad():
    #         # Find obstacle shape indices (shapes that are not drone bodies)
    #         all_shapes = self.indices_shape_with_collision.numpy()
    #         drone_shapes = self.sim_view._drone_indices_wp.numpy()
    #         obstacle_mask = ~np.isin(all_shapes, drone_shapes)
    #         obstacle_indices = all_shapes[obstacle_mask]
    #         N_obs = len(obstacle_indices)
            
    #         # Generate random positions within scaled room bounds
    #         random_factors = torch.rand(N_obs, 3, device=self.device)
    #         mins = self.room_bounds[:, 0]  # [x_min, y_min, z_min]
    #         maxs = self.room_bounds[:, 1]  # [x_max, y_max, z_max]
    #         ranges = maxs - mins        # [x_range, y_range, z_range]

    #         # generate random target attitudes            
    #         randomized_orientation = torch.randn(N_obs, device=self.device) * math.pi

    #         # Create quaternions for each rotation axis
    #         quat_azimuth = quat_from_angle_axis(randomized_orientation, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(N_obs, -1))
            
    #         cache_poses = wp.to_torch(self.model.shape_transform)

    #         for i, s in enumerate(obstacle_indices):
    #             cache_poses[s, :3] = (mins + ranges * random_factors[i])*0.0
    #             cache_poses[s, 3:7] = quat_azimuth[i]
                        
    #         self.model.shape_transform = wp.from_torch(cache_poses, dtype=wp.transform)

    @torch.no_grad()
    def randomize_init(self, env_ids):        
        """Randomize drone spawn positions using SimulationView"""
        N = len(env_ids)
        
        # Get drone states using the simulation view
        body_q, body_qd = self.sim_view.get_drone_states(self.state)

        ##  ================== sample initial pose ========================
        # generate random initial drone attitudes            
        randomized_initial_orientation = self.initial_drone_attitude_euler_deg.repeat(N, 1) / 180.0 * math.pi
        randomized_initial_orientation += torch.randn_like(randomized_initial_orientation) * self.three_sigma_initial_attitude_deg / 3.0 / 180.0 * math.pi

        # Create quaternions for each rotation axis
        quat_roll = quat_from_angle_axis(randomized_initial_orientation[:, 0], torch.tensor([1.0, 0.0, 0.0], device=self.device))
        quat_pitch = quat_from_angle_axis(randomized_initial_orientation[:, 1] , torch.tensor([0.0, 1.0, 0.0], device=self.device))
        quat_yaw = quat_from_angle_axis(randomized_initial_orientation[:, 2], torch.tensor([0.0, 0.0, 1.0], device=self.device))

        # Combine rotations: yaw * pitch * roll (Z-Y-X convention)
        initial_quat = quat_mul(quat_yaw, quat_mul(quat_pitch, quat_roll))
        
        # Set initial quaternion for all selected environments
        body_q[env_ids, 3:7] = initial_quat
        
        # Add random position offset
        # Generate random spawn positions within spawn bounds
        spawn_mins = self.spawn_bounds[:, 0]  # [x_min, y_min, z_min]
        spawn_maxs = self.spawn_bounds[:, 1]  # [x_max, y_max, z_max]
        spawn_ranges = spawn_maxs - spawn_mins
        body_q[env_ids, 0:3] = spawn_mins + torch.rand(size=(N, 3), device=self.device) * spawn_ranges

        self.spawn_positions[env_ids,:] = body_q[env_ids, 0:3].clone().detach()

        ##  ================== sample initial velocity ========================
        # linear random velocity
        sigma_initial_lin_vel_tensor = torch.tensor(self.three_sigma_initial_lin_vel, device=self.device) / 3.0
        body_qd[env_ids, 3:6] = torch.randn_like(body_qd[env_ids, 3:6]) * sigma_initial_lin_vel_tensor

        # angular random velocity
        sigma_initial_ang_vel_tensor = torch.tensor(self.three_sigma_initial_ang_vel, device=self.device) / 3.0
        body_qd[env_ids, :3] = torch.randn_like(body_qd[env_ids, :3]) * sigma_initial_ang_vel_tensor
        
        self.reset_targets(env_ids)


    def pre_physics_step(self, actions):        
        # Clamp actions to [-1, 1] range
        actions = torch.clamp(actions*0.0, -1.0, 1.0)                        
        self.last_actor_cmd = actions.clone()

        body_q, body_qd = self.sim_view.get_drone_states(self.state)
        omega_body_W = quat_rotate(body_q[:, 3:7], body_qd[:, 0:3])
                                 
        body_qd_desired_B = torch.zeros((self.num_envs, 6), device=self.device, dtype=body_qd.dtype)       
        # body_qd_desired_B[:, 0] = 0.1  
        # body_qd_desired_B[:, :3] = actions[:, :3] * self.max_target_ang_vel  # ang vel cmd
        body_qd_desired_B[:, 3:6] = actions[:, :3] * self.max_target_lin_vel  # lin vel cmd
        body_qd_desired_B[:, 1] = actions[:, 3] * self.max_target_ang_vel  # ang vel cmd        

        # ============== ATTITUDE CONTROL LAW ==============        
        # Get current orientation
        quat_current = body_q[:, 3:7]  # [x, y, z, w]
        body_y = quat_rotate(quat_current, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1))
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)

        attitude_error_W = torch.cross(body_y, world_up, dim=1)
        attitude_error_B = quat_rotate(quat_conjugate(quat_current), attitude_error_W)
       
        body_qd_desired_B[:, (0,2)] = 1.0 * attitude_error_B[:, (0,2)]
        
        # compute control input in world frame
        body_qd_desired_W = torch.zeros((self.num_envs, 6), device=self.device, dtype=body_qd.dtype)         
        body_qd_desired_W[:, 0:3] = quat_rotate(body_q[:, 3:7], body_qd_desired_B[:, 0:3]) 
        body_qd_desired_W[:, 3:6] = quat_rotate(body_q[:, 3:7], body_qd_desired_B[:, 3:6])

        # control input in world frame
        body_qdd_cmd_W = torch.zeros((self.num_envs, 6), device=self.device, dtype=body_qd.dtype) 

        gain_p_ang = 3.0     
        gain_p_lin = 10.0           
        
        body_qdd_cmd_W[:, 0:3] = gain_p_ang * (body_qd_desired_W[:, 0:3] - body_qd[:, 0:3]) * self.frame_dt # control input to control angular velocity
        body_qdd_cmd_W[:, 3:6] = gain_p_lin * (body_qd_desired_W[:, 3:6] - body_qd[:, 3:6]) * self.frame_dt # control input to control linear velocity

        # Get mass and inertia tensors (convert to torch first, then index)
        all_masses = wp.to_torch(self.model.body_mass)  # Convert full array to torch
        all_inertias = wp.to_torch(self.model.body_inertia)  # Convert full array to torch
        
        # Now index with torch tensors
        drone_indices_torch = wp.to_torch(self.sim_view._drone_indices_wp)
        m = all_masses[drone_indices_torch]  # shape: (num_envs,)
        I_b = all_inertias[drone_indices_torch]  # shape: (num_envs, 3, 3)
       
        # mass matrix
        num_envs = m.shape[0]
        M = torch.zeros((num_envs, 6, 6), device=self.device, dtype=m.dtype)
        M[:, 3, 3] = m  # m for x translation
        M[:, 4, 4] = m  # m for y translation  
        M[:, 5, 5] = m  # m for z translation
        M[:, :3, :3] = I_b
      
        # intrinsic forces in world frame
        f_W = torch.zeros((self.num_envs, 6), device=self.device, dtype=body_qd.dtype)
        f_W[:, 0:3] = torch.cross(omega_body_W, (I_b @ omega_body_W[:, 0:, None]).squeeze(-1), dim=1) # rotational
        f_W[:, 3:6] = m[:,None] * torch.tensor([0.0, self.gravity, 0.0], device=self.device)[None,:] # translational        

        # input force and wrench mapping
        F_B = wp.to_torch(self._drone_force_torque_mapping[0:3, :])[None,:,:].expand(self.num_envs, -1, -1)
        Tau_B = wp.to_torch(self._drone_force_torque_mapping[3:6, :])[None,:,:].expand(self.num_envs, -1, -1)

        # transform input force mapping to world frame
        F_W = torch.zeros_like(F_B)
        for i in range(F_B.shape[2]):  # For each of the 6 force vectors
            F_W[:, :, i] = quat_rotate(body_q[:, 3:7], F_B[:, :, i])
      
        J = torch.zeros((num_envs, 6, 6), device=self.device, dtype=body_qd.dtype)
        J[:, 0:3, :] = Tau_B
        J[:, 3:6, :] = F_W

        # Solve: u = pinv(F_world) @ ((M @ body_qdd_cmd) - f_W)
        rhs = (M @ body_qdd_cmd_W[:,:,None]) - f_W[:,:,None]
        rpm_cmd = torch.linalg.pinv(J) @ rhs       
        rpm_cmd = torch.sign(rpm_cmd) * torch.sqrt(torch.abs(rpm_cmd)) / (self.max_rpm * 2 * math.pi / 60)
        rpm_cmd = torch.clamp(rpm_cmd, -1.0, 1.0)        

        self.last_power_draw = torch.sum(rpm_cmd**2, dim=1).squeeze() / 6.0  # normalized power draw (0 to 1)
        self.control.assign("body_force", rpm_cmd.squeeze(-1))

    def compute_depth_observations(self, model, state):
        if not self.use_depth_observations:
            return
            
        if self.use_depth_observations and len(self.indices_shape_with_collision) > 0:        
            wp.launch(
                render_depth_kernel,
                dim=(self.num_envs, self.cam_width * self.cam_height),
                inputs=[
                    state.body_q,
                    self.sim_view._drone_indices_wp,
                    model.shape_transform,
                    model.shape_geo,
                    self.indices_shape_with_collision,                    
                    self._cam_dirs_local_wp,
                    self._cam_pos_local_wp,                    
                    float(self.cam_min_depth),
                    float(self.cam_max_depth),
                ],
                outputs=[self._cam_depths_wp],
                device=model.device,
            )   

    def compute_collision_distances(self, model, state):
        """Compute collision costs between drones and obstacles using SDF with graph capture"""
        if not self.use_collision_distances:
            return

        # Reset collision costs
        self._collision_distances.zero_()

        wp.launch(
            collision_distance_kernel,
            dim=(self.num_envs, len(self.indices_shape_with_collision)),
            inputs=[
                state.body_q,  # Detach to remove gradients for Warp kernel
                self.indices_shape_with_collision,
                self.sim_view._drone_indices_wp,
                model.shape_transform,
                model.shape_geo,  
                model.shape_body,                                            
            ],
            outputs=[self._collision_distances],
            device=model.device
        )                

    def compute_imu_readings(self, body_q, body_qd, body_f, body_mass):
           
        omega_body_B = body_qd[:, 0:3] 

        # compute specific force measured by IMU        
        specific_force_B = quat_rotate(quat_conjugate(body_q[:, 3:7]), - body_f[:, 3:6] / body_mass[:,None])
        
        acc_imu_B = specific_force_B + self.imu_accel_bias + torch.randn_like(specific_force_B) * self.imu_accel_noise_std
        gyro_imu_B = omega_body_B + self.imu_gyro_bias + torch.randn_like(omega_body_B) * self.imu_gyro_noise_std
        
        return acc_imu_B, gyro_imu_B        
    
    def compute_observations(self):
        """Observations: pos, up-axis, vel, body-rates, target unit vector, target distance, last actions"""
        # Get drone states using the simulation view
        body_q, body_qd = self.sim_view.get_drone_states(self.state)

        # target up-axis in world frame
        target_up_axis_W = quat_rotate(self.target_body_q[:, 3:7], torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1))
        target_up_axis_B = quat_rotate(quat_conjugate(body_q[:, 3:7]), target_up_axis_W)

        # target heading and target distance
        pos_err_B = quat_rotate(quat_conjugate(body_q[:, 3:7]), self.target_body_q[:, 0:3] - body_q[:, 0:3])
        target_dist = torch.norm(pos_err_B, dim=1)[:,None]
        target_dir_B = pos_err_B / torch.clamp(target_dist, min=0.1)

        # Compute IMU readings (accelerometer and gyroscope)
        drone_idx = wp.to_torch(self.sim_view._drone_indices_wp)
        # body_f = wp.to_torch(self.state_0_bwd.body_f)[drone_idx]
        body_f = wp.to_torch(self.state_0.body_f)[drone_idx]
        body_mass = wp.to_torch(self.model.body_mass)[drone_idx]
        linAcc_body_B, angRates_body_B = self.compute_imu_readings(body_q, body_qd, body_f, body_mass)

        # purely body centric observations
        obs_parts = [
            linAcc_body_B / 9.81,
            angRates_body_B,
            target_dist, 
            target_dir_B,
            target_up_axis_B
        ]

        obs = torch.cat(obs_parts, dim=-1)

        # data hygiene
        nan_mask = torch.isnan(obs)
        if torch.any(nan_mask):
            print("NaN detected in observations")
            print(obs)
        obs = torch.where(nan_mask, torch.randn_like(obs) * 0.1, obs)
        obs = torch.clamp(obs, -2e1, 2e1)      
        
        # visual_obs = (wp.to_torch(self._cam_depths_wp) - self.cam_min_depth) / (self.cam_max_depth - self.cam_min_depth)
        if self.use_depth_observations:
            obs_dict = {
                'obs': obs,
                'visual_obs': wp.to_torch(self._cam_depths_wp).reshape(self.num_envs, self.cam_height, self.cam_width).unsqueeze(-1)
            }
            self.obs_buf = obs_dict
        else:
            self.obs_buf = obs

    def create_obs_space(self, num_obs_dict):                
        vector_obs_size = num_obs_dict.get('obs', 16)
        if self.use_depth_observations:
            depth_obs_shape = num_obs_dict.get('obs_depth', None)
        obs_spaces = {}
        
        # Main observation vector (pose, velocity, targets, etc.)
        obs_spaces['obs'] = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(vector_obs_size,), 
            dtype=np.float32
        )
        
        # Visual observations (depth images)
        if self.use_depth_observations and depth_obs_shape is not None:
            obs_spaces['visual_obs'] = spaces.Box(
                low=self.cam_min_depth,
                high=self.cam_max_depth,
                shape=(depth_obs_shape[1], depth_obs_shape[0], 1),  # (H, W, C)
                dtype=np.float32
            )
            
        return spaces.Dict(obs_spaces)

    def compute_reward(self):
        # Get drone states using the simulation view
        body_q, body_qd = self.sim_view.get_drone_states(self.state)

        pos = body_q[:,0:3]

        # target state mismatch      
        pos_err_norm = torch.norm(self.target_body_q[:, :3] - body_q[:, :3], dim=1)
        vel_err_norm = torch.norm(self.target_body_qd[:, 3:6] - body_qd[:, 3:6], dim=1)
        ang_vel_err = torch.norm(self.target_body_qd[:, :3] - body_qd[:, :3], dim=1)

        # ===================== Arena Boundary Barriers =========================
        arena_mins = self.arena_bounds[:, 0]  # [x_min, y_min, z_min]
        arena_maxs = self.arena_bounds[:, 1]  # [x_max, y_max, z_max]
        start_penalizing_before_arenabounds = 0.5  # meters before hitting the boundary
        beta_softplus_arenabounds = 3.0 # the higher the sharper the corner

        # how much outside of the arena is the drone?
        dist_to_min = arena_mins - pos
        dist_to_max = pos - arena_maxs

        how_much_outside, _ = torch.max(torch.cat((dist_to_min, dist_to_max), dim=1), dim=1)
        outside_penalty = -1.0 * torch.nn.functional.softplus(how_much_outside + start_penalizing_before_arenabounds, beta=beta_softplus_arenabounds)

        action_penalty = (
            -0.01 * torch.sum(self.last_actor_cmd[:, :3]**2, dim=-1) + # lin vel cmd pealty
            -0.2 * torch.abs(self.last_actor_cmd[:, 3]) # yaw rate cmd penalty
        )

        # ===================== Target =========================                   
        target_proximity_reward = 1.0 / (1.0 + pos_err_norm)   

        distance_gate = 1.0 / (1.0 + pos_err_norm**2)    

        target_rate_mismatch_penalty = - distance_gate * torch.exp(0.1 * ang_vel_err)
        target_rate_mismatch_penalty = torch.clamp(target_rate_mismatch_penalty, min=-5.0)

        # body up-axis in world frame
        body_up_axis = quat_rotate(body_q[:, 3:7], torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1))        
        # target up-axis in world frame
        target_up_axis = quat_rotate(self.target_body_q[:, 3:7], torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1))
        
        orientation_reward = 0.1 * distance_gate * (1.0 + (body_up_axis[:,None,:] @ target_up_axis[:,:,None]).squeeze())

        # ======================= Exploration ===============================
        dist_from_start = torch.norm(pos - self.spawn_positions, dim=1)
        dist_start_goal = torch.norm(self.target_body_q[:, :3] - self.spawn_positions, dim=1)
        exploration_reward = 0.2 * (1.0 - 1.0 / (1.0 + 10.0 * dist_from_start / (dist_start_goal + 1e-6)))

        # ====================== Angular Speed Barrier =========================
        ang_speed = torch.linalg.norm(body_qd[:, :3], dim=1)
        max_ang_speed = 1.0  # rad/s
        ratelimit_penalty = -(torch.exp(ang_speed / max_ang_speed) - 1.0)
        ratelimit_penalty = torch.clamp(ratelimit_penalty, min=-5.0)

        # ===================== Collision Barrier =========================        
        max_collision_penalty = 1.0
        steepness = 4.0
        if self.use_collision_distances:
            closest_collision_distance = wp.to_torch(self._collision_distances).min(dim=1)[0] 
            collision_penalty = -max_collision_penalty / (1.0 + (closest_collision_distance * steepness)**2)
        else:
            collision_penalty = 0.0

        progress_reward = 0.01 * self.progress_buf
        
        reward = (
            # -pos_err_norm * 1.0 
            + target_proximity_reward
            + exploration_reward
            # + target_rate_mismatch_penalty     
            # + orientation_reward 
            # + outside_penalty             
            # + collision_penalty
            # + action_penalty
            # + orientation_penalty
            # + progress_reward
        )

        # data hygiene
        nan_mask = torch.isnan(reward)
        if torch.any(nan_mask):
            print("NaN detected in reward")
            print(nan_mask)
            reward = torch.where(nan_mask, torch.randn_like(reward) * 1.0, reward)

        reward = torch.clamp(reward, -1e5, 1e5)

        # Truncation/termination (ant-like structure, but with sensible failsafes)
        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination

        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)

        if early_termination:
            # Hard termination only for extreme violations -> if way beyond barrier functions            
            is_too_fast = ang_speed > 10.0
            is_way_outside = how_much_outside > 1.0
            if self.use_collision_distances:
                is_colliding = closest_collision_distance < self.arm_specs[2] / 2.0 + 0.2
            else:
                is_colliding = torch.zeros_like(is_too_fast)

            terminated = is_too_fast | is_colliding | is_way_outside
            reset = torch.where(terminated, torch.ones_like(reset), reset)

            # reward[terminated] -= 1.0 # termination penalty            
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)

        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = reward, reset, terminated, truncated
