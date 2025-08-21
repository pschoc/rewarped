# pyright: reportGeneralTypeIssues=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
import os
import math
import torch
import warp as wp
import warp.sim  # ensure wp.sim types are loaded

from rewarped.warp_env import WarpEnv
from rewarped.warp.model_monkeypatch import Model_control
from rewarped.environment import IntegratorType
from .utils.torch_utils import normalize, quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate
from warp.sim.collide import box_sdf, capsule_sdf, cone_sdf, cylinder_sdf, mesh_sdf, plane_sdf, sphere_sdf


class Propeller:
    """Physics-based propeller model using thrust and drag coefficients"""
    def __init__(self):
        self.body = 0
        self.pos = (0.0, 0.0, 0.0)
        self.dir = (0.0, 1.0, 0.0)
        self.diameter = 0.0
        self.k_f = 0.0  # thrust coefficient
        self.k_d = 0.0  # drag coefficient
        self.turning_direction = 0.0
        self.moment_of_inertia = 0.0  # propeller moment of inertia around spin axis


def define_propeller(
    drone: int,
    pos: tuple,
    diameter: float = 0.2286,  # diameter in meters
    pitch: float = 0.1016,     # pitch in meters (4 inches)
    thickness: float = 0.01,   # thickness in meters
    density: float = 1600.0,   # carbon fiber density kg/m³
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
    prop.pos = (pos[0], pos[1], pos[2])
    prop.dir = (0.0, 1.0, 0.0)
    prop.diameter = diameter
    prop.k_f = k_f
    prop.k_d = k_d
    prop.turning_direction = turning_direction    
    
    return prop

@wp.kernel
def apply_drone_forces_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    drone_body_index: wp.array(dtype=int),
    body_force: wp.array(dtype=float, ndim=2),  # shape [num_envs, 4] # type: ignore
    prop_positions: wp.array(dtype=wp.vec3),  # shape [4]
    prop_dir_local: wp.vec3,  # (0,1,0)
    k_f: wp.array(dtype=float),  # shape [4] - thrust coefficients
    k_d: wp.array(dtype=float),  # shape [4] - drag coefficients
    turning_dir: wp.array(dtype=float),  # shape [4]    
    max_rpm: float,  # maximum RPM from motor specs
):
    env_id = wp.tid()
    body_idx = drone_body_index[env_id]

    tf = body_q[body_idx]
    
    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_torque = wp.vec3(0.0, 0.0, 0.0)

    # accumulate force/torque from 4 propellers
    for i in range(4):
        # user_act is normalized [0,1], scale to actual RPM
        # user_act = 1.0 corresponds to max_rpm
        u = body_force[env_id, i]
        actual_rpm = u * max_rpm
        
        # Convert RPM to angular velocity (rad/s)
        omega = actual_rpm * 2.0 * 3.14159 / 60.0
        omega_squared = omega * omega
        
        # Thrust: T = k_f * omega²
        thrust_magnitude = k_f[i] * omega_squared
        
        # Drag torque: Q = k_d * omega²  
        drag_torque_magnitude = k_d[i] * omega_squared
        
        prop_spin_axis_world = wp.transform_vector(tf, prop_dir_local)
        moment_arm = wp.transform_point(tf, prop_positions[i]) - wp.transform_point(tf, body_com[body_idx])

        # Apply thrust force in world frame        
        f_i = prop_spin_axis_world * thrust_magnitude

        # Torque from thrust force (moment arm)        
        t_i = wp.cross(moment_arm, f_i)

        # Drag torque: Q = k_d * omega² (opposes propeller rotation)
        # This creates a reaction torque on the drone body
        drag_torque = prop_spin_axis_world * drag_torque_magnitude * turning_dir[i]

        total_force += f_i
        total_torque += t_i - drag_torque
   
    total_torque *= 0.9 # dampening

    sf = body_f[body_idx]
    body_f[body_idx] = wp.spatial_vector(  # type: ignore[attr-defined]
        sf[0] + total_torque[0],
        sf[1] + total_torque[1],
        sf[2] + total_torque[2],
        sf[3] + total_force[0],
        sf[4] + total_force[1],
        sf[5] + total_force[2],
    )

class DroneParcour(WarpEnv):
    sim_name = "DroneParcour"    

    state_tensors_names = ("body_q", "body_qd")    
    control_tensors_names = ("body_force",)

    def __init__(self, num_envs=16, episode_length=1000, early_termination=True, **kwargs):
        
        # Extract environment parameters before calling super().__init__()
        device = kwargs.get("device", "cuda:0")       
        max_episode_length = kwargs.pop("max_episode_length", episode_length)
        early_termination = kwargs.pop("early_termination", early_termination)
        self.render_fps = kwargs.pop("render_fps", 10)  # Frames per second for rendering
        
        # Define bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.spawn_bounds = torch.tensor(kwargs.pop("spawn_bounds", [[-2.0, 2.0], [1.0, 3.0], [-2.0, 2.0]]), device=device)
        self.arena_bounds = torch.tensor(kwargs.pop("arena_bounds", [[-8.0, 8.0], [0.0, 6.0], [-8.0, 8.0]]), device=device)
        self.target_bounds = torch.tensor(kwargs.pop("target_bounds", [[-6.0, 6.0], [1.0, 5.0], [-6.0, 6.0]]), device=device)
        self.num_obstacles_max = kwargs.pop("num_obstacles_max", 5)
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
        
        # Propeller specifications (now in inches)
        self.prop_diameter_inch = kwargs.pop("prop_diameter_inch", 12)    # 12 inch diameter
        self.prop_pitch_inch = kwargs.pop("prop_pitch_inch", 4)          # 4 inch pitch
        self.prop_thickness = kwargs.pop("prop_thickness", 0.003)        # 3mm thick
        
        # Battery and motor specifications
        self.lipo_cells = kwargs.pop("lipo_cells", 6)                    # 6S battery
        self.motor_kv = kwargs.pop("motor_kv", 170)                      # 170 KV
        self.nominal_cell_voltage = kwargs.pop("nominal_cell_voltage", 3.7)  # 3.7V per cell
        self.action_penalty = kwargs.pop("action_penalty", 0.01)
        
        # Evaluation parameters (used by agent, not environment)
        self.num_eval_episodes = kwargs.pop("num_eval_episodes", 10)  # Remove from kwargs but don't need to store
        
        # Convert to metric for internal calculations
        self.prop_diameter = self.prop_diameter_inch * 0.0254  # inches to meters
        self.prop_pitch = self.prop_pitch_inch * 0.0254       # inches to meters

        # Extract render settings from config
        render = kwargs.pop("render", False)
        render_mode = kwargs.pop("render_mode", "none")
        
        # Simple observation like ant: position + velocity + target + actions = 16
        num_obs = 30
        num_act = 4  # Four thrust controls
  
        self.applied_body_force = torch.zeros((num_envs, 4), device=device, requires_grad=True)        
        self.ground_plane = kwargs.pop("ground_plane", True)  # Whether to activate ground plane collision
        # Now call super with only the parameters it expects
        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act,
            episode_length=max_episode_length,
            early_termination=early_termination,
            render=render,
            render_mode=render_mode,
            use_graph_capture=False,
            **kwargs,
        )
       
    def create_modelbuilder(self):
        """Create the model builder with drone-specific settings"""
        builder = super().create_modelbuilder()
        builder.rigid_contact_margin = 0.02
        
        return builder

    def add_shared_obstacles(self, builder):
        # ================ add obstacles
        random_factors_position = torch.rand((self.num_obstacles_max, 3))
        random_factors_shapes = torch.rand((self.num_obstacles_max, 3))
        random_orientations = torch.rand((self.num_obstacles_max, 3)) * 2.0 * math.pi

        obstacle_positions = random_factors_position * (self.arena_bounds[:, 1] - self.arena_bounds[:, 0]).cpu() + self.arena_bounds[:, 0].cpu()
        obstacle_shapes = random_factors_shapes * (self.obstacle_max_size - self.obstacle_min_size) + self.obstacle_min_size

        quat_roll = quat_from_angle_axis(random_orientations[:,0], torch.tensor([1.0, 0.0, 0.0]).expand(self.num_obstacles_max, -1))
        quat_pitch = quat_from_angle_axis(random_orientations[:, 1] , torch.tensor([0.0, 1.0, 0.0]).expand(self.num_obstacles_max, -1))
        quat_yaw = quat_from_angle_axis(random_orientations[:, 2], torch.tensor([0.0, 0.0, 1.0]).expand(self.num_obstacles_max, -1))

        # Combine rotations: yaw * pitch * roll (Z-Y-X convention)
        target_attitudes = quat_mul(quat_yaw, quat_mul(quat_pitch, quat_roll))

        self.obstacle_indices = torch.zeros((self.num_obstacles_max,), dtype=torch.int32)
        for i in range(self.num_obstacles_max):
            obstacle_idx = builder.add_body()                            

            builder.add_shape_box(
                obstacle_idx,
                hx = obstacle_shapes[i, 0],
                hy = obstacle_shapes[i, 1],
                hz = obstacle_shapes[i, 2],                
                is_solid=True,
                has_shape_collision=True,
                has_ground_collision=True,
                is_visible=True,
            )

            self.obstacle_indices[i] = obstacle_idx
        
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
                    (num_envs, 4), dtype=float, device=device, requires_grad=requires_grad
                )
            return c

        model.control = control_with_user_act.__get__(model, model.__class__)
        return model

    def create_articulation(self, builder):
        """Create the drone as a rigid body with automatic mass calculation."""        
        
        # Create the drone as a single rigid body
        body = builder.add_body()
        
        # Calculate virtual density to achieve target body mass
        body_volume = self.body_dimensions[0] * self.body_dimensions[1] * self.body_dimensions[2]
        virtual_density = self.body_mass / body_volume
        
        # convex collision sphere 
        builder.add_shape_sphere(
            body,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            radius=self.arm_specs[2]/2,
            is_visible=False,
            density=0.0,
            has_ground_collision=True,  # Enable ground collision detection
            has_shape_collision=True,
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
       
        # build props and arms
        props = []
        
        # arm length
        arm_length = self.arm_specs[2] / 2
        print(f"arm_length: {arm_length}")
        
        arm_rot = [
            math.pi/4, 
            -math.pi/4, 
            -math.pi*3/4, 
            math.pi*3/4
        ]

        # X-configuration: [CW, CCW, CW, CCW] for [front left, front right, back right, back left]
        turning_directions = [
            1.0,
            -1.0,
            1.0,
            -1.0
        ]

        prop_positions_b = []
        for rot in arm_rot:
            # Rotate the vector (arm_length, 0.0, 0.0) by arm_rot around Y axis using warp
            base_vector = wp.vec3(arm_length, 0.0, 0.0)
            rotation_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rot)
            rotated_vector = wp.quat_rotate(rotation_quat, base_vector)            
            prop_positions_b.append((rotated_vector[0], rotated_vector[1], rotated_vector[2]))        
        
        for i, (prop_position, turning_dir, rot) in enumerate(zip(prop_positions_b, turning_directions, arm_rot)):
            
            # add arm
            arm_center_pos_b = (prop_position[0]/2.0, prop_position[1]/2.0, prop_position[2]/2.0)
            builder.add_shape_cylinder(
                body,
                pos=arm_center_pos_b,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rot),
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
                pos=(prop_position[0], prop_position[1] + self.arm_specs[0]/2 + self.motor_specs[1]/2, prop_position[2]),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rot),
                up_axis=1,
                radius=self.motor_specs[0]/2,            
                half_height=self.motor_specs[1]/2, 
                density=motor_equivalent_density,
                is_solid=True,                      
                has_shape_collision=False,
                has_ground_collision=False,
            )
            
            props.append(
                define_propeller(
                    body,
                    pos=prop_position,
                    diameter=self.prop_diameter,
                    pitch=self.prop_pitch,
                    thickness=self.prop_thickness,
                    density=self.density_carbon,
                    turning_direction=turning_dir,
                )
            )
        
        # Store prop data for force calculations
        self.props = props
        self.turning_directions = turning_directions
        self.prop_positions = prop_positions_b        

    def init_sim(self):
        super().init_sim()

        with torch.no_grad():
            self.start_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

            # Initialize target positions for each environment
            self.target_body_q = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
            self.target_body_qd = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
            self.reset_targets()

        # Provide an external-forces callback via the model so newly-created
        # Control objects inherit it each step.
        self.model.apply_external_forces = self.apply_drone_forces

        # Create Warp arrays directly from stored prop data
        self._prop_positions_wp = wp.array(self.prop_positions, dtype=wp.vec3, device=self.model.device)  # type: ignore[attr-defined]
        self._k_f_wp = wp.array([prop.k_f for prop in self.props], dtype=float, device=self.model.device)  # type: ignore[attr-defined]
        self._k_d_wp = wp.array([prop.k_d for prop in self.props], dtype=float, device=self.model.device)  # type: ignore[attr-defined]
        self._turning_dir_wp = wp.array(self.turning_directions, dtype=float, device=self.model.device)  # type: ignore[attr-defined]        
        self._thrust_dir_local_wp = wp.vec3(0.0, 1.0, 0.0)  # type: ignore[attr-defined]
        # drone body index per env (assumes 1 dynamic body per env)
        drone_indices = list(range(self.num_envs))
        self._drone_body_index_wp = wp.array(drone_indices, dtype=int, device=self.model.device)  # type: ignore[attr-defined]
        
        # Display drone specifications
        self.print_drone_specs()
        
        # Setup drone parameters and cache computed values
        self.setup_drone()
        
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
        max_thrust_per_motor = self.props[0].k_f * (max_omega ** 2)
        max_drag_per_motor = self.props[0].k_d * (max_omega ** 2)
        
        # Calculate total max thrust (4 motors)
        total_max_thrust = 4 * max_thrust_per_motor
        total_max_drag = 4 * max_drag_per_motor
        
        # Calculate thrust-to-weight ratio
        total_mass = wp.to_torch(self.model.body_mass)
        total_weight = total_mass * abs(self.gravity)
        thrust_to_weight_ratio = total_max_thrust / total_weight
        
        # Calculate body inertia and maximum angular acceleration
        body_inertia = self.model.body_inertia.numpy()  # [num_bodies, 3] - Ixx, Iyy, Izz
        if len(body_inertia) > 0:
            # Get inertia of first drone body (assuming all drones are identical)
            inertia = body_inertia[0]  # [Ixx, Iyy, Izz]
            
            # Calculate maximum torque from propellers
            # For a quadcopter with props at distance 'arm_length' from center
            arm_length = self.arm_specs[2] / 2.0  # half span = distance from center to prop
            
            # Maximum pitch/roll torque: differential thrust between front/back or left/right
            max_pitch_roll_torque = math.sqrt(2) * max_thrust_per_motor * arm_length
            
            # Maximum yaw torque: all props spinning in same direction contribute drag torque
            max_yaw_torque = total_max_drag
            
            # Maximum angular accelerations (rad/s²)
            max_roll_accel = max_pitch_roll_torque / inertia[0,0]   # around x-axis
            max_pitch_accel = max_pitch_roll_torque / inertia[1,1]  # around y-axis  
            max_yaw_accel = max_yaw_torque / inertia[2,2]           # around z-axis
        else:
            inertia = [0, 0, 0]
            max_roll_accel = max_pitch_accel = max_yaw_accel = 0
        
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
        print(f"   Total mass: {total_mass[0]:.3f} kg")
        print(f"   Body inertia (kg·m²):")
        print(f"     Ixx (roll):  {inertia[0,0]:.6f}")        
        print(f"     Izz (pitch):   {inertia[2,2]:.6f}")
        print(f"     Iyy (yaw): {inertia[1,1]:.6f}")

        print(f"\n⚡ Thrust & Performance:")
        print(f"   Max thrust per motor: {max_thrust_per_motor:.2f} N")
        print(f"   Max drag per motor: {max_drag_per_motor:.2f} N")
        print(f"   Total max thrust: {total_max_thrust:.2f} N")
        print(f"   Total max drag: {total_max_drag:.2f} N")
        print(f"   Total weight: {total_weight[0]:.2f} N ({total_mass[0]:.3f} kg)")
        print(f"   Thrust-to-weight ratio: {thrust_to_weight_ratio[0]:.2f}")
        
        print(f"\n🔄 Angular Performance:")
        print(f"   Max roll acceleration:  {max_roll_accel:.1f} rad/s² ({max_roll_accel*180/3.14159:.0f} deg/s²)")
        print(f"   Max pitch acceleration: {max_pitch_accel:.1f} rad/s² ({max_pitch_accel*180/3.14159:.0f} deg/s²)")
        print(f"   Max yaw acceleration:   {max_yaw_accel:.1f} rad/s² ({max_yaw_accel*180/3.14159:.0f} deg/s²)")
        
        # Overall performance based on thrust-to-weight ratio
        if thrust_to_weight_ratio[0] > 2.0:
            print(f"   🚀 Performance: Excellent (>2.0)")
        elif thrust_to_weight_ratio[0] > 1.5:
            print(f"   ✈️ Performance: Good (1.5-2.0)")
        elif thrust_to_weight_ratio[0] > 1.0:
            print(f"   🛸 Performance: Adequate (1.0-1.5)")
        else:
            print(f"   ⚠️ Performance: Poor (<1.0) - may not hover!")
        
        # Responsiveness based on angular acceleration
        avg_angular_accel = (max_roll_accel + max_pitch_accel + max_yaw_accel) / 3.0
        if avg_angular_accel > 50.0:
            print(f"   ⚡ Responsiveness: Extremely agile (>{50:.0f} rad/s²)")
        elif avg_angular_accel > 30.0:
            print(f"   🎯 Responsiveness: Very agile ({30:.0f}-{50:.0f} rad/s²)")
        elif avg_angular_accel > 15.0:
            print(f"   🎮 Responsiveness: Agile ({15:.0f}-{30:.0f} rad/s²)")
        elif avg_angular_accel > 8.0:
            print(f"   📐 Responsiveness: Moderate ({8:.0f}-{15:.0f} rad/s²)")
        else:
            print(f"   🐌 Responsiveness: Sluggish (<{8:.0f} rad/s²)")
        
        print("="*60 + "\n")

    def apply_drone_forces(self, model, state, control=None):        
        wp.launch(
            kernel=apply_drone_forces_kernel,
            dim=self.num_envs,
            inputs=(
                state.body_q,
                model.body_com,
                state.body_qd,
                state.body_f,
                self._drone_body_index_wp,                
                control.body_force,
                self._prop_positions_wp,
                self._thrust_dir_local_wp,
                self._k_f_wp,
                self._k_d_wp,
                self._turning_dir_wp,                
                float(self.max_rpm),
            ),
            device=self.model.device,
        )

    def render(self, state=None):
        """Override render to show dynamic target spheres"""
        self.render_time += self.frame_dt
        if self.render_time % (1.0 / self.render_fps) < self.frame_dt and self.renderer is not None:

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

    def reset_targets(self):
        """Reset target positions to random locations within target bounds"""
        with torch.no_grad():            
            # Generate random positions within scaled target bounds
            random_factors = torch.rand([self.num_envs, 3], device=self.device)
            mins = self.target_bounds[:, 0]  # [x_min, y_min, z_min]
            maxs = self.target_bounds[:, 1]  # [x_max, y_max, z_max]
            ranges = maxs - mins        # [x_range, y_range, z_range]
            
            # generate random target attitudes            
            randomized_target_orientation = self.target_attitude_euler_deg.repeat(self.num_envs, 1) / 180.0 * math.pi
            randomized_target_orientation += torch.randn_like(randomized_target_orientation) * self.three_sigma_target_attitude_deg / 3.0 / 180.0 * math.pi

            # Create quaternions for each rotation axis
            quat_roll = quat_from_angle_axis(randomized_target_orientation[:, 0], torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1))
            quat_pitch = quat_from_angle_axis(randomized_target_orientation[:, 1] , torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1))
            quat_yaw = quat_from_angle_axis(randomized_target_orientation[:, 2], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
            
            # Combine rotations: yaw * pitch * roll (Z-Y-X convention)
            target_attitudes = quat_mul(quat_yaw, quat_mul(quat_pitch, quat_roll))
            
            self.target_body_q[:, 0:3] = mins + random_factors * ranges
            self.target_body_q[:, 3:7] = target_attitudes # Set target orientations as quaternions
            
            # Set target velocities
            self.target_body_qd[:, 3:6] = self.target_lin_vel.repeat(self.num_envs, 1) + torch.randn_like(self.target_lin_vel) * self.three_sigma_target_lin_vel / 3.0
            self.target_body_qd[:, :3] = self.target_ang_vel.repeat(self.num_envs, 1) + torch.randn_like(self.target_ang_vel) * self.three_sigma_target_ang_vel / 3.0

    @torch.no_grad()
    def randomize_init(self, env_ids):        
        """Randomize drone spawn positions following ant pattern"""
        # For rigid body drone, use body_q instead of joint_q
        body_q = self.state.body_q[int(self.obstacle_indices[-1]+1):, :]
        body_qd = self.state.body_qd[int(self.obstacle_indices[-1]+1):, :]

        N = len(env_ids)
        
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

        ##  ================== sample initial velocity ========================
        # linear random velocity
        sigma_initial_lin_vel_tensor = torch.tensor(self.three_sigma_initial_lin_vel, device=self.device) / 3.0
        body_qd[env_ids, 3:6] = torch.randn_like(body_qd[env_ids, 3:6]) * sigma_initial_lin_vel_tensor

        # angular random velocity
        sigma_initial_ang_vel_tensor = torch.tensor(self.three_sigma_initial_ang_vel, device=self.device) / 3.0
        body_qd[env_ids, :3] = torch.randn_like(body_qd[env_ids, :3]) * sigma_initial_ang_vel_tensor
        
        self.reset_targets()

    def pre_physics_step(self, actions):        
        # Clamp actions to [-1, 1] range
        actions = torch.clamp(actions, -1.0, 1.0)        
        body_force = (actions + 1.0) * 0.5    
        self.applied_body_force = body_force.detach()       
        # body_force = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        self.control.assign("body_force", body_force)        
    
    def compute_observations(self):
        """Observations: pos, up-axis, vel, body-rates, target unit vector, target distance, last actions"""
        body_q = self.state.body_q.clone()
        body_qd = self.state.body_qd.clone()

        # position and orientation
        pos = body_q[:, 0:3]
        quat = body_q[:, 3:7]

        # up-axis in world (rotate local Y by body orientation)
        up_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
        up_axis = quat_rotate(quat, up_local)

        # linear velocity and body rates
        lin_vel = body_qd[:, 3:6]
        ang_vel = body_qd[:, 0:3]
        # target direction in body frame and distance
        vec_to_target_world = self.target_body_q[:, :3] - pos
        # vec_to_target_body = quat_rotate(quat_conjugate(quat), vec_to_target_world)
        # dir_to_target_body = normalize(vec_to_target_body)
        dir_to_target_world = vec_to_target_world / torch.norm(vec_to_target_world, dim=1, keepdim=True)
        dist_to_target = torch.norm(vec_to_target_world, dim=1, keepdim=True)

        # last actions
        last_actions = self.applied_body_force # shape (N, 4)

        # obs_parts = [
        #     pos,                # 3
        #     up_axis,            # 3 (world up)
        #     lin_vel,            # 3
        #     ang_vel,            # 3
        #     dir_to_target_world, # 3 (in world frame)
        #     dist_to_target,     # 1
        #     last_actions,       # 4
        # ]
        obs_parts = [
            body_q,
            body_qd,
            self.target_body_q,
            self.target_body_qd,
            last_actions
        ]
        obs = torch.cat(obs_parts, dim=-1)

        # Pad to configured num_obs if needed
        if hasattr(self, "num_obs") and obs.shape[-1] < self.num_obs:
            pad = torch.zeros((self.num_envs, self.num_obs - obs.shape[-1]), device=self.device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

        self.obs_buf = obs
            
    def compute_reward(self):
        # Basic state
        body_q = self.state.body_q.clone()
        body_qd = self.state.body_qd.clone()

        pos = body_q[:,0:3]

        # target state mismatch      
        pos_err_norm = torch.sum((self.target_body_q[:, 0:3] - body_q[:, 0:3])**2, dim=1)
        vel_err_norm = torch.norm(self.target_body_qd[:, 3:6] - body_qd[:, 3:6], dim=1)
        ang_vel_err = torch.norm(self.target_body_qd[:, :3] - body_qd[:, :3], dim=1)

        # Orientation error: q_err rotates current -> target
        q_err = quat_mul(body_q[:, 3:7], quat_conjugate(self.target_body_q[:, 3:7]))
        q_err_norm = torch.norm(q_err, dim=-1, keepdim=True)
        q_err = q_err / (q_err_norm + 1e-9)
        q_err = torch.where(q_err[:, 3:4] < 0.0, -q_err, q_err) # Enforce shortest rotation

        v = q_err[:, :3]
        w = q_err[:, 3:4]
        v_norm = torch.norm(v, dim=1, keepdim=True)
        ori_err_angle = 2.0 * torch.atan2(v_norm, torch.clamp(w, min=1e-9))  # (N,1)
        ori_err_axis = v / (v_norm + 1e-9)
        ori_err_vec = ori_err_axis * ori_err_angle                            # (N,3)

        # target proximity reward
        target_proximity_reward = 10.0 / (1.0 + pos_err_norm) 

        orientation_penalty = -1.0* torch.norm(ori_err_vec, dim=-1)
        target_rate_mismatch_penalty = -1.0*vel_err_norm - 1.0*ang_vel_err

        action_penalty = -0.01 * torch.sum(self.applied_body_force**2, dim=-1)

        # ===================== Arena Boundary Barriers =========================
        arena_mins = self.arena_bounds[:, 0]  # [x_min, y_min, z_min]
        arena_maxs = self.arena_bounds[:, 1]  # [x_max, y_max, z_max]
        start_penalizing_before_arenabounds = 1.0  # meters before hitting the boundary
        beta_softplus_arenabounds = 2 # the higher the sharper the corner

        # how much outside of the arena is the drone?
        dist_to_min = arena_mins - pos
        dist_to_max = pos - arena_maxs

        how_much_outside, _ = torch.max(torch.cat((dist_to_min, dist_to_max), dim=1), dim=1)
        outside_penalty = -torch.nn.functional.softplus(how_much_outside + start_penalizing_before_arenabounds, beta=beta_softplus_arenabounds)

        # ====================== Angular Speed Barrier =========================
        ang_speed = torch.linalg.norm(body_qd[:, :3], dim=1)
        max_ang_speed = 100.0  # rad/s
        start_penalizing_before_ratelimit = 10.0
        beta_softplus_ang_speed = 0.5  # the higher the sharper the corner
        how_much_spinning_too_fast = ang_speed - max_ang_speed
        ratelimit_penalty = -torch.nn.functional.softplus(how_much_spinning_too_fast + start_penalizing_before_ratelimit, beta=beta_softplus_ang_speed)

        progress_reward = 0.001 * self.progress_buf
        
        reward = (
            target_proximity_reward  # goal directedness
            # + progress_reward        # progress reward  
            # + action_penalty         # action cost
            + outside_penalty  # arena boundary barriers (differentiable)
            + ratelimit_penalty  # angular speed barrier (differentiable)
            # + orientation_penalty * target_proximity_reward # penalize orientation error at goal
            # + target_rate_mismatch_penalty * target_proximity_reward # penalize target rate mismatch at goal
        )

        # Truncation/termination (ant-like structure, but with sensible failsafes)
        reset_buf, progress_buf = self.reset_buf, self.progress_buf
        max_episode_steps, early_termination = self.episode_length, self.early_termination

        truncated = progress_buf > max_episode_steps - 1
        reset = torch.where(truncated, torch.ones_like(reset_buf), reset_buf)

        if early_termination:
            # Hard termination only for extreme violations -> if way beyond barrier functions            
            is_too_fast = how_much_spinning_too_fast > start_penalizing_before_ratelimit*3
            is_way_outside = how_much_outside > start_penalizing_before_arenabounds*3

            terminated = is_too_fast | is_way_outside
            reset = torch.where(terminated, torch.ones_like(reset), reset)
        else:
            terminated = torch.where(torch.zeros_like(reset), torch.ones_like(reset), reset)

        self.rew_buf, self.reset_buf, self.terminated_buf, self.truncated_buf = reward, reset, terminated, truncated
