# from https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/utils/torch_utils.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

# torch quat/vector utils


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def euler_to_rotation_matrix(roll, pitch, yaw, order: str = 'zyx'):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll: Rotation around x-axis (in radians)
        pitch: Rotation around y-axis (in radians)  
        yaw: Rotation around z-axis (in radians)
        order: Rotation order, default 'zyx' (z->y->x)
    
    Returns:
        3x3 rotation matrix
    """
    # Convert to tensors if needed
    if not isinstance(roll, torch.Tensor):
        roll = torch.tensor(roll, dtype=torch.float32)
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(pitch, dtype=torch.float32)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=torch.float32)
    
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    if order == 'zyx':
        # Z-Y-X rotation order (yaw->pitch->roll)
        R11 = cos_yaw * cos_pitch
        R12 = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
        R13 = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
        
        R21 = sin_yaw * cos_pitch
        R22 = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
        R23 = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
        
        R31 = -sin_pitch
        R32 = cos_pitch * sin_roll
        R33 = cos_pitch * cos_roll
        
    else:
        raise ValueError(f"Rotation order '{order}' not implemented")
    
    # Stack into rotation matrix
    row1 = torch.stack([R11, R12, R13], dim=-1)
    row2 = torch.stack([R21, R22, R23], dim=-1)
    row3 = torch.stack([R31, R32, R33], dim=-1)
    
    return torch.stack([row1, row2, row3], dim=-2)


@torch.jit.script
def quat_from_euler_ypr(yaw, pitch, roll):
    """Convert yaw, pitch, roll to quaternion [x, y, z, w]"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5) 
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    
    return torch.stack([x, y, z, w], dim=-1)


@torch.jit.script
def quat_to_yaw(q):
    """Extract yaw angle from quaternion [x, y, z, w]"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@torch.jit.script
def quat_error_to_axis_angle(q_error):
    """Convert quaternion error to axis-angle representation for control"""
    # Ensure we take the shortest path (positive w)
    w = q_error[:, 3]
    xyz = q_error[:, :3]
    
    # Flip quaternion if w < 0 to ensure shortest path
    flip = (w < 0).float()
    w = torch.abs(w)
    xyz = xyz * (1 - 2 * flip.unsqueeze(-1))
    
    # For small angles, use linear approximation: axis*angle ≈ 2*xyz
    small_angle = w > 0.9999  # cos(angle/2) > cos(0.01) 
    
    # Small angle case
    axis_angle_small = 2.0 * xyz
    
    # Regular case: axis*angle = 2*atan2(|xyz|, w) * xyz/|xyz|
    xyz_norm = torch.norm(xyz, dim=-1, keepdim=True)
    safe_norm = torch.clamp(xyz_norm, min=1e-8)
    angle = 2.0 * torch.atan2(xyz_norm.squeeze(-1), w)
    axis_angle_regular = (xyz / safe_norm) * angle.unsqueeze(-1)
    
    return torch.where(small_angle.unsqueeze(-1), axis_angle_small, axis_angle_regular)

    