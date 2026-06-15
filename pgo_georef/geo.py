"""Pose/rotation helpers shared across the package (no heavy dependencies)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def rotation_matrix_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a rotation matrix from roll/pitch/yaw using the ZYX convention.

    Matches the convention used by the existing metadata-to-CloudCompare tooling.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Return ``[qx, qy, qz, qw]`` for the rotation part of a 4x4/3x3 matrix."""
    r = mat[:3, :3]
    trace = np.trace(r)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_vec_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a ``[tx, ty, tz, qx, qy, qz, qw]`` vector to a 4x4 matrix."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(pose[3], pose[4], pose[5], pose[6])
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose_vec(matrix: np.ndarray) -> np.ndarray:
    q = matrix_to_quaternion(matrix)
    t = matrix[:3, 3]
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=np.float64)


def voxel_downsample(xyz: np.ndarray, voxel: Optional[float]) -> np.ndarray:
    if voxel is None or voxel <= 0.0 or len(xyz) == 0:
        return xyz
    xyz_min = xyz.min(axis=0)
    indices = np.floor((xyz - xyz_min) / voxel).astype(np.int64)
    _, keep = np.unique(indices, axis=0, return_index=True)
    return xyz[np.sort(keep)]
