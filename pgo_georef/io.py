"""Discover scan directories and load their priors + point clouds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .geo import rotation_matrix_zyx, voxel_downsample

_CLOUD_SUFFIXES = (".laz", ".las")


@dataclass
class ScanNode:
    """A single pose-graph node loaded from one scan directory."""

    node_id: str
    directory: Path
    cloud_path: Path
    metadata_path: Path
    prior_matrix: np.ndarray  # 4x4 world pose from GNSS/IMU (before offset)
    points: np.ndarray  # Nx3 in the local sensor frame


def _json_to_matrix4x4(value, source: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"{source} is not a 4x4 matrix")
    return matrix


def metadata_to_matrix(metadata_path: Path, use_antenna: bool = False) -> np.ndarray:
    """Return the GNSS/IMU world pose as a 4x4 matrix.

    Supports ``estimated_pose`` as a 4x4 list, ``estimated_pose`` as the
    HK1980 + roll/pitch/yaw object, or a ``transform.matrix`` fallback.
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if "estimated_pose" not in metadata:
        transform = metadata.get("transform")
        if isinstance(transform, dict) and "matrix" in transform:
            return _json_to_matrix4x4(transform["matrix"], f"{metadata_path}: transform.matrix")
        raise KeyError(f"{metadata_path}: no estimated_pose or transform.matrix")

    pose = metadata["estimated_pose"]
    if isinstance(pose, list):
        return _json_to_matrix4x4(pose, f"{metadata_path}: estimated_pose")

    if use_antenna and "antenna_hk1980_easting_m" in pose:
        tx = pose["antenna_hk1980_easting_m"]
        ty = pose["antenna_hk1980_northing_m"]
        tz = pose["antenna_hk1980_hkpd_m"]
    else:
        tx = pose["hk1980_easting_m"]
        ty = pose["hk1980_northing_m"]
        tz = pose["hk1980_hkpd_m"]

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix_zyx(pose["roll_deg"], pose["pitch_deg"], pose["yaw_deg"])
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def find_cloud(directory: Path, cloud_glob: Optional[str] = None) -> Optional[Path]:
    if cloud_glob:
        matches = sorted(directory.glob(cloud_glob))
        return matches[0] if matches else None
    for suffix in _CLOUD_SUFFIXES:
        matches = sorted(directory.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def find_metadata(directory: Path) -> Optional[Path]:
    candidate = directory / "metadata.json"
    if candidate.is_file():
        return candidate
    matches = sorted(directory.glob("metadata.json"))
    return matches[0] if matches else None


def load_points(
    cloud_path: Path,
    voxel: Optional[float] = None,
    max_points: int = 0,
) -> np.ndarray:
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError("reading .las/.laz requires laspy") from exc

    las = laspy.read(str(cloud_path))
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)

    xyz = voxel_downsample(xyz, voxel)
    if max_points > 0 and len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[idx]
    return np.ascontiguousarray(xyz, dtype=np.float64)


def load_scan_node(
    directory: Path,
    *,
    voxel: Optional[float] = None,
    max_points: int = 0,
    use_antenna: bool = False,
    cloud_glob: Optional[str] = None,
) -> ScanNode:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    cloud_path = find_cloud(directory, cloud_glob)
    if cloud_path is None:
        raise FileNotFoundError(f"no .laz/.las cloud in {directory}")
    metadata_path = find_metadata(directory)
    if metadata_path is None:
        raise FileNotFoundError(f"no metadata.json in {directory}")

    prior_matrix = metadata_to_matrix(metadata_path, use_antenna)
    points = load_points(cloud_path, voxel, max_points)

    return ScanNode(
        node_id=directory.name,
        directory=directory,
        cloud_path=cloud_path,
        metadata_path=metadata_path,
        prior_matrix=prior_matrix,
        points=points,
    )


def load_scan_nodes(
    directories: List[Path],
    *,
    voxel: Optional[float] = None,
    max_points: int = 0,
    use_antenna: bool = False,
    cloud_glob: Optional[str] = None,
) -> List[ScanNode]:
    nodes: List[ScanNode] = []
    seen_ids = set()
    for directory in directories:
        node = load_scan_node(
            directory,
            voxel=voxel,
            max_points=max_points,
            use_antenna=use_antenna,
            cloud_glob=cloud_glob,
        )
        node_id = node.node_id
        suffix = 1
        while node_id in seen_ids:
            suffix += 1
            node_id = f"{node.node_id}_{suffix}"
        if node_id != node.node_id:
            node.node_id = node_id
        seen_ids.add(node_id)
        nodes.append(node)
    return nodes
