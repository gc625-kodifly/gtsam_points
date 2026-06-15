"""High-level pose-graph optimization over a set of scan directories."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .binding import load_binding
from .geo import matrix_to_pose_vec, pose_vec_to_matrix
from .io import ScanNode, load_scan_nodes

Vec3 = Union[float, Sequence[float]]


def _as_vec3(value: Vec3) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    seq = [float(v) for v in value]
    if len(seq) != 3:
        raise ValueError("expected a scalar or 3 values")
    return seq


@dataclass
class PGOParams:
    """Tunable pose-graph optimization parameters (CPU only)."""

    voxel: Optional[float] = 0.25
    max_points: int = 0
    factor_type: str = "GICP"  # ICP | ICP_PLANE | GICP | VGICP (CPU only)
    optimizer: str = "LM"  # LM | ISAM2
    full_connection: bool = False
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 1)
    corr_rot_tol: float = 0.0
    corr_trans_tol: float = 0.0
    # GNSS prior standard deviations applied to every node.
    trans_sigma: Vec3 = (0.02, 0.02, 0.05)  # metres (1-3 cm horizontal is typical)
    rot_sigma_deg: Vec3 = (1.0, 1.0, 2.0)  # degrees (heading is the loose axis)
    min_trans_sigma: float = 0.0
    per_node_priors: bool = True
    use_antenna: bool = False
    cloud_glob: Optional[str] = None


@dataclass
class PGOResult:
    """Optimization output: one georeferenced world transform per node."""

    transforms: Dict[str, np.ndarray]  # node_id -> 4x4 world matrix
    initial_transforms: Dict[str, np.ndarray]  # node_id -> 4x4 GNSS prior matrix
    cloud_paths: Dict[str, Path]
    offset: np.ndarray  # XYZ subtracted before optimization
    termination_reason: str

    @property
    def node_ids(self) -> List[str]:
        return list(self.transforms.keys())


def run(
    directories: Sequence[Union[str, os.PathLike]],
    params: Optional[PGOParams] = None,
    *,
    build_dir: Optional[os.PathLike] = None,
) -> PGOResult:
    """Run pose-graph optimization over the given scan directories.

    Returns a :class:`PGOResult` mapping each node id to the georeferenced 4x4
    transform that places its local cloud into the HK1980 world frame.
    """
    params = params or PGOParams()
    dirs = [Path(d) for d in directories]
    if len(dirs) < 2:
        raise ValueError("pose-graph optimization needs at least two scan directories")

    gtsam_points_py = load_binding(build_dir)

    nodes: List[ScanNode] = load_scan_nodes(
        dirs,
        voxel=params.voxel,
        max_points=params.max_points,
        use_antenna=params.use_antenna,
        cloud_glob=params.cloud_glob,
    )

    # Shift everything near the origin so GTSAM stays numerically well conditioned.
    offset = nodes[0].prior_matrix[:3, 3].copy()

    frames = []
    initial_transforms: Dict[str, np.ndarray] = {}
    cloud_paths: Dict[str, Path] = {}
    for node in nodes:
        initial_transforms[node.node_id] = node.prior_matrix.copy()
        cloud_paths[node.node_id] = node.cloud_path

        shifted = node.prior_matrix.copy()
        shifted[:3, 3] -= offset
        # The GUI demos load LAS coordinates as float before building frames.
        # Round through float32 here so Python runs match that optimizer input.
        frame_points = np.asarray(node.points, dtype=np.float32).astype(np.float64)
        frames.append(gtsam_points_py.FrameData(node.node_id, matrix_to_pose_vec(shifted), frame_points))

    min_trans_sigma = float(params.min_trans_sigma)
    trans_sigma = [max(value, min_trans_sigma) for value in _as_vec3(params.trans_sigma)]
    rot_sigma = _as_vec3(params.rot_sigma_deg)

    opt_params = gtsam_points_py.OptimizerParams(
        params.full_connection,
        int(params.num_threads),
        float(params.corr_rot_tol),
        float(params.corr_trans_tol),
        params.optimizer,
        params.factor_type,
        trans_sigma[0],
        trans_sigma[1],
        trans_sigma[2],
        rot_sigma[0],
        rot_sigma[1],
        rot_sigma[2],
        params.per_node_priors,
        min_trans_sigma,
    )

    optimizer = gtsam_points_py.CostFactorMerge(opt_params)
    optimizer.load_frames(frames)
    optimized_shifted, stats = optimizer.run_optimization()

    transforms: Dict[str, np.ndarray] = {}
    for node in nodes:
        if node.node_id not in optimized_shifted:
            continue
        matrix = pose_vec_to_matrix(np.asarray(optimized_shifted[node.node_id], dtype=np.float64))
        matrix[:3, 3] += offset
        transforms[node.node_id] = matrix

    return PGOResult(
        transforms=transforms,
        initial_transforms=initial_transforms,
        cloud_paths=cloud_paths,
        offset=offset,
        termination_reason=getattr(stats, "termination_reason", ""),
    )
