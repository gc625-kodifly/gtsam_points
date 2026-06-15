"""Minimal CPU-only pose-graph georeferencing for pan/tilt LiDAR scans.

Each input *directory* is one scan node: a point cloud (``.laz``/``.las``) in the
local sensor frame plus a ``metadata.json`` holding the GNSS/IMU ``estimated_pose``.
The pose graph anchors every node with its GNSS prior and ties neighbouring scans
together with integrated GICP/VGICP factors (CPU), producing one georeferenced
4x4 transform per scan.
"""

from .pipeline import PGOParams, PGOResult, run
from .apply import write_outputs

__all__ = ["PGOParams", "PGOResult", "run", "write_outputs"]
