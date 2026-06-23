#!/usr/bin/env python3
"""Prepare already-georeferenced daily LAZ clouds for demo_csv_las_matching_gui.

The CSV/LAS GUI expects each cloud to be in its own local sensor frame, with
the world prior supplied through the CSV. Daily reference `merged.laz` clouds
are already in world/HK coordinates, so feeding them directly can trigger
small_gicp voxel coordinate overflow and double-apply the prior.

This script creates local-frame copies:

  root/YYYY-MM-DD/merged.laz
  root/YYYY-MM-DD/optimized_transforms.json

becomes:

  root/csv_las_gui_inputs_local/YYYY-MM-DD-00-00-00.laz
  root/csv_las_gui_inputs_local/poses.csv

For each day, it uses the first node's `world_matrix` in
`optimized_transforms.json` as the daily prior and transforms points with:

  local_xyz = R_prior.T @ (world_xyz - t_prior)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

import laspy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create local-frame LAZs and poses.csv for demo_csv_las_matching_gui."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing YYYY-MM-DD subdirs with merged.laz and optimized_transforms.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output dir (default: <root>/csv_las_gui_inputs_local).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of LAS points per chunk (default: 1,000,000).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Output LAS XYZ scale in metres (default: 0.001).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output LAZs.",
    )
    return parser.parse_args()


def day_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.startswith("20"):
            yield path


def first_world_matrix(transforms_path: Path) -> Tuple[str, np.ndarray]:
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", {})
    if not nodes:
        raise RuntimeError(f"no nodes in {transforms_path}")

    node_id = next(iter(nodes))
    matrix = np.asarray(nodes[node_id]["world_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"{transforms_path}: {node_id}.world_matrix is not 4x4")
    return node_id, matrix


def to_local(xyz: np.ndarray, prior: np.ndarray) -> np.ndarray:
    # xyz rows are world coordinates; for row vectors this is R^T * (world - t).
    return (xyz - prior[:3, 3]) @ prior[:3, :3]


def rpy_from_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
    roll = np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))
    pitch = np.degrees(np.arcsin(-matrix[2, 0]))
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return float(roll), float(pitch), float(yaw)


def local_bounds(src: Path, prior: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with laspy.open(str(src)) as reader:
        for index, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
            xyz = np.column_stack((points.x, points.y, points.z))
            local = to_local(xyz, prior)
            mins = np.minimum(mins, local.min(axis=0))
            maxs = np.maximum(maxs, local.max(axis=0))
            print(f"[bounds] {src.parent.name}: chunk {index}", flush=True)

    return mins, maxs


def write_local_laz(
    src: Path,
    dst: Path,
    prior: np.ndarray,
    mins: np.ndarray,
    chunk_size: int,
    scale: float,
) -> int:
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    with laspy.open(str(src)) as reader:
        header = laspy.LasHeader(point_format=reader.header.point_format, version=reader.header.version)
        header.scales = np.array([scale, scale, scale], dtype=np.float64)
        header.offsets = mins - 1.0

        total = 0
        with laspy.open(str(tmp), mode="w", header=header) as writer:
            for index, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
                xyz = np.column_stack((points.x, points.y, points.z))
                local = to_local(xyz, prior)

                # The chunk carries the source LAS scaling. Switch it before
                # assigning local coordinates, otherwise laspy can overflow.
                points.change_scaling(scales=header.scales, offsets=header.offsets)
                points.x = local[:, 0]
                points.y = local[:, 1]
                points.z = local[:, 2]

                writer.write_points(points)
                total += len(points)
                print(f"[write] {dst.name}: chunk {index}, points={total:,}", flush=True)

    tmp.replace(dst)
    return total


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or (root / "csv_las_gui_inputs_local")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day_dir in day_dirs(root):
        src = day_dir / "merged.laz"
        transforms = day_dir / "optimized_transforms.json"
        if not src.is_file() or not transforms.is_file():
            print(f"[skip] {day_dir.name}: missing merged.laz or optimized_transforms.json", flush=True)
            continue

        out_name = f"{day_dir.name}-00-00-00.laz"
        dst = output_dir / out_name
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst} exists; pass --overwrite to regenerate", flush=True)
            prior_node, prior = first_world_matrix(transforms)
        else:
            prior_node, prior = first_world_matrix(transforms)
            start = time.time()
            print(f"[bounds] {day_dir.name}: using prior node {prior_node}", flush=True)
            mins, maxs = local_bounds(src, prior, args.chunk_size)
            print(f"[write] {out_name}: local min={mins} max={maxs}", flush=True)
            total = write_local_laz(src, dst, prior, mins, args.chunk_size, args.scale)
            print(f"[done] {dst} points={total:,} elapsed={time.time() - start:.1f}s", flush=True)

        roll, pitch, yaw = rpy_from_matrix(prior)
        rows.append(
            (
                f"{day_dir.name}T00:00:00Z",
                out_name,
                prior[0, 3],
                prior[1, 3],
                prior[2, 3],
                roll,
                pitch,
                yaw,
                prior_node,
            )
        )

    if not rows:
        print(f"error: no usable day directories found under {root}", file=sys.stderr)
        return 1

    csv_path = output_dir / "poses.csv"
    csv_path.write_text(
        "effective_timestamp,cloud_file,hk1980_easting,hk1980_northing,hk1980_hkpd,"
        "roll_deg,pitch_deg,yaw_deg,prior_node\n"
        + "\n".join(
            f"{ts},{name},{e},{n},{u},{r},{p},{y},{prior_node}"
            for ts, name, e, n, u, r, p, y, prior_node in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
