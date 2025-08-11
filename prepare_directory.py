#!/usr/bin/env python3
"""
LAS → BIN conversion utility for the CWLPH_* dataset structure.

Features
--------
1.  **Random down-sampling** via `--keep <pct>` (legacy behaviour).
2.  **Spatial voxel down-sampling** via `--voxel <size>` (preferred).
    The voxel size is expressed in the same length units as the LAS file
    (usually metres).  Only the first point falling into each voxel is kept.

Only one of `--keep` or `--voxel` can be specified at a time; if neither is
given the full-resolution cloud is written.

The script also converts the 4×4 pose matrices in `init_pos.yaml`
into `graph.txt` (one line per frame: `id tx ty tz qx qy qz qw`).
"""

import argparse
from pathlib import Path
import numpy as np
import laspy
import yaml

# --------------------------------------------------------------------------- #
# Globals and numeric settings
# --------------------------------------------------------------------------- #
DTYPE = np.float32          # change to np.float64 if you need double precision
ROOT  = None                # filled at runtime (for prettier log output)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """
    Down-sample `xyz` by keeping the first point encountered in each voxel.

    Parameters
    ----------
    xyz : (N,3) float array
    voxel : float
        Edge length of the cubic voxel grid (same units as the LAS file).

    Returns
    -------
    (M,3) float array with M ≤ N.
    """
    if voxel <= 0:
        return xyz

    # Bring the cloud's minimum corner to the origin to avoid negative indices
    xyz_min = xyz.min(axis=0)
    indices = np.floor((xyz - xyz_min) / voxel).astype(np.int64)

    # np.unique on rows, return first occurrence per voxel
    _, uniq_idx = np.unique(indices, axis=0, return_index=True)
    return xyz[np.sort(uniq_idx)]


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """
    Convert a 4×4 pose matrix to quaternion (x,y,z,w).

    Assumes right-handed coordinate system and rotation in the upper-left 3×3.
    """
    R = mat[:3, :3]
    trace = np.trace(R)
    if trace > 0:
        S  = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S  = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S  = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S  = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=float)


def convert_las(
    las_path: Path,
    keep_pct: float,
    voxel_size: float ,
):
    """
    Convert a LAS file to BIN with optional down-sampling.

    Parameters
    ----------
    las_path : Path
    keep_pct : float
        Percentage (0–100) of points to randomly keep (ignored if voxel_size
        is given).
    voxel_size : float | None
        Edge length of spatial voxel grid (m).  If provided, overrides
        keep_pct.
    """
    out = las_path.with_name(f"{las_path.parent.name}.bin")

    # read LAS → Nx3 array
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T  # (N,3) float64

    original_n = len(xyz)

    if voxel_size is not None:
        xyz = voxel_downsample(xyz, voxel_size)
        print(
            f"   voxel={voxel_size} → {len(xyz):,}/{original_n:,} points "
            f"({100*len(xyz)/original_n:.1f}%)"
        )
    elif keep_pct < 100:
        m   = max(1, int(round(original_n * keep_pct / 100)))
        idx = np.random.choice(original_n, m, replace=False)
        xyz = xyz[idx]
        print(
            f"   keep={keep_pct:.1f}% → {len(xyz):,}/{original_n:,} points"
        )

    # write BIN in single-precision (or DTYPE)
    xyz.astype(DTYPE, copy=False).tofile(out)
    print(f"[LAS→BIN] {las_path.relative_to(ROOT)} → {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global ROOT

    parser = argparse.ArgumentParser(
        description="Convert LAS→BIN for each CWLPH_* directory and create "
        "graph.txt from init_pos.yaml."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Root folder containing init_pos.yaml and CWLPH_* sub-dirs",
    )

    # mutually exclusive down-sampling options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--keep",
        type=float,
        metavar="PCT",
        default=100.0,
        help="Randomly keep PCT percent of points (default 100 = all points)",
    )
    group.add_argument(
        "--voxel",
        type=float,
        metavar="SIZE",
        default=None,
        help="Voxel size (same units as LAS; e.g. 0.1 = 10 cm). "
        "Overrides --keep.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="name of nodes, eg: CWLPH"
    )


    args = parser.parse_args()
    ROOT = args.root_dir.resolve()
    node = args.nodes
    # --------------------------------------------------------------------- #
    # Load init_pos.yaml
    # --------------------------------------------------------------------- #
    init_file = ROOT / "init_pos.yaml"
    if not init_file.exists():
        raise SystemExit(f"❌ init_pos.yaml not found in {ROOT}")

    poses = yaml.safe_load(init_file.read_text())

    # --------------------------------------------------------------------- #
    # Convert each CWLPH_* directory
    # --------------------------------------------------------------------- #
    for sub in sorted(ROOT.iterdir()):
        if sub.is_dir() and sub.name.startswith(node):
            las_files = list(sub.glob("*.las"))
            if not las_files:
                print(f"⚠️  No .las found in {sub.name}, skipping")
                continue
            convert_las(
                las_path=las_files[0],
                keep_pct=args.keep,
                voxel_size=args.voxel,
            )

    # --------------------------------------------------------------------- #
    # Emit graph.txt
    # --------------------------------------------------------------------- #
    graph_path = ROOT / "graph.txt"
    with graph_path.open("w") as f:
        for key in sorted(poses):
            M = np.array(poses[key], dtype=float)
            tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
            qx, qy, qz, qw = matrix_to_quaternion(M)
            f.write(
                f"{key} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )

    print(f"✅ graph.txt written to {graph_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
