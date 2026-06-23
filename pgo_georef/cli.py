"""Command-line entry point: ``pgo-georef DIR... [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .apply import write_outputs
from .pipeline import PGOParams, run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgo-georef",
        description="CPU pose-graph georeferencing over per-scan directories.",
    )
    parser.add_argument("directories", type=Path, nargs="+", help="scan dirs, one node each (cloud + metadata.json)")
    parser.add_argument("--output-dir", type=Path, help="output dir (default: <first-dir-parent>/pgo_out)")
    parser.add_argument("--voxel", type=float, default=0.25, help="voxel size (m) for per-cloud downsampling")
    parser.add_argument("--max-points", type=int, default=0, help="cap each cloud to this many points (0 = no cap)")
    parser.add_argument(
        "--factor-type",
        choices=["ICP", "ICP_PLANE", "GICP", "VGICP"],
        default="GICP",
        help="pairwise scan-matching factor (CPU only)",
    )
    parser.add_argument("--optimizer", choices=["LM", "ISAM2"], default="LM")
    parser.add_argument("--full-connection", action="store_true", help="connect every pair instead of neighbours")
    parser.add_argument(
        "--no-bidirectional-factors",
        action="store_true",
        help="only add one pairwise factor per connected pair (default matches GUI: both directions)",
    )
    parser.add_argument("--num-threads", type=int, default=0, help="0 = use all cores")
    parser.add_argument("--corr-rot-tol", type=float, default=0.0)
    parser.add_argument("--corr-trans-tol", type=float, default=0.0)
    parser.add_argument(
        "--trans-sigma",
        type=float,
        nargs="+",
        default=[0.02, 0.02, 0.05],
        metavar="S",
        help="GNSS position prior sigma (m): one value or three (x y z)",
    )
    parser.add_argument(
        "--min-trans-sigma",
        type=float,
        default=0.0,
        help="floor applied to translation prior sigmas (GUI default behavior is 0.05)",
    )
    parser.add_argument(
        "--rot-sigma-deg",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 2.0],
        metavar="S",
        help="orientation prior sigma (deg): one value or three (roll pitch yaw)",
    )
    parser.add_argument("--anchor-only", action="store_true", help="prior only the first node, not every node")
    parser.add_argument("--antenna", action="store_true", help="use antenna HK1980 XYZ from metadata")
    parser.add_argument("--cloud-glob", help="glob for the cloud inside each dir, e.g. '*_las.las'")
    parser.add_argument("--merge", action="store_true", help="also write one merged georeferenced LAZ")
    parser.add_argument("--merged-output", type=Path, help="merged LAZ path (default: <output-dir>/merged.laz)")
    parser.add_argument("--no-clouds", action="store_true", help="only write transforms, do not rewrite clouds")
    parser.add_argument("--build-dir", type=Path, help="override the build dir holding gtsam_points_py")
    return parser


def _normalize_sigma(values):
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values
    raise SystemExit("sigma options take either 1 or 3 values")


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    params = PGOParams(
        voxel=args.voxel,
        max_points=args.max_points,
        factor_type=args.factor_type,
        optimizer=args.optimizer,
        full_connection=args.full_connection,
        bidirectional_factors=not args.no_bidirectional_factors,
        num_threads=args.num_threads if args.num_threads > 0 else (__import__("os").cpu_count() or 1),
        corr_rot_tol=args.corr_rot_tol,
        corr_trans_tol=args.corr_trans_tol,
        trans_sigma=_normalize_sigma(args.trans_sigma),
        min_trans_sigma=args.min_trans_sigma,
        rot_sigma_deg=_normalize_sigma(args.rot_sigma_deg),
        per_node_priors=not args.anchor_only,
        use_antenna=args.antenna,
        cloud_glob=args.cloud_glob,
    )

    output_dir = args.output_dir or (args.directories[0].expanduser().resolve().parent / "pgo_out")

    result = run(args.directories, params, build_dir=args.build_dir)
    print(f"[pgo] {result.termination_reason} | {len(result.transforms)} nodes optimized", file=sys.stderr)

    written = write_outputs(
        result,
        output_dir,
        merge=args.merge,
        merged_output=args.merged_output,
        write_clouds=not args.no_clouds,
    )
    print(f"[write] matrices: {written['matrices_txt']}", file=sys.stderr)
    print(f"[write] transforms json: {written['matrices_json']}", file=sys.stderr)
    for cloud in written["clouds"]:
        print(f"[write] {cloud}", file=sys.stderr)
    if written["merged"]:
        print(f"[write] merged: {written['merged']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
