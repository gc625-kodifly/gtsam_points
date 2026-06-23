"""Apply optimized transforms to the source clouds and write outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import PGOResult


def _transform_xyz(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return np.ascontiguousarray(xyz @ rotation.T + translation, dtype=np.float64)


def _transformed_las(source: Path, matrix: np.ndarray, offsets: Optional[np.ndarray] = None):
    import laspy

    las = laspy.read(str(source))
    xyz = np.column_stack((las.x, las.y, las.z))
    transformed = _transform_xyz(xyz, matrix)

    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales = np.asarray(las.header.scales, dtype=np.float64)
    header.offsets = transformed.min(axis=0) if offsets is None else np.asarray(offsets, dtype=np.float64)

    out = laspy.LasData(header)
    out.x = transformed[:, 0]
    out.y = transformed[:, 1]
    out.z = transformed[:, 2]
    for dimension in las.point_format.dimension_names:
        name = dimension.lower()
        if name in {"x", "y", "z"}:
            continue
        if hasattr(las, name):
            setattr(out, name, getattr(las, name))
    return out


def _cloudcompare_block(name: str, matrix: np.ndarray) -> str:
    lines = [f"# {name}"]
    for row in matrix:
        lines.append("  ".join(f"{value: .12f}" for value in row))
    return "\n".join(lines)


def write_outputs(
    result: PGOResult,
    output_dir,
    *,
    merge: bool = False,
    merged_output=None,
    write_clouds: bool = True,
) -> dict:
    """Write per-node georeferenced clouds, matrices, and an optional merged cloud.

    Returns a dict of the paths written.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {"clouds": [], "merged": None, "matrices_txt": None, "matrices_json": None}

    # Always emit the transforms (cheap, useful even without rewriting clouds).
    txt_blocks = []
    json_payload = {"offset": result.offset.tolist(), "nodes": {}}
    for node_id, matrix in result.transforms.items():
        cloud_name = result.cloud_paths[node_id].name
        txt_blocks.append(_cloudcompare_block(cloud_name, matrix))
        json_payload["nodes"][node_id] = {
            "cloud_file": cloud_name,
            "world_matrix": matrix.tolist(),
            "initial_matrix": result.initial_transforms[node_id].tolist(),
        }
    matrices_txt = output_dir / "optimized_cloudcompare_matrices.txt"
    matrices_txt.write_text("\n\n".join(txt_blocks) + "\n", encoding="utf-8")
    matrices_json = output_dir / "optimized_transforms.json"
    matrices_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    written["matrices_txt"] = matrices_txt
    written["matrices_json"] = matrices_json

    if not write_clouds and not merge:
        return written

    import laspy

    merge_writer = None
    merge_offsets = None
    merge_format = None
    merge_version = None
    merge_count = 0
    if merge and merged_output is None:
        merged_output = output_dir / "merged.laz"
    merged_output = Path(merged_output).expanduser().resolve() if merged_output else None

    for node_id, matrix in result.transforms.items():
        source = result.cloud_paths[node_id]
        out = _transformed_las(source, matrix)

        if write_clouds:
            destination = output_dir / source.name
            out.write(str(destination))
            written["clouds"].append(destination)

        if merged_output is not None:
            if merge_writer is None:
                merged_output.parent.mkdir(parents=True, exist_ok=True)
                merge_offsets = np.asarray(out.header.offsets, dtype=np.float64)
                merge_format = out.header.point_format.id
                merge_version = str(out.header.version)
                merge_header = laspy.LasHeader(
                    point_format=out.header.point_format, version=out.header.version
                )
                merge_header.scales = np.asarray(out.header.scales, dtype=np.float64)
                merge_header.offsets = merge_offsets
                merge_writer = laspy.open(str(merged_output), mode="w", header=merge_header)
            elif out.header.point_format.id != merge_format or str(out.header.version) != merge_version:
                raise SystemExit(f"cannot merge {source}: point format/version differs from first cloud")

            part = _transformed_las(source, matrix, offsets=merge_offsets)
            merge_writer.write_points(part.points)
            merge_count += len(part.points)

    if merge_writer is not None:
        merge_writer.close()
        written["merged"] = merged_output

    return written
