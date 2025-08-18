import numpy as np 
import os
from pathlib import Path
import quaternion

site_path = Path("/home/gabriel/media/data/shek_kip_mei_excavation_site")
# date = "2025-08-15-09-30-00"
# node_prefix = "SKMES"
# georef_suffix = "georef"




def build_graph(site_path: Path, 
                date: str, 
                node_prefix: str, 
                georef_suffix: str = "georef" ):
    graph_path = site_path / "graph.txt"

    lines = []
    for entry in site_path.iterdir():
        if not entry.is_dir():
            print(f"{entry} is not a directory")
            continue
        if not entry.name.startswith(node_prefix):
            print(f"{entry} isnt a node dir")
            continue

        georef_file = entry / f"{date}_{georef_suffix}.txt"
        if not georef_file.exists():
            continue

        georef_matrix = np.loadtxt(georef_file)
        q = quaternion.from_rotation_matrix(georef_matrix[:3, :3])
        tx, ty, tz = georef_matrix[0:3, 3]

        lines.append(f"{entry.name} {tx} {ty} {tz} {q.x} {q.y} {q.z} {q.w}\n")

    Path(graph_path).write_text("".join(lines))