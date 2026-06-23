# pgo-georef

Minimal, **CPU-only** pose-graph georeferencing for pan/tilt LiDAR scanstations.

Each input **directory is one scan node**:

```
2026-06-10-11-39-20/
  2026-06-10-11-39-20.laz      # point cloud in the local sensor frame
  metadata.json                # estimated_pose (HK1980 E/N/U + roll/pitch/yaw)
  ...
```

The pipeline:

1. Reads each `metadata.json` and builds a GNSS/IMU world pose (HK1980 + RPY).
2. Loads + voxel-downsamples each cloud (local frame).
3. Builds a GTSAM pose graph: a **per-node GNSS prior** on every scan plus
   integrated GICP/VGICP scan-matching factors between scans.
4. Optimizes with Levenberg-Marquardt and applies the result to the raw clouds.

Output: one georeferenced 4x4 transform per scan (`optimized_transforms.json`
and CloudCompare matrices), plus the transformed `.laz` clouds and an optional
merged cloud.

## Requirements

- `numpy`, `laspy[laszip]`
- The CPU-built `gtsam_points_py` extension. Build it once:

```bash
cmake -B build_cpu -DBUILD_WITH_CUDA=OFF -DBUILD_PYTHON=ON -DBUILD_DEMO=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu --target gtsam_points_py -j
```

The package finds `build_cpu/python/gtsam_points_py*.so` automatically (override
with `--build-dir` or the `GTSAM_POINTS_BUILD_DIR` env var).

## Deploy to a new device (self-contained image)

Build once (context = repo root), optionally push to a registry:

```bash
docker build -f pgo_georef/Dockerfile -t pgo-georef:cpu .
```

On any device with Docker (no GPU, no GTSAM install needed):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v /data:/data \
  pgo-georef:cpu \
  /data/scanA /data/scanB /data/scanC \
  --voxel 0.25 --factor-type GICP --full-connection --merge \
  --output-dir /data/pgo_out
```

`--user "$(id -u):$(id -g)"` makes the outputs owned by you instead of the
container user.

## CLI

```bash
pgo-georef DIR1 DIR2 DIR3 ... \
  --voxel 0.25 \
  --factor-type GICP \
  --trans-sigma 0.02 0.02 0.05 \
  --rot-sigma-deg 1.0 1.0 2.0 \
  --full-connection \
  --merge \
  --output-dir /path/to/pgo_out
```

## Python API

```python
from pgo_georef import PGOParams, run, write_outputs

result = run(
    ["/data/2026-06-10-11-39-20", "/data/2026-06-10-11-45-00"],
    PGOParams(voxel=0.25, factor_type="GICP", trans_sigma=0.02, rot_sigma_deg=1.0),
)
write_outputs(result, "/data/pgo_out", merge=True)
```

## Parameters

| Param | Meaning |
|-------|---------|
| `voxel` | downsample voxel size (m); larger = faster, coarser |
| `factor_type` | `ICP`, `ICP_PLANE`, `GICP`, `VGICP` (all CPU) |
| `full_connection` | connect every scan pair (small N, weak overlap) vs neighbours |
| `trans_sigma` | GNSS position prior std-dev (m); tight (1-3 cm) trusts GNSS XYZ |
| `rot_sigma_deg` | orientation prior std-dev (deg); keep heading loose |
| `per_node_priors` | anchor every node with its GNSS prior (default); `--anchor-only` for first-node only |
