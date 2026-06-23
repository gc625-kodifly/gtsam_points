# Self-contained CPU-only pose-graph georeferencing image.
# Build context must be the gtsam_points repo root:
#   docker build -f pgo_georef/Dockerfile -t pgo-georef:cpu .
#
# The base image already provides GTSAM, Boost, Eigen, OpenMP, LASlib and
# Python 3.8, so we only build the CPU binding and install the Python package.
FROM docker.io/gc625kodifly/gtsam_docker:focal_cuda12.2

ARG PYBIND11_DIR=/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11

WORKDIR /opt/gtsam_points
COPY . /opt/gtsam_points

# Build the CPU-only binding (no CUDA) and its core lib.
RUN git config --global --add safe.directory /opt/gtsam_points || true && \
    cmake -B build_cpu -S . \
      -DBUILD_WITH_CUDA=OFF -DBUILD_PYTHON=ON -DBUILD_DEMO=OFF \
      -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=${PYBIND11_DIR} && \
    cmake --build build_cpu --target gtsam_points_py -j"$(nproc)"

# Make libgtsam_points.so loadable and the package importable.
ENV LD_LIBRARY_PATH=/opt/gtsam_points/build_cpu:${LD_LIBRARY_PATH}
ENV GTSAM_POINTS_BUILD_DIR=/opt/gtsam_points/build_cpu
ENV PYTHONPATH=/opt/gtsam_points:${PYTHONPATH}
RUN pip3 install --no-cache-dir "numpy>=1.20" "laspy[laszip]>=2.0"

# `docker run pgo-georef:cpu DIR1 DIR2 ... --voxel 0.25 --merge`
ENTRYPOINT ["python3", "-m", "pgo_georef.cli"]
