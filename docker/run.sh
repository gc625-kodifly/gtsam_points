#!/bin/bash

xhost +local:docker  # More secure than 'xhost +'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "workspace dir: $WORKSPACE_DIR}"

docker run -it --rm --runtime=nvidia --net=host --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/home/admin/.Xauthority:rw \
  -v /dev:/dev \
  -v /home/gabriel/gtsam_points:/root/gtsam_points \
  -v /home/gabriel/gtsam_points/data:/root/gtsam_points/data \
  docker.io/gc625kodifly/gtsam_docker:focal_cuda12.2
  # docker.io/gc625kodifly/server:smart-eye