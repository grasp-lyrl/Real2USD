#!/bin/bash

echo "Building docker image... with a new user 'me' and password '123'"

sudo docker build . --network=host -t hsu/ros:humble_jammy -f ubuntu22_amd64_humble_newuser.dockerfile

echo "once inside the container run: bash setup.sh, sudo password is '123'"

BAGS_DIR='/data/go2'
DATA_DIR='/data'

xhost +local:root
sudo docker run -it --rm \
    --network=host \
    --ipc=host \
    --pid=host \
    --gpus all \
    --env="DISPLAY=${DISPLAY}" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="./humble_ws:/home/me" \
    --volume="/home/$USER/.bash_aliases:/root/.bash_aliases" \
    --volume="$BAGS_DIR:/opt/bags" \
    --volume="/home/$USER/isaacsim:/opt/isaacsim" \
    --volume="/home/$USER/repos/sam-3d-objects:/home/me/src_Real2USD/real2sam3d/sam-3d-objects" \
    --volume="$DATA_DIR:/data" \
    --env HOME=/home/me/tmp \
    --env XDG_CACHE_HOME=/home/me/tmp/.cache \
    --user me \
    --privileged \
    --name="real2usd" \
    hsu/ros:humble_jammy

# in a new terminal run:
# docker exec -it real2usd bash