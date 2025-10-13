#!/bin/bash

echo "Running docker image... with a new user 'me' and password '123'"

# echo "once inside the container run: bash setup.sh, sudo password is '123'"

# if you get errors related with nvidia or gpus, you need to do the following:
# sudo apt install -y nvidia-docker2
# sudo systemctl daemon-reload
# sudo systemctl restart docker

BAGS_DIR='/data/go2'
DATA_DIR='/data'

xhost +local:root
sudo docker run -it --rm\
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
    --volume="$DATA_DIR:/data" \
    --env HOME=/home/me/tmp \
    --env XDG_CACHE_HOME=/home/me/tmp/.cache \
    --user me \
    --privileged \
    --name="real2usd" \
    hsu/ros2_humble_jammy_amd64:latest

# in a new terminal run:
# docker exec -it real2usd bash
