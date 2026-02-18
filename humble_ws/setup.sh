#!/bin/bash

cd /home/me/

# Create COLCON_IGNORE in tmp directory to prevent colcon from scanning it
# All hidden files will automatically go to /home/me/tmp since HOME is set there
mkdir -p "$HOME/.local" "$HOME/.cache/pip" "$HOME/.ros" 2>/dev/null || true
touch "$HOME/COLCON_IGNORE" 2>/dev/null || true

# this is for isaac sim its ros2 bridge
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/me/fastdds.xml

sudo apt-get update \
    && sudo apt-get -y --quiet --no-install-recommends install \
    gcc \
    git \
    python3 \
    python3-pip \
    wget

# go2_ros2_sdk installations
sudo apt install ros-$ROS_DISTRO-image-tools
sudo apt install ros-$ROS_DISTRO-vision-msgs
sudo apt install clang portaudio19-dev -y
pip install --user -r src_go2_ros2_webrtc_sdk/requirements.txt

pip install --user -r src_Real2USD/requirements.txt

rosdep update
rosdep install --from-paths src_go2_ros2_webrtc_sdk --ignore-src --rosdistro=humble -y
rosdep install --from-paths src_Real2USD --ignore-src --rosdistro=humble -y
rosdep install --from-paths src --ignore-src --rosdistro=humble -y

# Add local bin to PATH to avoid warnings
export PATH="$HOME/.local/bin:$PATH"

# ros2 building
rm -rf build install log
colcon build

source /opt/ros/humble/setup.sh
source install/setup.bash

export CONN_TYPE='webrtc'
export ROBOT_IP='192.168.0.211'