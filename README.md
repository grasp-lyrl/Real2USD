# Real2USD: Scene Representations with Universal Scene Description Language ([Paper](https://arxiv.org/abs/2510.10778))
Authors: Christopher D. Hsu and Pratik Chaudhari

Large Language Models (LLMs) can help robots reason about abstract task specifications. This requires augmenting classical representations of the environment used by robots with natural language-based priors. There are a number of existing approaches to doing so, but they are tailored to specific tasks, e.g., visual-language models for navigation, language-guided neural radiance fields for mapping, etc. This paper argues that the Universal Scene Description (USD) language is an effective and general representation of geometric, photometric and semantic information in the environment for LLM-based robotics tasks. Our argument is simple: a USD is an XML-based scene graph, readable by LLMs and humans alike, and rich enough to support essentially any task---Pixar developed this language to store assets, scenes and even movies. We demonstrate a ``Real to USD'' system using a Unitree Go2 quadruped robot carrying LiDAR and a RGB camera that (i) builds an explicit USD representation of indoor environments with diverse objects and challenging settings with lots of glass, and (ii) parses the USD using Google's Gemini to demonstrate scene understanding, complex inferences, and planning. We also study different aspects of this system in simulated warehouse and hospital settings using Nvidia's Issac Sim.

1. ubuntu 22
2. ros2 humble
3. unitree go2 edu
4. isaac sim 4.5

For semantic navigation fully in simulation (IsaacSim) Sec III of the paper, please see my [MobilityGen Fork](https://github.com/christopher-hsu/MobilityGen).

For a baseline in Sec V, we utilize [Clio: Real-time Task-Driven Open-Set 3D Scene Graphs](https://arxiv.org/abs/2404.13696) from MIT. Please see our forked version of [Clio-Eval github](https://github.com/christopher-hsu/Clio-Eval), which requires RealSense data.

## Clone
```
git clone --recurse-submodules git@github.com:grasp-lyrl/Real2USD.git
```
if you just git clone and forget to use the `--recurse-submodules` flag you can do:
```
git submodule init
git submodule update
```
for the go2_ros2_webrtc_sdk please use the `update` branch, if not already on that branch
```
cd src_go2_ros2_webrtc_sdk 
git checkout update
```

## Docker container
We will work with ros2 humble in a docker container. First download [docker for ubuntu](https://docs.docker.com/engine/install/ubuntu/).

You will need to run the arm64 bash script if running on the jetson otherwise amd64 for standard desktop.

We will work off the `osrf/ros:humble-desktop-full-jammy` docker image that is ros2 humble and ubuntu 22 jammy. In the bash script that utilizes the dockerfile (ubuntu_22_humble_newuser.dockerfile) we want to add a user `me` to the docker image so that when we are in the container we are not root. This allows for us to work in the `humble_ws` as a volume that is mounted inside the container at `/home/me` but any read/write/create actions are not protected by permissions. The new user we create can still use sudo where the password is `123`. After we create the new image we use `docker run` to create the container. All /tmp files will be stored in local memory, e.g. `.cache`, `.local`, `.ros`, and `.rviz2` in `humble_ws/tmp`, feel free to change `--env` to fit you configuration, i.e. in `/tmp`. If you are running on the gpu you will need to add the flags `--gpu` see [docker run docs](https://docs.docker.com/reference/cli/docker/container/run/#gpus).

You will need to set a Ros2 bags directory and a database directory.

```
bash build_arm64_docker_ubuntu22_humble.sh
```
This creates a temporary docker image and runs the container that will be deleted if all sessions are exited. Once you are inside the docker container run the setup bash script.
```
bash setup.sh
```
This script will set up necessary packages. It will also set up ros2 humble with `colcon build` and source the build.

Now that the container has started we want to open new terminal sessions to access the container. You want to always keep one open so that the container continues to run. After you go into the container with `docker exec` source the ros2 installation.
```
docker exec -it what_changed bash
source install/setup.bash
```

### Subsequent uses of Docker container
You can commit the docker container to an image so that you dont have to go through building and running setup.sh again.

List the running containers and their images
```
docker ps
```
Commit the current docker (it must be running) so that it shows up in docker images
```
sudo docker commit <CONTAINER ID> <new image name> 
#for ex sudo docker commit ed2da6356e88 hsu/ros2_humble_jammy_arm64
```
See the docker images
```
docker images
```
And now you can run the "run_arm64.." bash scripts. Just make sure to change the last line in docker run to what you named it and include the tag, i.e. hsu/ros2_humble_jammy_arm64:latest

You should be in the docker container now! Just do the usual business of:
```
source install/setup.bash
```

## Isaac Sim 4.5
This code supports Isaac Sim 4.5. We follow [Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html)

source the ROS installation from docker and then you can run Isaac with ROS2 bridge
```
source ./humble_ws/install/setup.bash
~/isaacsim/isaacsim.sh
```


# Workflow

### Data and database setup
1. Download database to a database directory. For example we use `/data/`
2. If you want an example with a ros2 bag download the Ros2 bags to `/data/go2`
3. Preprocess database. This process will go do the setup for FAISS CLIP Similarity Search as well as index the USD database for point cloud collection. Update paths within file.
```
bash Real2USD/data_setup.sh
```
4. Using the outputs of the above process we need to collect the simulate point clouds in Isaac Sim and save them as pkl files. It is easier to run this code OUTSIDE of the docker container. The docker container is not set up to run Isaac Sim within it. Create a new terminal tab and run.
```
USD_LIST_PATH=src_Real2USD/real2usd/config/usd_list.txt
OUTPUT_DIR=/data/preprocessed_usd_data

python3 src_Real2USD/real2usd/scripts_isaacsim/preprocess_usd_dataset.py --usd_list $USD_LIST_PATH --output_dir $OUTPUT_DIR --start_id 0
```

### Running the code without a robot
Now that we have the database set up and bags downloaded run an example. You will need multiple terminal tabs.

1. Play the bag. 
```
ros2 bag play /opt/bags/example/
```
2. Launch the real2usd package
```
ros2 launch real2usd real2usd.launch.py
```

### Running the code with a robot
Start up your Unitree Go2. Follow the RoboVerse Unitree GO2 Ros2 SDK project setup where we utilize webRTC. The main requirement is to set the docker envirnoment variable `ROBOT_IP` which is also set in `setup.sh`. To get the robot ip, connect to your Go2 via the Unitree app and find your wifi ip address.

```
export CONN_TYPE='webrtc'
export ROBOT_IP='192.168.0.211'

ros2 launch go2_robot_sdk robot.launch.py
```
In another terminal, launch the real2usd package
```
ros2 launch real2usd real2usd.launch.py
```

### Visualizing the built USD in Isaac Sim
The `real2usd` package and launch file will save the registered objects in a buffer in a `json` file. This was done so that an Isaac Sim instance was not needed during execution. We can upload the json buffer file to Isaac Sim to create a USD scene. Once again, outside of the docker container run the following using the `python.sh` from Isaacsim.
```
BUFFER_PATH=/data/SimIsaacData/buffer/matched_buffer_123.json

~/isaacsim/python.sh ./humble_ws/src_Real2USD/scripts_isaacsim/usd_builder.py --buffer-file $BUFFER_PATH
```

Once you have the buffer file loaded into Isaac Sim, you can export it as a usda file type which is the file type that can be used as context to an LLM. You can also just use the buffer.json file as context.


### Semantic Task Navigation with USD as context
The following node will take as input the context file and a task that is published. You will need three terminals with docker running and a Go2. 

1. Start up go2_ros2_sdk webRTC connections
```
export CONN_TYPE='webrtc'
export ROBOT_IP='192.168.0.211'

ros2 launch go2_robot_sdk robot.launch.py
```
2. Start up the LLM navigator node that publishes Nav2 waypoints
```
ros2 run real2usd llm_navigator_node --ros-args -p context_file:=/path/to/your/context.txt
```
3. Publish a task message
```
ros2 topic pub --once /nav_query std_msgs/msg/String "{data: 'plan a path that goes to each of the chairs'}"
```

The navigator node will receive the task, utilize the context, and output structured outputs via Nav2 waypoints which is sent to the Go2 via webRTC. Your Go2 will then be able to semantically navigate!

## Isaac Sim ROS & ROS2 Workspaces
`src` packages are forked from from [IsaacSim-ros_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces?tab=readme-ov-file)

1. ROS2 drivers for go2 topics.
2. ROS2 nodes for images to objects in the world


## RoboVerse Unitree Go2 Ros2 SDK project
### mainly used if you want to use webrtc
`src_go2_ros2_sdk` is built from [go2_ros_sdk](https://github.com/abizovnuralem/go2_ros2_sdk) by [@abizovnuralem](https://github.com/abizovnuralem)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
![ROS2 Build](https://github.com/abizovnuralem/go2_ros2_sdk/actions/workflows/ros_build.yaml/badge.svg)
[![License](https://img.shields.io/badge/license-BSD--2-yellow.svg)](https://opensource.org/licenses/BSD-2-Clause)

We are happy to present you our integration of the Unitree Go2 with ROS2 over Wi-Fi, that was designed by the talented [@tfoldi](https://github.com/tfoldi). You can explore his groundbreaking work at [go2-webrtc](https://github.com/tfoldi/go2-webrtc).

This repo will empower your Unitree GO2 AIR/PRO/EDU robots with ROS2 capabilities, using both WebRTC (Wi-Fi) and CycloneDDS (Ethernet) protocols.

If you are using WebRTC (Wi-Fi) protocol, close the connection with a mobile app before connecting to the robot.
