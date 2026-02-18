import os
from glob import glob
from setuptools import find_packages, setup

package_name = "real2usd"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
            (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
            (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
            (os.path.join('share', package_name, 'scripts_r2u'), glob(os.path.join('scripts_r2u', '*'))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="christopher hsu",
    maintainer_email="chsu8@seas.upenn.edu",
    description="a package that does observes the world and builds a usd",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lidar_cam_node = real2usd.lidar_cam_node:main",
            "retrieval_node = real2usd.retrieval_node:main",
            "isaac_lidar_node_preprocessed = real2usd.isaac_lidar_node_preprocessed:main",
            "registration_node = real2usd.registration_node:main",
            "usd_buffer_node = real2usd.usd_buffer_node:main",
            "navigator_llm_node = real2usd.navigator_llm_node:main",
            "overlay_node = real2usd.usd_overlay_node:main",
            "vid_recorder_node = real2usd.vid_recorder_node:main",
            "isaac_lidar_node = real2usd.nonessential_nodes.isaac_lidar_node:main",   
            "pc_node = real2usd.nonessential_nodes.pc_node:main",
            "stats_node = real2usd.nonessential_nodes.stats_node:main",
            "timing_node = real2usd.nonessential_nodes.timing_node:main",
            "color_pc_node = real2usd.nonessential_nodes.color_pc_node:main",
        ],
    },
)
