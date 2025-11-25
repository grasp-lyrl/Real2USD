import os
from glob import glob
from setuptools import find_packages, setup

package_name = "real2sam3d"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
            (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
            (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
            (os.path.join('share', package_name, 'scripts_r2s3d'), glob(os.path.join('scripts_r2s3d', '*'))),
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
            "lidar_cam_node = real2sam3d.lidar_cam_node:main",
            "registration_node = real2sam3d.registration_node:main",
            "usd_buffer_node = real2sam3d.usd_buffer_node:main",
            "navigator_llm_node = real2sam3d.navigator_llm_node:main",
        ],
    },
)
