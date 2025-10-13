import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    with_rviz2 = LaunchConfiguration('rviz2', default='true')

    rviz_config = "real2usd_conf.rviz"

    ld = LaunchDescription([
        Node(
            package='real2usd',
            executable='lidar_cam_node',
        ),
        Node(
            package='real2usd',
            executable='retrieval_node',
        ),
        Node(
            package='real2usd',
            executable='isaac_lidar_node_preprocessed',
        ),
        Node(
            package='real2usd',
            executable='registration_node',
        ),
        Node(
            package='real2usd',
            executable='usd_buffer_node',
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            condition=IfCondition(with_rviz2),
            name='rviz2',
            arguments=['-d' + os.path.join(get_package_share_directory('real2usd'), 'config', rviz_config)]
        ),
    ])

    return ld
