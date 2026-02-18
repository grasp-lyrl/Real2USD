import os
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def _create_run_dir_and_set_queue(context, *args, **kwargs):
    """Create a per-run subdir under sam3d_queue_dir and set sam3d_queue_dir to it."""
    base = context.perform_substitution(LaunchConfiguration('sam3d_queue_dir'))
    use_subdir = context.perform_substitution(LaunchConfiguration('use_run_subdir')).lower() == 'true'
    if not use_subdir:
        return []
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'input').mkdir(exist_ok=True)
    (run_dir / 'output').mkdir(exist_ok=True)
    return [SetLaunchConfiguration('sam3d_queue_dir', TextSubstitution(text=str(run_dir)))]


def generate_launch_description():

    with_rviz2 = LaunchConfiguration('rviz2', default='true')
    with_sam3d_job_writer = LaunchConfiguration('sam3d_job_writer', default='true')
    with_sam3d_injector = LaunchConfiguration('sam3d_injector', default='true')
    run_sam3d_worker = LaunchConfiguration('run_sam3d_worker', default='false')
    use_run_subdir = LaunchConfiguration('use_run_subdir', default='true')
    sam3d_queue_dir = LaunchConfiguration('sam3d_queue_dir', default='/data/sam3d_queue')

    pkg_share = get_package_share_directory('real2sam3d')
    rviz_config = "real2usd_conf.rviz"
    worker_script = os.path.join(pkg_share, 'scripts_sam3d_worker', 'run_sam3d_worker.py')

    ld = LaunchDescription([
        DeclareLaunchArgument('rviz2', default_value='true', description='Run rviz2'),
        DeclareLaunchArgument('sam3d_job_writer', default_value='true',
                             description='Run SAM3D job writer (writes CropImgDepth to disk)'),
        DeclareLaunchArgument('sam3d_injector', default_value='true',
                             description='Run SAM3D injector (publishes worker results to /usd/StringIdPose)'),
        DeclareLaunchArgument('run_sam3d_worker', default_value='false',
                             description='Run SAM3D worker in this launch (uses --dry-run; for real SAM3D run worker in separate terminal with conda)'),
        DeclareLaunchArgument('use_run_subdir', default_value='true',
                             description='Create a new run subdir per launch (e.g. sam3d_queue/run_YYYYMMDD_HHMMSS) so each run has its own input/output'),
        DeclareLaunchArgument('sam3d_queue_dir', default_value='/data/sam3d_queue',
                             description='Base queue directory; if use_run_subdir=true, a run_<timestamp> subdir is created here'),
        OpaqueFunction(function=_create_run_dir_and_set_queue),
        # Pipeline: lidar_cam -> job_writer -> [worker] -> injector -> usd_buffer
        Node(
            package='real2sam3d',
            executable='lidar_cam_node',
        ),
        Node(
            package='real2sam3d',
            executable='registration_node',
        ),
        Node(
            package='real2sam3d',
            executable='usd_buffer_node',
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_job_writer_node',
            condition=IfCondition(with_sam3d_job_writer),
            parameters=[{'queue_dir': sam3d_queue_dir}],
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_injector_node',
            condition=IfCondition(with_sam3d_injector),
            parameters=[{'queue_dir': sam3d_queue_dir}],
        ),
        # Optional: run worker in same launch (run_sam3d_worker:=true; sam3d_worker_dry_run:=true to test without conda)
        ExecuteProcess(
            cmd=['python3', worker_script, '--queue-dir', sam3d_queue_dir, '--dry-run'],
            condition=IfCondition(run_sam3d_worker),
            name='sam3d_worker',
            output='screen',
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            condition=IfCondition(with_rviz2),
            name='rviz2',
            arguments=['-d', os.path.join(pkg_share, 'config', rviz_config)],
        ),
    ])

    return ld
