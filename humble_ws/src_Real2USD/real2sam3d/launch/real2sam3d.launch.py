import json
import os
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch_ros.actions import Node


def _create_run_dir_and_set_queue(context, *args, **kwargs):
    """Create a per-run subdir under sam3d_queue_dir, write current_run.json, and set sam3d_queue_dir."""
    base = context.perform_substitution(LaunchConfiguration('sam3d_queue_dir'))
    use_subdir = context.perform_substitution(LaunchConfiguration('use_run_subdir')).lower() == 'true'
    if not use_subdir:
        return []
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'input').mkdir(exist_ok=True)
    (run_dir / 'output').mkdir(exist_ok=True)
    # FAISS index for this run: indexer and retrieval use <run_dir>/faiss/ (same convention as index_sam3d_faiss)
    run_dir_resolved = str(run_dir.resolve())
    current_run = {
        "queue_dir": run_dir_resolved,
        "base_dir": str(Path(base).resolve()),
        "run_name": run_name,
        "faiss_index_path": run_dir_resolved,
        "created_at": datetime.now().isoformat(),
    }
    json_path = Path(base) / "current_run.json"
    try:
        with open(json_path, "w") as f:
            json.dump(current_run, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write {json_path}: {e}", file=__import__("sys").stderr)
    # Point sam3d_queue_dir and faiss_index_path at this run (index lives at <run_dir>/faiss/)
    return [
        SetLaunchConfiguration('sam3d_queue_dir', TextSubstitution(text=run_dir_resolved)),
        SetLaunchConfiguration('faiss_index_path', TextSubstitution(text=run_dir_resolved)),
    ]


def _write_run_config(context, *args, **kwargs):
    """Write launch arguments to run_config.json in the run directory so each experiment is self-described."""
    run_dir = context.perform_substitution(LaunchConfiguration('sam3d_queue_dir'))
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    launch_args = {
        "use_realsense_cam": context.perform_substitution(LaunchConfiguration('use_realsense_cam')),
        "sam3d_retrieval": context.perform_substitution(LaunchConfiguration('sam3d_retrieval')),
        "no_faiss_mode": context.perform_substitution(LaunchConfiguration('no_faiss_mode')),
        "glb_registration_bridge": context.perform_substitution(LaunchConfiguration('glb_registration_bridge')),
        "sam3d_job_writer": context.perform_substitution(LaunchConfiguration('sam3d_job_writer')),
        "sam3d_injector": context.perform_substitution(LaunchConfiguration('sam3d_injector')),
        "simple_scene_buffer": context.perform_substitution(LaunchConfiguration('simple_scene_buffer')),
        "pipeline_profiler": context.perform_substitution(LaunchConfiguration('pipeline_profiler')),
        "use_run_subdir": context.perform_substitution(LaunchConfiguration('use_run_subdir')),
        "run_sam3d_worker": context.perform_substitution(LaunchConfiguration('run_sam3d_worker')),
        "use_yolo_pf": context.perform_substitution(LaunchConfiguration('use_yolo_pf')),
        "enable_pre_sam3d_quality_filter": context.perform_substitution(LaunchConfiguration('enable_pre_sam3d_quality_filter')),
        "save_full_pointcloud": context.perform_substitution(LaunchConfiguration('save_full_pointcloud')),
        "pointcloud_save_period_sec": context.perform_substitution(LaunchConfiguration('pointcloud_save_period_sec')),
        "debug_verbose": context.perform_substitution(LaunchConfiguration('debug_verbose')),
        "realsense_min_depth_m": context.perform_substitution(LaunchConfiguration('realsense_min_depth_m')),
        "realsense_max_depth_m": context.perform_substitution(LaunchConfiguration('realsense_max_depth_m')),
        "lidar_min_range_m": context.perform_substitution(LaunchConfiguration('lidar_min_range_m')),
        "lidar_max_range_m": context.perform_substitution(LaunchConfiguration('lidar_max_range_m')),
        "rviz2": context.perform_substitution(LaunchConfiguration('rviz2')),
        "faiss_index_path": context.perform_substitution(LaunchConfiguration('faiss_index_path')),
        "use_init_odom": context.perform_substitution(LaunchConfiguration('use_init_odom')),
        "yaw_only_registration": context.perform_substitution(LaunchConfiguration('yaw_only_registration')),
        "registration_min_fitness": context.perform_substitution(LaunchConfiguration('registration_min_fitness')),
        "registration_icp_distance_threshold_m": context.perform_substitution(LaunchConfiguration('registration_icp_distance_threshold_m')),
        "registration_icp_max_iteration": context.perform_substitution(LaunchConfiguration('registration_icp_max_iteration')),
        "registration_min_target_points": context.perform_substitution(LaunchConfiguration('registration_min_target_points')),
        "registration_max_translation_delta_m": context.perform_substitution(LaunchConfiguration('registration_max_translation_delta_m')),
        "registration_target_mode": context.perform_substitution(LaunchConfiguration('registration_target_mode')),
        "registration_target_radius_m": context.perform_substitution(LaunchConfiguration('registration_target_radius_m')),
    }
    # Effective registration target: segment for RealSense (local points fed to SAM3D), else registration_target_mode
    use_rs = (context.perform_substitution(LaunchConfiguration('use_realsense_cam')) or '').strip().lower() == 'true'
    launch_args["registration_target_effective"] = "segment" if use_rs else (launch_args.get("registration_target_mode") or "global")
    config = {
        "run_dir": run_dir,
        "launch": launch_args,
        "created_at": datetime.now().isoformat(),
    }
    try:
        pkg_share = Path(get_package_share_directory('real2sam3d'))
        cfg_path = pkg_share / "config" / "tracking_pre_sam3d_filter.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            config["tracking_pre_sam3d_filter"] = {
                "path": str(cfg_path.resolve()),
                "content": cfg,
            }
            tracker_yaml = (cfg.get("tracker") or {}).get("tracker_yaml") if isinstance(cfg, dict) else None
            if tracker_yaml:
                tracker_path = pkg_share / "config" / str(tracker_yaml)
                if tracker_path.exists():
                    with open(tracker_path) as f:
                        config["tracker_yaml"] = {
                            "path": str(tracker_path.resolve()),
                            "content": f.read(),
                        }
    except Exception as e:
        print(f"[WARN] Could not embed tracking/filter config in run_config: {e}", file=__import__("sys").stderr)
    config_path = run_path / "run_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write {config_path}: {e}", file=__import__("sys").stderr)
    return []


def generate_launch_description():

    with_rviz2 = LaunchConfiguration('rviz2', default='true')
    with_sam3d_job_writer = LaunchConfiguration('sam3d_job_writer', default='true')
    with_sam3d_injector = LaunchConfiguration('sam3d_injector', default='true')
    with_glb_registration_bridge = LaunchConfiguration('glb_registration_bridge', default='true')
    with_sam3d_retrieval = LaunchConfiguration('sam3d_retrieval', default='true')
    with_simple_scene_buffer = LaunchConfiguration('simple_scene_buffer', default='true')
    with_pipeline_profiler = LaunchConfiguration('pipeline_profiler', default='true')
    no_faiss_mode = LaunchConfiguration('no_faiss_mode', default='false')
    run_sam3d_worker = LaunchConfiguration('run_sam3d_worker', default='false')
    use_yolo_pf = LaunchConfiguration('use_yolo_pf', default='false')
    enable_pre_sam3d_quality_filter = LaunchConfiguration('enable_pre_sam3d_quality_filter', default='false')
    save_full_pointcloud = LaunchConfiguration('save_full_pointcloud', default='true')
    pointcloud_save_period_sec = LaunchConfiguration('pointcloud_save_period_sec', default='5.0')
    debug_verbose = LaunchConfiguration('debug_verbose', default='false')
    realsense_min_depth_m = LaunchConfiguration('realsense_min_depth_m', default='0.2')
    realsense_max_depth_m = LaunchConfiguration('realsense_max_depth_m', default='4.0')
    lidar_min_range_m = LaunchConfiguration('lidar_min_range_m', default='0.2')
    lidar_max_range_m = LaunchConfiguration('lidar_max_range_m', default='8.0')
    registration_min_fitness = LaunchConfiguration('registration_min_fitness', default='0.10')
    registration_icp_distance_threshold_m = LaunchConfiguration('registration_icp_distance_threshold_m', default='0.03')
    registration_icp_max_iteration = LaunchConfiguration('registration_icp_max_iteration', default='50')
    registration_min_target_points = LaunchConfiguration('registration_min_target_points', default='30')
    registration_max_translation_delta_m = LaunchConfiguration('registration_max_translation_delta_m', default='2.5')
    registration_target_mode = LaunchConfiguration('registration_target_mode', default='global')
    registration_target_radius_m = LaunchConfiguration('registration_target_radius_m', default='2.5')
    use_run_subdir = LaunchConfiguration('use_run_subdir', default='true')
    sam3d_queue_dir = LaunchConfiguration('sam3d_queue_dir', default='/data/sam3d_queue')
    faiss_index_path = LaunchConfiguration('faiss_index_path', default='/data/sam3d_faiss')

    pkg_share = get_package_share_directory('real2sam3d')
    rviz_config = "real2sam3d_conf.rviz"
    worker_script = os.path.join(pkg_share, 'scripts_sam3d_worker', 'run_sam3d_worker.py')

    ld = LaunchDescription([
        DeclareLaunchArgument('rviz2', default_value='true', description='Run rviz2'),
        DeclareLaunchArgument('sam3d_job_writer', default_value='true',
                             description='Run SAM3D job writer (writes CropImgDepth to disk)'),
        DeclareLaunchArgument('sam3d_injector', default_value='true',
                             description='Run SAM3D slot node (publishes /usd/SlotReady; retrieval picks best object)'),
        DeclareLaunchArgument('sam3d_retrieval', default_value='true',
                             description='Run SAM3D retrieval node (FAISS+CLIP; publishes best object for each slot to bridge)'),
        DeclareLaunchArgument('no_faiss_mode', default_value='false',
                             description='No-FAISS mode: skip retrieval and have injector publish object-for-slot directly (set true when sam3d_retrieval:=false)'),
        DeclareLaunchArgument('glb_registration_bridge', default_value='true',
                             description='Run GLB→registration bridge (subscribes to /usd/Sam3dObjectForSlot, publishes src+targ for ICP)'),
        DeclareLaunchArgument('run_sam3d_worker', default_value='false',
                             description='Run SAM3D worker in this launch (uses --dry-run; for real SAM3D run worker in separate terminal with conda)'),
        DeclareLaunchArgument('use_yolo_pf', default_value='true',
                             description='Use prompt-free YOLOE segmentation weights (models/yoloe-11l-seg-pf.pt). Default false uses prompted model.'),
        DeclareLaunchArgument('enable_pre_sam3d_quality_filter', default_value='true',
                             description='Enable tracker tuning + strict pre-SAM3D quality filtering from config/tracking_pre_sam3d_filter.json.'),
        DeclareLaunchArgument('save_full_pointcloud', default_value='true',
                             description='Save full accumulated odom-frame point cloud snapshots to run_dir/diagnostics/pointclouds (npy).'),
        DeclareLaunchArgument('pointcloud_save_period_sec', default_value='5.0',
                             description='Periodic snapshot interval (seconds) for full point cloud saving. Final snapshot is also saved at shutdown.'),
        DeclareLaunchArgument('debug_verbose', default_value='false',
                             description='Enable verbose debug logging in camera nodes (lidar/realsense).'),
        DeclareLaunchArgument('realsense_min_depth_m', default_value='0.2',
                             description='RealSense reliability gate: minimum usable depth in meters.'),
        DeclareLaunchArgument('realsense_max_depth_m', default_value='4.0',
                             description='RealSense reliability gate: maximum usable depth in meters.'),
        DeclareLaunchArgument('lidar_min_range_m', default_value='0.2',
                             description='Lidar reliability gate: minimum usable range in meters.'),
        DeclareLaunchArgument('lidar_max_range_m', default_value='8.0',
                             description='Lidar reliability gate: maximum usable range in meters.'),
        DeclareLaunchArgument('use_run_subdir', default_value='true',
                             description='Create a new run subdir per launch (e.g. sam3d_queue/run_YYYYMMDD_HHMMSS) so each run has its own input/output'),
        DeclareLaunchArgument('sam3d_queue_dir', default_value='/data/sam3d_queue',
                             description='Base queue directory; if use_run_subdir=true, a run_<timestamp> subdir is created here'),
        DeclareLaunchArgument('faiss_index_path', default_value='/data/sam3d_faiss',
                             description='FAISS index base path (index at <path>/faiss/). When use_run_subdir=true (default), overridden to run_dir so index is <run_dir>/faiss/.'),
        DeclareLaunchArgument('simple_scene_buffer', default_value='true',
                             description='Run simple scene buffer node (writes scene_graph.json + scene.glb from /usd/StringIdPose)'),
        DeclareLaunchArgument('pipeline_profiler', default_value='true',
                             description='Run pipeline profiler and SAM3D profiler (timeline + sam3d_worker/inference latency)'),
        DeclareLaunchArgument('use_realsense_cam', default_value='false',
                             description='Use realsense_cam_node instead of lidar_cam_node (RealSense topics: aligned_depth, color, camera_info, /utlidar/robot_pose)'),
        DeclareLaunchArgument('use_init_odom', default_value='false',
                             description='Injector: normalize poses by first-frame odom (init_odom). Set true to match demo_go2 / compare; default false uses raw odom.'),
        DeclareLaunchArgument('yaw_only_registration', default_value='true',
                             description='Registration: when true, ICP result is projected to yaw-only (rotate about Z) so objects stay z-up. Set false for full 6-DOF.'),
        DeclareLaunchArgument('registration_min_fitness', default_value='0.10',
                             description='Registration quality gate: minimum ICP fitness to accept an update.'),
        DeclareLaunchArgument('registration_icp_distance_threshold_m', default_value='0.03',
                             description='ICP correspondence distance threshold in meters.'),
        DeclareLaunchArgument('registration_icp_max_iteration', default_value='50',
                             description='ICP maximum iterations.'),
        DeclareLaunchArgument('registration_min_target_points', default_value='30',
                             description='Minimum target points required for registration branch.'),
        DeclareLaunchArgument('registration_max_translation_delta_m', default_value='2.5',
                             description='Max allowed translation jump from initial pose before fallback.'),
        DeclareLaunchArgument('registration_target_mode', default_value='global',
                             description='Registration target source for bridge: global or segment.'),
        DeclareLaunchArgument('registration_target_radius_m', default_value='2.5',
                             description='When using global target, crop target points around initial pose within this radius.'),
        OpaqueFunction(function=_create_run_dir_and_set_queue),
        OpaqueFunction(function=_write_run_config),
        # Pipeline: [lidar_cam_node | realsense_cam_node] -> job_writer -> ... -> registration -> scene buffer
        Node(
            package='real2sam3d',
            executable='lidar_cam_node',
            condition=IfCondition(PythonExpression(["'", LaunchConfiguration('use_realsense_cam'), "' != 'true'"])),
            parameters=[{
                'use_yolo_pf': use_yolo_pf,
                'enable_pre_sam3d_quality_filter': enable_pre_sam3d_quality_filter,
                'save_full_pointcloud': save_full_pointcloud,
                'pointcloud_save_period_sec': pointcloud_save_period_sec,
                'pointcloud_save_root_dir': sam3d_queue_dir,
                'debug_verbose': debug_verbose,
                'lidar_min_range_m': lidar_min_range_m,
                'lidar_max_range_m': lidar_max_range_m,
            }],
        ),
        Node(
            package='real2sam3d',
            executable='realsense_cam_node',
            condition=IfCondition(LaunchConfiguration('use_realsense_cam')),
            parameters=[{
                'use_yolo_pf': use_yolo_pf,
                'enable_pre_sam3d_quality_filter': enable_pre_sam3d_quality_filter,
                'save_full_pointcloud': save_full_pointcloud,
                'pointcloud_save_period_sec': pointcloud_save_period_sec,
                'pointcloud_save_root_dir': sam3d_queue_dir,
                'debug_verbose': debug_verbose,
                'realsense_min_depth_m': realsense_min_depth_m,
                'realsense_max_depth_m': realsense_max_depth_m,
            }],
        ),
        Node(
            package='real2sam3d',
            executable='registration_node',
            parameters=[{
                'yaw_only_registration': LaunchConfiguration('yaw_only_registration', default='true'),
                'registration_min_fitness': registration_min_fitness,
                'registration_icp_distance_threshold_m': registration_icp_distance_threshold_m,
                'registration_icp_max_iteration': registration_icp_max_iteration,
                'registration_min_target_points': registration_min_target_points,
                'registration_max_translation_delta_m': registration_max_translation_delta_m,
                'registration_metrics_path': PythonExpression(["'", sam3d_queue_dir, "' + '/diagnostics/registration_metrics.jsonl'"]),
            }],
        ),
        # Node(
        #     package='real2sam3d',
        #     executable='usd_buffer_node',
        # ),
        Node(
            package='real2sam3d',
            executable='simple_scene_buffer_node',
            condition=IfCondition(with_simple_scene_buffer),
            parameters=[{'output_dir': sam3d_queue_dir}],
        ),
        Node(
            package='real2sam3d',
            executable='pipeline_profiler_node',
            condition=IfCondition(with_pipeline_profiler),
            parameters=[{'timing_log_dir': sam3d_queue_dir}],
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_profiler_node',
            condition=IfCondition(with_pipeline_profiler),
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_job_writer_node',
            condition=IfCondition(with_sam3d_job_writer),
            parameters=[{'queue_dir': sam3d_queue_dir, 'enable_pre_sam3d_quality_filter': enable_pre_sam3d_quality_filter}],
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_injector_node',
            condition=IfCondition(with_sam3d_injector),
            parameters=[{'queue_dir': sam3d_queue_dir, 'publish_object_for_slot': no_faiss_mode, 'use_init_odom': LaunchConfiguration('use_init_odom', default='false')}],
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_retrieval_node',
            condition=IfCondition(
                PythonExpression(["'", with_sam3d_retrieval, "' == 'true' and '", no_faiss_mode, "' == 'false'"])
            ),
            parameters=[{'queue_dir': sam3d_queue_dir, 'faiss_index_path': faiss_index_path}],
        ),
        Node(
            package='real2sam3d',
            executable='sam3d_glb_registration_bridge_node',
            condition=IfCondition(with_glb_registration_bridge),
            parameters=[{
                'queue_dir': sam3d_queue_dir,
                'world_point_cloud_topic': '/global_lidar_points',
                # RealSense: use segment (local) target — the same points fed into SAM3D — not the accumulated global PC. Lidar: use registration_target_mode (default global).
                'registration_target': PythonExpression([
                    "'segment' if '", LaunchConfiguration('use_realsense_cam'), "' == 'true' else '",
                    LaunchConfiguration('registration_target_mode'), "'"
                ]),
                'global_target_radius_m': registration_target_radius_m,
            }],
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
