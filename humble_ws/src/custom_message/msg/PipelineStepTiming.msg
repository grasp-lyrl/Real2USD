# Pipeline profiling: one message per step (e.g. lidar_cam/segment, registration/icp).
std_msgs/Header header
string node_name   # e.g. lidar_cam_node, registration_node
string step_name   # e.g. frame_total, segment, crop_0, get_pose
float64 duration_ms
int64 sequence_id  # monotonic per node for ordering
