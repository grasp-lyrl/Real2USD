std_msgs/Header header
string data_path
int64 id
sensor_msgs/PointCloud2 src_pc
sensor_msgs/PointCloud2 targ_pc
geometry_msgs/Pose initial_pose   # optional; from SAM3D+go2. When set, registration uses this as starting pose (e.g. ICP from here to global PC).