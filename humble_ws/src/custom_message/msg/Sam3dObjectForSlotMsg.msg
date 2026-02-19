# Best object chosen for a slot; bridge uses this for registration (source=data_path, target=segment PC from job_id).
std_msgs/Header header
string job_id    # slot job: where to get target segment PC (output/<job_id>/ or input_processed/<job_id>/)
int64 id        # track_id
string data_path # best object path (GLB/PLY)
geometry_msgs/Pose pose
