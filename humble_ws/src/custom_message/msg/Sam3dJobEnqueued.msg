# SAM3D job written to queue (input/<job_id>/). Used to infer worker latency when output appears.
std_msgs/Header header
string job_id
# Wall clock (sec) when crop was received, for E2E full-pipeline timing. Profiler uses this so start and end use same clock.
float64 frame_wall_sec