# Emitted when an object is added to the scene buffer (scene_graph.json). "Done" for full-pipeline E2E.
# inference_ms: worker-measured inference time (ms) from pose.json; 0 or negative = not set.
std_msgs/Header header
string job_id
float64 inference_ms
