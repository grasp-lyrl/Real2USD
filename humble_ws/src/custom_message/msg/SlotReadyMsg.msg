# Slot ready for retrieval: a new SAM3D job completed; choose best object for this slot.
std_msgs/Header header
string job_id              # output/<job_id>/ has rgb.png, depth.npy, etc.
int64 track_id             # slot id in the world
string candidate_data_path # newly generated object path (use if FAISS empty or best match)
