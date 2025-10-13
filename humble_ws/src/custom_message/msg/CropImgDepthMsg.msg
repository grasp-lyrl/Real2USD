std_msgs/Header header

# RGB image
sensor_msgs/Image rgb_image

# Segmentation points as flat array [u1,v1,u2,v2,...]
int32[] seg_points  # Flattened array of u,v coordinates from numpy array

# Depth image
sensor_msgs/Image depth_image

# Camera information
sensor_msgs/CameraInfo camera_info

# Odometry information
nav_msgs/Odometry odometry

# Optional: Track ID for the cropped object
int64 track_id

# Optional: Label of the detected object
string label
