# real2usd Pipeline and track_id

## Pipeline flow

1. **Camera node** (`lidar_cam_node` or `realsense_cam_node`)
   - Subscribes to: RGB, depth/camera_info (or lidar), odom.
   - Runs segmentation (and optional tracking when `use_tracking:=true`).
   - Publishes **one** `CropImgDepthMsg` per detected object to `/usd/CropImgDepth`.
   - Each message includes: crop image, **current-frame depth**, camera_info, odometry, **track_id**, label.
   - **RealSense and local depth**: For `realsense_cam_node`, by default `use_local_depth_for_global_pc:=true`: `/global_lidar_points` is the **current frame's** unprojected points only (no accumulation), so registration and buffer align against local depth like real2sam3d. Set to `false` to publish accumulated depth on `/global_lidar_points` instead.

2. **Retrieval node** (`retrieval_node`)
   - Subscribes to `/usd/CropImgDepth`.
   - For each crop: CLIP embedding → FAISS search → label filter → picks best USD.
   - Publishes `UsdStringIdPCMsg` to `/usd/StringIdPC` (USD path, **id = track_id**, point cloud).

3. **Isaac lidar node** (`isaac_lidar_node_preprocessed`)
   - Subscribes to `/usd/StringIdPC`, loads preprocessed source point cloud for the USD.
   - Publishes `UsdStringIdSrcTargMsg` to `/usd/StringIdSrcTarg` (src PC, target PC, id).

4. **Registration node** (`registration_node`)
   - Subscribes to `/usd/StringIdSrcTarg`, runs FGR registration.
   - Publishes `UsdStringIdPoseMsg` to `/usd/StringIdPose` (USD path, id, pose).

5. **USD buffer node** (`usd_buffer_node`)
   - Subscribes to `/usd/StringIdPose` and `/global_lidar_points`.
   - Uses **id** (track_id) as `obj_id` in `update_object_buffer(obj_id, ...)`.
   - **Update by track_id**: If `obj_id >= 0` and that track_id already exists in the buffer, the node **updates** that entry (position, orientation, USD path) instead of appending. So you get one buffer entry per track; repeated detections of the same track refine the same instance.
   - Does spatial clustering (its own `cluster_id`) and point-in-hull matching to decide which buffer entry matches the scene.

## How track_id is used

- **Set**: Camera node sets `CropImgDepthMsg.track_id` from the segmenter/tracker (-1 when untracked).
- **Passed through**: Retrieval node copies it to `UsdStringIdPCMsg.id`; registration passes it through.
- **Filtering** (optional): `retrieval_node` parameter `skip_untracked_detections:=true` skips crops with `track_id < 0`.
- **One instance per track**: `usd_buffer_node` keeps a map `track_id → buffer index`. When a pose arrives with a track_id that already exists, it updates that buffer entry instead of appending a new one.

## Limiting detections with track_id

To reduce noise and only add **tracked** detections to the scene, use the retrieval node option:

- **`skip_untracked_detections`** (default: `true` when you want to limit by track_id):  
  If `msg.track_id == -1`, the retrieval node does not run CLIP search and does not publish `UsdStringIdPC`. So only detections with a valid track_id are sent down the pipeline.

Use tracking on the camera node (`use_tracking:=true`) so that track_id is stable; then set `skip_untracked_detections:=true` on the retrieval node to limit detections to tracked objects only.
