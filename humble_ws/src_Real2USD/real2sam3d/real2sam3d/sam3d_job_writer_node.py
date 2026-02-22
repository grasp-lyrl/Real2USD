"""
SAM3D job writer node: subscribes to CropImgDepthMsg and writes job directories
for the SAM3D worker (disk-based handoff). Each job contains rgb.png, mask.png,
depth.npy, and meta.json so SAM3D can be run in a separate env/container.

Does NOT import or call SAM3D — only ROS, cv_bridge, OpenCV, numpy. Works
without SAM3D installed. The worker script (run_sam3d_worker.py) is the only
component that may load SAM3D; when run with --dry-run it skips inference.

Pure function write_sam3d_job() is used by the node and can be unit-tested
without ROS (see test/test_sam3d_job_writer.py).
"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from custom_message.msg import CropImgDepthMsg, PipelineStepTiming, Sam3dJobEnqueued
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def write_sam3d_job(
    rgb: np.ndarray,
    depth_full: np.ndarray,
    seg_pts: np.ndarray,
    meta: dict,
    job_path: Path,
    crop_bbox: Optional[List[int]] = None,
) -> None:
    """
    Write a SAM3D job directory (rgb.png, mask.png, depth.npy, meta.json).
    Callable without ROS for testing. Raises on invalid inputs.
    """
    job_path = Path(job_path)
    job_path.mkdir(parents=True, exist_ok=True)
    h_full, w_full = depth_full.shape[:2]

    if len(seg_pts) == 0 and not (crop_bbox and len(crop_bbox) >= 4):
        raise ValueError("Need seg_pts or crop_bbox to define crop region")

    if crop_bbox and len(crop_bbox) >= 4:
        x_min, y_min, x_max, y_max = crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]
        x_min = max(0, min(x_min, w_full - 1))
        y_min = max(0, min(y_min, h_full - 1))
        x_max = max(x_min + 1, min(x_max, w_full))
        y_max = max(y_min + 1, min(y_max, h_full))
    else:
        pad = 10
        cols, rows = seg_pts[:, 0], seg_pts[:, 1]
        x_min = int(np.clip(np.min(cols) - pad, 0, w_full - 1))
        y_min = int(np.clip(np.min(rows) - pad, 0, h_full - 1))
        x_max = int(np.clip(np.max(cols) + 1 + pad, 0, w_full))
        y_max = int(np.clip(np.max(rows) + 1 + pad, 0, h_full))

    depth_crop = depth_full[y_min:y_max, x_min:x_max].copy()

    mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    valid = (
        (seg_pts[:, 0] >= 0) & (seg_pts[:, 0] < w_full)
        & (seg_pts[:, 1] >= 0) & (seg_pts[:, 1] < h_full)
    )
    seg_valid = seg_pts[valid]
    if len(seg_valid) > 0:
        mask_full[seg_valid[:, 1], seg_valid[:, 0]] = 255
    mask_crop = mask_full[y_min:y_max, x_min:x_max]

    rh, rw = rgb.shape[:2]
    dh, dw = depth_crop.shape[:2]
    if (rh, rw) != (dh, dw):
        mask_crop = cv2.resize(mask_crop, (rw, rh), interpolation=cv2.INTER_NEAREST)
        depth_crop = cv2.resize(depth_crop, (rw, rh), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(job_path / "rgb.png"), rgb)
    cv2.imwrite(str(job_path / "mask.png"), mask_crop)
    np.save(str(job_path / "depth.npy"), depth_crop.astype(np.float32) / 1000.0)

    out_meta = dict(meta)
    out_meta["crop_bbox"] = [x_min, y_min, x_max, y_max]
    with open(job_path / "meta.json", "w") as f:
        json.dump(out_meta, f, indent=2)


class Sam3dDedupState:
    """
    Pure dedup state and logic for SAM3D job writer (track_id + position+label).
    Testable without ROS. Node delegates _should_skip_dedup / _record_written to this.
    """
    def __init__(self, dedup_track_id_sec: float, dedup_position_m: float):
        self.dedup_track_id_sec = dedup_track_id_sec
        self.dedup_position_m = dedup_position_m
        self._recent_jobs_max_age = max(dedup_track_id_sec, 1.0)
        self._track_id_last_write: dict = {}
        self._recent_jobs: List[Tuple[str, Tuple[float, float, float], float]] = []

    def should_skip(self, track_id: int, label: str, position: Tuple[float, float, float]) -> bool:
        now = time.monotonic()
        if self.dedup_track_id_sec > 0 and track_id >= 0 and track_id in self._track_id_last_write:
            if now - self._track_id_last_write[track_id] < self.dedup_track_id_sec:
                return True
        if self.dedup_position_m > 0 and label:
            pos_arr = np.array(position)
            for recent_label, recent_pos, recent_t in self._recent_jobs:
                if now - recent_t > self._recent_jobs_max_age:
                    continue
                if recent_label != label:
                    continue
                if np.linalg.norm(pos_arr - np.array(recent_pos)) <= self.dedup_position_m:
                    return True
        return False

    def record(self, track_id: int, label: str, position: Tuple[float, float, float]) -> None:
        now = time.monotonic()
        if track_id >= 0:
            self._track_id_last_write[track_id] = now
        if self.dedup_position_m > 0 and label:
            self._recent_jobs.append((label, position, now))
        self._recent_jobs = [(l, p, t) for l, p, t in self._recent_jobs if now - t <= self._recent_jobs_max_age]
        self._track_id_last_write = {k: v for k, v in self._track_id_last_write.items() if now - v <= self._recent_jobs_max_age}


class Sam3dJobWriterNode(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__("sam3d_job_writer_node", parameter_overrides=parameter_overrides or [])

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("write_every_n", 1)  # 1 = every message (for testing); increase to throttle
        self.declare_parameter("enable", True)
        self.declare_parameter("enable_pre_sam3d_quality_filter", False)
        # Dedup: avoid enqueueing the same object many times so SAM3D is not run redundantly
        self.declare_parameter("dedup_track_id_sec", 60.0)  # skip same track_id if we wrote a job in last N sec (0 = off)
        self.declare_parameter("dedup_position_m", 0.5)  # skip if same label and position within this many meters (0 = off)

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.write_every_n = self.get_parameter("write_every_n").value
        self.enable = self.get_parameter("enable").value
        self.enable_pre_sam3d_quality_filter = self.get_parameter("enable_pre_sam3d_quality_filter").value
        self.dedup_track_id_sec = self.get_parameter("dedup_track_id_sec").value
        self.dedup_position_m = self.get_parameter("dedup_position_m").value

        self.input_dir = self.queue_dir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self._msg_count = 0
        self._dedup = Sam3dDedupState(self.dedup_track_id_sec, self.dedup_position_m)
        self._filter_cfg = self._load_tracking_filter_config() if self.enable_pre_sam3d_quality_filter else {}
        self._run_stats = {
            "received_total": 0,
            "enqueued_total": 0,
            "skipped_dedup_total": 0,
            "skipped_pre_filter_total": 0,
            "skip_reason_counts": {},
        }
        self._label_counts_all = {}
        self._label_counts_enqueued = {}
        self._label_counts_skipped_pre_filter = {}
        self._skip_reason_log_path = self.queue_dir / "pre_sam3d_filter_log.json"
        self._unique_labels_log_path = self.queue_dir / "unique_labels_log.json"

        self.sub = self.create_subscription(
            CropImgDepthMsg,
            "/usd/CropImgDepth",
            self.callback,
            10,
        )
        self.pub_debug_last_crop = self.create_publisher(Image, "/debug/job_writer/last_crop", 10)
        self.pub_timing = self.create_publisher(PipelineStepTiming, "/pipeline/timings", 10)
        self.pub_job_enqueued = self.create_publisher(Sam3dJobEnqueued, "/sam3d/job_enqueued", 10)
        self._timing_sequence = 0
        self.get_logger().info(
            f"SAM3D job writer: queue_dir={self.queue_dir}, write_every_n={self.write_every_n}, enable={self.enable}, "
            f"dedup_track_id_sec={self.dedup_track_id_sec}, dedup_position_m={self.dedup_position_m}, "
            f"enable_pre_sam3d_quality_filter={self.enable_pre_sam3d_quality_filter}"
        )

    def _should_skip_dedup(self, track_id: int, label: str, position: Tuple[float, float, float]) -> bool:
        return self._dedup.should_skip(track_id, label, position)

    def _record_written(self, track_id: int, label: str, position: Tuple[float, float, float]) -> None:
        self._dedup.record(track_id, label, position)

    def callback(self, msg):
        if not self.enable:
            return
        self._msg_count += 1
        if self._msg_count % self.write_every_n != 0:
            return
        self._run_stats["received_total"] += 1

        track_id = int(msg.track_id)
        label = (msg.label or "").strip()
        if label:
            self._label_counts_all[label] = self._label_counts_all.get(label, 0) + 1

        if self.enable_pre_sam3d_quality_filter:
            should_skip, reason = self._should_skip_pre_sam3d_filter(msg, track_id)
            if should_skip:
                self._run_stats["skipped_pre_filter_total"] += 1
                self._run_stats["skip_reason_counts"][reason] = self._run_stats["skip_reason_counts"].get(reason, 0) + 1
                if label:
                    self._label_counts_skipped_pre_filter[label] = self._label_counts_skipped_pre_filter.get(label, 0) + 1
                self._flush_run_logs()
                self.get_logger().debug(f"Skipping SAM3D enqueue due to pre-filter: reason={reason} track_id={track_id}")
                return

        position = (
            msg.odometry.pose.pose.position.x,
            msg.odometry.pose.pose.position.y,
            msg.odometry.pose.pose.position.z,
        )
        if self._should_skip_dedup(track_id, label, position):
            self._run_stats["skipped_dedup_total"] += 1
            self._run_stats["skip_reason_counts"]["dedup"] = self._run_stats["skip_reason_counts"].get("dedup", 0) + 1
            self._flush_run_logs()
            return

        job_id = f"{track_id}_{msg.header.stamp.sec}_{msg.header.stamp.nanosec}_{uuid.uuid4().hex[:8]}"
        job_path = self.input_dir / job_id

        t_start = time.perf_counter()
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding="bgr8")
            depth_full = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding="16UC1")
            seg_pts = np.array(msg.seg_points, dtype=np.int32).reshape(-1, 2)
            crop_bbox = list(msg.crop_bbox) if len(msg.crop_bbox) >= 4 else None
            meta = {
                "job_id": job_id,
                "track_id": int(msg.track_id),
                "label": msg.label,
                "frame_id": msg.header.frame_id,
                "stamp_sec": msg.header.stamp.sec,
                "stamp_nanosec": msg.header.stamp.nanosec,
                "camera_info": {
                    "K": list(msg.camera_info.k),
                    "width": msg.camera_info.width,
                    "height": msg.camera_info.height,
                },
                "odometry": {
                    "position": [
                        msg.odometry.pose.pose.position.x,
                        msg.odometry.pose.pose.position.y,
                        msg.odometry.pose.pose.position.z,
                    ],
                    "orientation": [
                        msg.odometry.pose.pose.orientation.x,
                        msg.odometry.pose.pose.orientation.y,
                        msg.odometry.pose.pose.orientation.z,
                        msg.odometry.pose.pose.orientation.w,
                    ],
                },
            }
            write_sam3d_job(rgb, depth_full, seg_pts, meta, job_path, crop_bbox=crop_bbox)
            self._record_written(track_id, label, position)
            self._run_stats["enqueued_total"] += 1
            if label:
                self._label_counts_enqueued[label] = self._label_counts_enqueued.get(label, 0) + 1
            self._flush_run_logs()
            self.get_logger().info(f"Wrote SAM3D job: {job_path}")
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            timing_msg = PipelineStepTiming()
            timing_msg.header.stamp = self.get_clock().now().to_msg()
            timing_msg.header.frame_id = "map"
            timing_msg.node_name = "sam3d_job_writer_node"
            timing_msg.step_name = "write_job"
            timing_msg.duration_ms = duration_ms
            timing_msg.sequence_id = self._timing_sequence
            self._timing_sequence += 1
            self.pub_timing.publish(timing_msg)
            enq = Sam3dJobEnqueued()
            enq.header.stamp = self.get_clock().now().to_msg()
            enq.header.frame_id = "map"
            enq.job_id = job_id
            self.pub_job_enqueued.publish(enq)
            self.pub_debug_last_crop.publish(msg.rgb_image)
        except Exception as e:
            self.get_logger().error(f"Failed to write SAM3D job {job_id}: {e}")

    def _load_tracking_filter_config(self) -> dict:
        cfg_path = Path(get_package_share_directory("real2sam3d")) / "config" / "tracking_pre_sam3d_filter.json"
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().warn(f"Failed to load pre-SAM3D filter config {cfg_path}: {e}")
            return {}

    def _should_skip_pre_sam3d_filter(self, msg: CropImgDepthMsg, track_id: int):
        cfg = self._filter_cfg.get("pre_sam3d_filter", {}) if isinstance(self._filter_cfg, dict) else {}
        if not cfg or not bool(cfg.get("enabled", False)):
            return False, ""
        if bool(cfg.get("skip_untracked", True)) and track_id < 0:
            return True, "untracked"

        seg_pts = np.array(msg.seg_points, dtype=np.int32).reshape(-1, 2)
        mask_area = int(seg_pts.shape[0])
        min_mask_area = int(cfg.get("min_mask_area_px", 0))
        use_depth_valid_ratio_gate = bool(cfg.get("use_depth_valid_ratio_gate", True))
        min_depth_valid_ratio = float(cfg.get("min_depth_valid_ratio", 0.0))
        max_edge_contact_ratio = float(cfg.get("max_edge_contact_ratio", 1.0))
        edge_band_px = int(cfg.get("edge_band_px", 0))
        small_object_area_px = int(cfg.get("small_object_area_px", 0))

        if mask_area == 0:
            return True, "empty_mask"

        depth_full = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding="16UC1")
        h, w = depth_full.shape[:2]
        valid = (
            (seg_pts[:, 0] >= 0) & (seg_pts[:, 0] < w)
            & (seg_pts[:, 1] >= 0) & (seg_pts[:, 1] < h)
        )
        seg_valid = seg_pts[valid]
        if seg_valid.size == 0:
            return True, "mask_outside_image"

        depth_vals = depth_full[seg_valid[:, 1], seg_valid[:, 0]]
        depth_valid_ratio = float(np.count_nonzero(depth_vals > 0)) / float(seg_valid.shape[0])

        if edge_band_px > 0:
            on_edge = (
                (seg_valid[:, 0] < edge_band_px)
                | (seg_valid[:, 0] >= (w - edge_band_px))
                | (seg_valid[:, 1] < edge_band_px)
                | (seg_valid[:, 1] >= (h - edge_band_px))
            )
            edge_contact_ratio = float(np.count_nonzero(on_edge)) / float(seg_valid.shape[0])
        else:
            edge_contact_ratio = 0.0

        if use_depth_valid_ratio_gate and mask_area < min_mask_area and depth_valid_ratio < min_depth_valid_ratio:
            return True, "small_and_low_depth"
        if mask_area < small_object_area_px and edge_contact_ratio > max_edge_contact_ratio:
            return True, "small_and_edge"
        return False, ""

    def _flush_run_logs(self):
        try:
            stats = dict(self._run_stats)
            stats["updated_at"] = time.time()
            with open(self._skip_reason_log_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed writing skip-reason log: {e}")
        try:
            payload = {
                "updated_at": time.time(),
                "labels_seen_all": sorted(self._label_counts_all.keys()),
                "labels_enqueued": sorted(self._label_counts_enqueued.keys()),
                "labels_skipped_pre_filter": sorted(self._label_counts_skipped_pre_filter.keys()),
                "label_counts_all": self._label_counts_all,
                "label_counts_enqueued": self._label_counts_enqueued,
                "label_counts_skipped_pre_filter": self._label_counts_skipped_pre_filter,
            }
            with open(self._unique_labels_log_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed writing unique-label log: {e}")


def main():
    rclpy.init()
    node = Sam3dJobWriterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
