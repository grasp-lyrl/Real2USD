"""
Pipeline profiler: subscribes to /pipeline/timings (PipelineStepTiming), aggregates
durations and rates per (node, step), logs periodic summaries, and publishes a
timeline image for RViz (profiler-style: time vs step, color = duration).

When parameter timing_log_dir is set (e.g. to the run directory), writes:
  - timing_events.csv: one row per event (stamp_sec, node_name, step_name, duration_ms)
  - timing_summary.json: per (node, step) stats (count, mean_ms, std_ms, min_ms, max_ms)
  so inference times and other step latencies can be reported across pipeline variations.
"""

import collections
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from custom_message.msg import PipelineStepTiming


# Timeline image size and history length
TIMELINE_HEIGHT = 120  # pixels per row (one row per step)
TIMELINE_WIDTH = 640   # pixels (time axis)
MAX_HISTORY_SEC = 30.0  # keep last N seconds for timeline
MAX_SAMPLES = 2000     # max (step, duration) samples to keep

TIMING_EVENTS_CSV = "timing_events.csv"
TIMING_SUMMARY_JSON = "timing_summary.json"


class PipelineProfilerNode(Node):
    def __init__(self):
        super().__init__("pipeline_profiler_node")
        self.declare_parameter("summary_interval_sec", 10.0)
        self.declare_parameter("timeline_topic", "/pipeline/profile_timeline")
        self.declare_parameter("timing_log_dir", "")
        self.summary_interval = self.get_parameter("summary_interval_sec").value
        self.timeline_topic_name = self.get_parameter("timeline_topic").value
        self._timing_log_dir = (self.get_parameter("timing_log_dir").value or "").strip()

        self.sub_timing = self.create_subscription(
            PipelineStepTiming,
            "/pipeline/timings",
            self._on_timing,
            50,
        )
        self.pub_timeline = self.create_publisher(Image, self.timeline_topic_name, 10)
        self.bridge = CvBridge()

        # Per (node, step): list of (stamp_sec, duration_ms)
        self._samples: dict = collections.defaultdict(list)
        # Ordered list of (stamp_sec, node_name, step_name, duration_ms) for timeline
        self._timeline: list = []
        self._last_summary_time = time.monotonic()
        self._step_to_row: dict = {}  # (node, step) -> row index
        self._row_to_step: list = []  # row index -> (node, step) label

        # Timing log: CSV file handle and path for summary
        self._timing_csv_path = None
        self._timing_csv_file = None
        self._timing_csv_writer = None
        if self._timing_log_dir:
            Path(self._timing_log_dir).mkdir(parents=True, exist_ok=True)
            self._timing_csv_path = os.path.join(self._timing_log_dir, TIMING_EVENTS_CSV)
            try:
                write_header = not os.path.isfile(self._timing_csv_path)
                self._timing_csv_file = open(self._timing_csv_path, "a", newline="")
                self._timing_csv_writer = csv.writer(self._timing_csv_file)
                if write_header:
                    self._timing_csv_writer.writerow(["stamp_sec", "node_name", "step_name", "duration_ms"])
                    self._timing_csv_file.flush()
                self.get_logger().info(f"Logging pipeline timings to {self._timing_csv_path}")
            except OSError as e:
                self.get_logger().warn(f"Could not open timing log {self._timing_csv_path}: {e}")
                self._timing_csv_file = None
                self._timing_csv_writer = None

    def _on_timing(self, msg: PipelineStepTiming):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        key = (msg.node_name, msg.step_name)
        self._samples[key].append((stamp_sec, msg.duration_ms))
        # Prune old samples per key using node clock (matches message stamps; works with use_sim_time when playing bags)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        cutoff = now_sec - MAX_HISTORY_SEC
        self._samples[key] = [(s, d) for s, d in self._samples[key] if s >= cutoff][-500:]

        self._timeline.append((stamp_sec, msg.node_name, msg.step_name, msg.duration_ms))
        if len(self._timeline) > MAX_SAMPLES:
            self._timeline = self._timeline[-MAX_SAMPLES:]
        self._timeline = [x for x in self._timeline if x[0] >= cutoff]

        if self._timing_csv_writer is not None:
            try:
                self._timing_csv_writer.writerow([stamp_sec, msg.node_name, msg.step_name, msg.duration_ms])
                if self._timing_csv_file:
                    self._timing_csv_file.flush()
            except OSError:
                pass

        self._update_step_rows()
        self._publish_timeline_image()
        self._maybe_log_summary()

    def _update_step_rows(self):
        keys = sorted(set((n, s) for _, n, s, _ in self._timeline))
        if keys != self._row_to_step:
            self._row_to_step = keys
            self._step_to_row = {k: i for i, k in enumerate(keys)}

    def _publish_timeline_image(self):
        if not self._timeline or not self._row_to_step:
            return
        n_rows = len(self._row_to_step)
        if n_rows == 0:
            return
        n_rows = min(n_rows, 32)  # cap for image height
        img = np.zeros((TIMELINE_HEIGHT * n_rows, TIMELINE_WIDTH, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        t_min = min(t for t, _, _, _ in self._timeline)
        t_max = max(t for t, _, _, _ in self._timeline) or (t_min + 1.0)
        t_range = max(t_max - t_min, 0.1)
        d_max = max(d for _, _, _, d in self._timeline) or 1.0
        step_rows = self._row_to_step[:n_rows]
        for (t, node, step, d) in self._timeline:
            if (node, step) not in self._step_to_row:
                continue
            row = self._step_to_row[(node, step)]
            if row >= n_rows:
                continue
            x = int((t - t_min) / t_range * (TIMELINE_WIDTH - 1))
            w = max(1, int(d / d_max * (TIMELINE_WIDTH * 0.3)))
            x0 = min(x, TIMELINE_WIDTH - 1)
            x1 = min(x + w, TIMELINE_WIDTH)
            y0 = row * TIMELINE_HEIGHT
            y1 = (row + 1) * TIMELINE_HEIGHT
            # Color: green (fast) -> red (slow) in BGR
            ratio = min(1.0, d / max(d_max, 1e-6))
            b = int(255 * (1 - ratio))
            r = int(255 * ratio)
            g = int(255 * (1 - abs(ratio - 0.5) * 2))
            img[y0:y1, x0:x1] = (b, g, r)
        img_bgr = img
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        self.pub_timeline.publish(msg)

    def _maybe_log_summary(self):
        now = time.monotonic()
        if now - self._last_summary_time < self.summary_interval:
            return
        self._last_summary_time = now
        lines = ["[pipeline profiler] ---"]
        for (node, step), samples in sorted(self._samples.items()):
            if not samples:
                continue
            recent = samples[-100:]
            avg_ms = sum(d for _, d in recent) / len(recent)
            rate = len(recent) / self.summary_interval if self.summary_interval > 0 else 0
            lines.append(f"  {node} / {step}: avg={avg_ms:.1f} ms  rate={rate:.1f} Hz  n={len(samples)}")
        if len(lines) > 1:
            self.get_logger().info("\n".join(lines))
        if self._timing_log_dir:
            self._write_timing_summary()

    def _write_timing_summary(self):
        """Write per (node, step) stats to timing_summary.json in timing_log_dir."""
        if not self._timing_log_dir:
            return
        summary_path = os.path.join(self._timing_log_dir, TIMING_SUMMARY_JSON)
        steps = []
        for (node_name, step_name), samples in sorted(self._samples.items()):
            if not samples:
                continue
            d_list = [d for _, d in samples]
            arr = np.array(d_list, dtype=np.float64)
            steps.append({
                "node_name": node_name,
                "step_name": step_name,
                "count": int(len(arr)),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)) if len(arr) > 1 else 0.0,
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            })
        out = {
            "steps": steps,
            "written_at_sec": time.time(),
        }
        try:
            with open(summary_path, "w") as f:
                json.dump(out, f, indent=2)
        except OSError as e:
            self.get_logger().warn(f"Could not write timing summary {summary_path}: {e}")

    def close_timing_log(self):
        """Close CSV file and write final summary (call before shutdown)."""
        if self._timing_csv_file is not None:
            try:
                self._timing_csv_file.close()
            except OSError:
                pass
            self._timing_csv_file = None
            self._timing_csv_writer = None
        if self._timing_log_dir:
            self._write_timing_summary()


def main():
    rclpy.init()
    node = PipelineProfilerNode()
    try:
        rclpy.spin(node)
    finally:
        node.close_timing_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
