"""
Pipeline profiler: subscribes to /pipeline/timings (PipelineStepTiming), aggregates
durations and rates per (node, step), logs periodic summaries, and publishes a
timeline image for RViz (profiler-style: time vs step, color = duration).
"""

import collections
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


class PipelineProfilerNode(Node):
    def __init__(self):
        super().__init__("pipeline_profiler_node")
        self.declare_parameter("summary_interval_sec", 10.0)
        self.declare_parameter("timeline_topic", "/pipeline/profile_timeline")
        self.summary_interval = self.get_parameter("summary_interval_sec").value
        self.timeline_topic_name = self.get_parameter("timeline_topic").value

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


def main():
    rclpy.init()
    node = PipelineProfilerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
