"""
Pipeline profiler: subscribes to /pipeline/timings (PipelineStepTiming), aggregates
durations and rates per (node, step), logs periodic summaries, and publishes a
timeline image for RViz (profiler-style: time vs step, color = duration).

When parameter timing_log_dir is set (e.g. to the run directory), writes:
  - timing_events.csv: one row per event (stamp_sec, node_name, step_name, duration_ms)
  - timing_summary.json: per (node, step) cumulative stats over the entire run (count, mean_ms, std_ms, min_ms, max_ms)

Summary and timing_summary.json use cumulative stats (entire experiment), not a sliding window.

For sam3d_worker: reported "inference" and e2e_sam3d_only are latency (enqueue->done), so mean grows with queue
depth when the worker is sequential. timing_summary.json includes inference_approx_ms = min(inference) as an
estimate of per-object inference time.
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

        # Per (node, step): list of (stamp_sec, duration_ms) for timeline + recent rate (pruned by time)
        self._samples: dict = collections.defaultdict(list)
        # Cumulative over entire run: (node, step) -> {count, sum_ms, sum_sq_ms, min_ms, max_ms, last_stamp_sec, rate_count, rate_sum_hz, rate_sum_sq_hz}
        def _new_cumulative():
            return {
                "count": 0, "sum_ms": 0.0, "sum_sq_ms": 0.0, "min_ms": None, "max_ms": None,
                "last_stamp_sec": None, "rate_count": 0, "rate_sum_hz": 0.0, "rate_sum_sq_hz": 0.0,
            }
        self._cumulative: dict = collections.defaultdict(_new_cumulative)
        self._run_start_sec: float = None  # set on first timing (node clock)
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
        d_ms = msg.duration_ms
        if self._run_start_sec is None:
            self._run_start_sec = stamp_sec
        # Update cumulative stats (entire run)
        c = self._cumulative[key]
        c["count"] += 1
        c["sum_ms"] += d_ms
        c["sum_sq_ms"] += d_ms * d_ms
        if c["min_ms"] is None or d_ms < c["min_ms"]:
            c["min_ms"] = d_ms
        if c["max_ms"] is None or d_ms > c["max_ms"]:
            c["max_ms"] = d_ms
        # Instantaneous rate = 1 / (time since last event); for rate mean ± std Hz
        if c["last_stamp_sec"] is not None:
            delta_sec = stamp_sec - c["last_stamp_sec"]
            if delta_sec > 0:
                rate_hz = 1.0 / delta_sec
                c["rate_count"] += 1
                c["rate_sum_hz"] += rate_hz
                c["rate_sum_sq_hz"] += rate_hz * rate_hz
        c["last_stamp_sec"] = stamp_sec
        self._samples[key].append((stamp_sec, d_ms))
        # Prune old samples per key for timeline + recent rate only (keep last N sec)
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

    def _cumulative_mean_std(self, c: dict) -> tuple:
        """Return (mean_ms, std_ms) from cumulative count, sum_ms, sum_sq_ms. std=0 if count<2."""
        n = c["count"]
        if n == 0:
            return (0.0, 0.0)
        mean_ms = c["sum_ms"] / n
        if n < 2:
            return (mean_ms, 0.0)
        # variance = E[X^2] - E[X]^2 = (sum_sq/n) - (sum/n)^2
        var = (c["sum_sq_ms"] / n) - (mean_ms * mean_ms)
        std_ms = (var ** 0.5) if var > 0 else 0.0
        return (mean_ms, std_ms)

    def _cumulative_rate_mean_std_hz(self, c: dict) -> tuple:
        """Return (mean_hz, std_hz) from rate_count, rate_sum_hz, rate_sum_sq_hz. std=0 if rate_count<2."""
        n = c.get("rate_count", 0)
        if n == 0:
            return (0.0, 0.0)
        mean_hz = c["rate_sum_hz"] / n
        if n < 2:
            return (mean_hz, 0.0)
        var = (c["rate_sum_sq_hz"] / n) - (mean_hz * mean_hz)
        std_hz = (var ** 0.5) if var > 0 else 0.0
        return (mean_hz, std_hz)

    def _maybe_log_summary(self):
        now = time.monotonic()
        if now - self._last_summary_time < self.summary_interval:
            return
        self._last_summary_time = now
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed_sec = (now_sec - self._run_start_sec) if self._run_start_sec is not None else 0.0
        lines = [
            "[pipeline profiler] ---",
            "  (cumulative over entire run: avg ± std ms, rate ± std Hz from inter-arrival, n = total count)",
        ]
        frame_total_avgs = []
        for (node, step), c in sorted(self._cumulative.items()):
            if c["count"] == 0:
                continue
            mean_ms, std_ms = self._cumulative_mean_std(c)
            rate_mean_hz, rate_std_hz = self._cumulative_rate_mean_std_hz(c)
            if c.get("rate_count", 0) > 0:
                rate_str = f"rate={rate_mean_hz:.3f} ± {rate_std_hz:.3f} Hz"
            else:
                rate_overall = c["count"] / elapsed_sec if elapsed_sec > 0 else 0.0
                rate_str = f"rate={rate_overall:.3f} Hz"
            lines.append(f"  {node} / {step}: avg={mean_ms:.1f} ± {std_ms:.1f} ms  {rate_str}  n={c['count']}")
            if step == "frame_total":
                frame_total_avgs.append((node, mean_ms))
        if frame_total_avgs:
            lines.append("  --- time per frame (camera):")
            for node, avg_ms in frame_total_avgs:
                fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                lines.append(f"    {node}: {avg_ms:.1f} ms per frame ({fps:.2f} fps)")
        # E2E and inference: cumulative over entire run
        e2e_keys = [
            ("e2e_sam3d_only_ms", "e2e sam3d only (job received -> done), ms per object [latency]"),
            ("e2e_full_pipeline_ms", "e2e full pipeline (frame -> done), ms per object [latency]"),
            ("e2e_frame_ms", "e2e per frame, full pipeline (frame -> last object done), ms"),
            ("e2e_frame_sam3d_only_ms", "e2e per frame, sam3d only (first enqueue -> last object done), ms"),
            ("inference_time_ms", "inference time per object (worker-reported), ms"),
            ("inference_time_per_frame_ms", "inference time per frame (worker-reported sum), ms"),
        ]
        any_e2e = any(self._cumulative[("sam3d_worker", step)].get("count", 0) > 0 for step, _ in e2e_keys)
        if any_e2e:
            lines.append("  --- e2e / inference (cumulative over run, recorded in timing_summary.json):")
            for step, label in e2e_keys:
                c = self._cumulative.get(("sam3d_worker", step), {})
                if c.get("count", 0) == 0:
                    continue
                mean_ms, std_ms = self._cumulative_mean_std(c)
                rate_mean_hz, rate_std_hz = self._cumulative_rate_mean_std_hz(c)
                if c.get("rate_count", 0) > 0:
                    rate_str = f"{rate_mean_hz:.3f} ± {rate_std_hz:.3f} Hz"
                else:
                    rate_overall = c["count"] / elapsed_sec if elapsed_sec > 0 else 0.0
                    rate_str = f"{rate_overall:.3f} Hz"
                lines.append(f"    {label}: {mean_ms:.1f} ± {std_ms:.1f} ms  {rate_str}  n={c['count']}")
            c_inf = self._cumulative.get(("sam3d_worker", "inference"), {})
            if c_inf.get("count", 0) > 0 and c_inf.get("min_ms") is not None:
                _, std_inf = self._cumulative_mean_std(c_inf)
                lines.append(
                    f"    approx per-object inference (min): {c_inf['min_ms']:.1f} ms  std(latencies): {std_inf:.1f} ms  (mean latency grows with queue depth)"
                )
        if len(lines) > 1:
            self.get_logger().info("\n".join(lines))
        if self._timing_log_dir:
            self._write_timing_summary()

    def _write_timing_summary(self):
        """Write full summary (same as printed) to timing_summary.json: steps, time_per_frame_camera, e2e."""
        if not self._timing_log_dir:
            return
        summary_path = os.path.join(self._timing_log_dir, TIMING_SUMMARY_JSON)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed_sec = (now_sec - self._run_start_sec) if self._run_start_sec is not None else 0.0

        steps = []
        for (node_name, step_name), c in sorted(self._cumulative.items()):
            if c["count"] == 0:
                continue
            mean_ms, std_ms = self._cumulative_mean_std(c)
            rate_mean_hz, rate_std_hz = self._cumulative_rate_mean_std_hz(c)
            rate_overall_hz = c["count"] / elapsed_sec if elapsed_sec > 0 else 0.0
            steps.append({
                "node_name": node_name,
                "step_name": step_name,
                "count": int(c["count"]),
                "mean_ms": float(mean_ms),
                "std_ms": float(std_ms),
                "min_ms": float(c["min_ms"]) if c["min_ms"] is not None else None,
                "max_ms": float(c["max_ms"]) if c["max_ms"] is not None else None,
                "rate_hz": float(rate_mean_hz if c.get("rate_count", 0) > 0 else rate_overall_hz),
                "rate_std_hz": float(rate_std_hz) if c.get("rate_count", 0) >= 2 else None,
                "rate_overall_hz": float(rate_overall_hz),
            })

        time_per_frame_camera = []
        for (node_name, step_name), c in sorted(self._cumulative.items()):
            if step_name != "frame_total" or c["count"] == 0:
                continue
            mean_ms, _ = self._cumulative_mean_std(c)
            fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            time_per_frame_camera.append({
                "node_name": node_name,
                "avg_ms": float(mean_ms),
                "fps": float(fps),
            })

        e2e_keys = [
            ("e2e_sam3d_only_ms", "e2e sam3d only (job received -> done), ms per object [latency]"),
            ("e2e_full_pipeline_ms", "e2e full pipeline (frame -> done), ms per object [latency]"),
            ("e2e_frame_ms", "e2e per frame, full pipeline (frame -> last object done), ms"),
            ("e2e_frame_sam3d_only_ms", "e2e per frame, sam3d only (first enqueue -> last object done), ms"),
            ("inference_time_ms", "inference time per object (worker-reported), ms"),
            ("inference_time_per_frame_ms", "inference time per frame (worker-reported sum), ms"),
        ]
        e2e = []
        for step_name, label in e2e_keys:
            c = self._cumulative.get(("sam3d_worker", step_name), {})
            if c.get("count", 0) == 0:
                continue
            mean_ms, std_ms = self._cumulative_mean_std(c)
            rate_mean_hz, rate_std_hz = self._cumulative_rate_mean_std_hz(c)
            rate_overall_hz = c["count"] / elapsed_sec if elapsed_sec > 0 else 0.0
            e2e.append({
                "step_name": step_name,
                "label": label,
                "mean_ms": float(mean_ms),
                "std_ms": float(std_ms),
                "rate_hz": float(rate_mean_hz if c.get("rate_count", 0) > 0 else rate_overall_hz),
                "rate_std_hz": float(rate_std_hz) if c.get("rate_count", 0) >= 2 else None,
                "rate_overall_hz": float(rate_overall_hz),
                "count": int(c["count"]),
            })
        # Per-object inference approx: min(inference) + std of latencies (mean grows with queue depth when sequential).
        c_inf = self._cumulative.get(("sam3d_worker", "inference"), {})
        if c_inf.get("count", 0) > 0 and c_inf.get("min_ms") is not None:
            _, std_lat = self._cumulative_mean_std(c_inf)
            e2e.append({
                "step_name": "inference_approx_ms",
                "label": "approx per-object inference (min enqueue->done; std = std of latencies)",
                "mean_ms": float(c_inf["min_ms"]),
                "std_ms": float(std_lat) if std_lat is not None else None,
                "rate_hz": None,
                "rate_std_hz": None,
                "rate_overall_hz": c_inf["count"] / elapsed_sec if elapsed_sec > 0 else None,
                "count": int(c_inf["count"]),
            })

        # node_durations_ms: one entry per node (aggregate all steps), same shape as real2usd timing report.
        # For sam3d_worker we use only "inference" so n = object count (not sum of inference + e2e_* events).
        NODE_PRIMARY_STEP = {"sam3d_worker": "inference"}
        node_agg = {}
        for (node_name, step_name), c in self._cumulative.items():
            if c["count"] == 0:
                continue
            primary = NODE_PRIMARY_STEP.get(node_name)
            if primary is not None and step_name != primary:
                continue
            if node_name not in node_agg:
                node_agg[node_name] = {"count": 0, "sum_ms": 0.0, "sum_sq_ms": 0.0, "min_ms": None, "max_ms": None}
            a = node_agg[node_name]
            a["count"] += c["count"]
            a["sum_ms"] += c["sum_ms"]
            a["sum_sq_ms"] += c["sum_sq_ms"]
            if c["min_ms"] is not None:
                a["min_ms"] = c["min_ms"] if a["min_ms"] is None else min(a["min_ms"], c["min_ms"])
            if c["max_ms"] is not None:
                a["max_ms"] = c["max_ms"] if a["max_ms"] is None else max(a["max_ms"], c["max_ms"])
        node_durations_ms = {}
        for node_name, a in sorted(node_agg.items()):
            n = a["count"]
            mean_ms = a["sum_ms"] / n if n > 0 else 0.0
            if n >= 2 and a["sum_sq_ms"] > 0:
                var = (a["sum_sq_ms"] / n) - (mean_ms * mean_ms)
                std_ms = (var ** 0.5) if var > 0 else 0.0
            else:
                std_ms = 0.0
            node_durations_ms[node_name] = {
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(std_ms, 3),
                "n": n,
            }
            if a["min_ms"] is not None:
                node_durations_ms[node_name]["min_ms"] = round(float(a["min_ms"]), 3)
            if a["max_ms"] is not None:
                node_durations_ms[node_name]["max_ms"] = round(float(a["max_ms"]), 3)

        out = {
            "run_start_sec": self._run_start_sec,
            "elapsed_sec": elapsed_sec,
            "written_at_sec": time.time(),
            "node_durations_ms": node_durations_ms,
            "steps": steps,
            "time_per_frame_camera": time_per_frame_camera,
            "e2e": e2e,
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
