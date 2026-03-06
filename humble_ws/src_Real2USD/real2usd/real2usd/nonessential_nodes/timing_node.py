"""
Timing node: subscribes to pipeline output topics and optional /timing/<node> topics
to report frequency (mean ± std) and per-node processing time (mean ± std),
and end-to-end latency from CropImgDepth to StringIdPose.

How it works:
- Stats are driven by message flow. Each time a message is received on a pipeline
  topic, we record the receive time → used to compute topic frequency (period / Hz).
- Each node publishes its own processing time (Float64, seconds) on /timing/<node>
  when it finishes a callback → we record those for mean ± std per node.
- End-to-end (per object): latency = receive_time - header.stamp for each StringIdPose.
- Per-frame: lidar_cam_node sets one stamp per frame for all CropImgDepth; we close a frame
  when we see a newer stamp and record frame_latency = max(recv_times for that stamp) - stamp.
- Every report_interval seconds we compute mean/std and log; we also write the same
  stats to a JSON file so you can average over multiple runs.

How to end a test:
- Stop the timing node (Ctrl+C in the terminal where it runs). The final report
  is written to JSON in the report_dir (see parameter) on shutdown. Reports are
  also written every report_interval seconds, so you keep the latest snapshot.
"""

import rclpy
from rclpy.node import Node
from collections import deque
import numpy as np
import time
import os
import json
from datetime import datetime

from std_msgs.msg import Float64
from custom_message.msg import (
    CropImgDepthMsg,
    UsdStringIdPCMsg,
    UsdStringIdSrcTargMsg,
    UsdStringIdPoseMsg,
    UsdBufferPoseMsg,
)


def stamp_to_float(header_stamp):
    """Convert builtin_interfaces/Time to seconds since epoch."""
    return float(header_stamp.sec) + 1e-9 * float(header_stamp.nanosec)


class TimingNode(Node):
    def __init__(self):
        super().__init__("timing_node")

        self.declare_parameter("report_dir", ".")
        self.declare_parameter("report_interval", 5.0)
        self.report_dir = self.get_parameter("report_dir").value
        self.report_interval = self.get_parameter("report_interval").value
        os.makedirs(self.report_dir, exist_ok=True)

        # One JSON file per run; run_id set on first report
        self.run_start_time = None
        self.run_id = None
        self.report_path = None

        self.max_samples = 500
        # Topic -> deque of receive timestamps (wall time)
        self.recv_times = {
            "/usd/CropImgDepth": deque(maxlen=self.max_samples),
            "/usd/StringIdPC": deque(maxlen=self.max_samples),
            "/usd/StringIdSrcTarg": deque(maxlen=self.max_samples),
            "/usd/StringIdPose": deque(maxlen=self.max_samples),
            "/usd/SimUsdPoseBuffer": deque(maxlen=self.max_samples),
        }
        # /timing/<node> -> deque of durations (seconds)
        self.durations = {
            "lidar_cam_node": deque(maxlen=self.max_samples),
            "retrieval_node": deque(maxlen=self.max_samples),
            "registration_node": deque(maxlen=self.max_samples),
            "usd_buffer_node": deque(maxlen=self.max_samples),
        }
        # End-to-end: for each StringIdPose we have header.stamp from frame; latency = recv_time - stamp
        self.e2e_latencies = deque(maxlen=self.max_samples)
        # Per-frame: stamp -> list of recv_times; when we see a newer stamp, close older frames
        self.frame_recv_times = {}  # stamp_float -> [recv_time, ...]
        self.frame_latencies = deque(maxlen=self.max_samples)  # time from frame start to last object received
        # Order check: detect if StringIdPose ever arrives with stamp older than the previous message
        self.last_string_id_pose_stamp = None
        self.string_id_pose_total_count = 0
        self.string_id_pose_out_of_order_count = 0

        # Subscriptions to pipeline outputs (for frequency)
        self.create_subscription(
            CropImgDepthMsg, "/usd/CropImgDepth",
            lambda msg: self._record("/usd/CropImgDepth"), 10
        )
        self.create_subscription(
            UsdStringIdPCMsg, "/usd/StringIdPC",
            self._cb_string_id_pc, 10
        )
        self.create_subscription(
            UsdStringIdSrcTargMsg, "/usd/StringIdSrcTarg",
            lambda msg: self._record("/usd/StringIdSrcTarg"), 10
        )
        self.create_subscription(
            UsdStringIdPoseMsg, "/usd/StringIdPose",
            self._cb_string_id_pose, 10
        )
        self.create_subscription(
            UsdBufferPoseMsg, "/usd/SimUsdPoseBuffer",
            lambda msg: self._record("/usd/SimUsdPoseBuffer"), 10
        )

        # Subscriptions to per-node timing (Float64 duration in seconds)
        for name in self.durations:
            self.create_subscription(
                Float64, f"/timing/{name}",
                lambda msg, n=name: self._cb_duration(n, msg.data), 10
            )

        self.report_timer = self.create_timer(self.report_interval, self.report_callback)

    def _record(self, topic: str):
        t = time.time()
        self.recv_times[topic].append(t)

    def _cb_string_id_pc(self, msg):
        self._record("/usd/StringIdPC")

    def _cb_string_id_pose(self, msg):
        t = time.time()
        self.recv_times["/usd/StringIdPose"].append(t)
        try:
            stamp = msg.header.stamp
            stamp_s = stamp_to_float(stamp)
            # Order check: out-of-order if this stamp is older than the previous message's stamp
            self.string_id_pose_total_count += 1
            if self.last_string_id_pose_stamp is not None and stamp_s < self.last_string_id_pose_stamp:
                self.string_id_pose_out_of_order_count += 1
            self.last_string_id_pose_stamp = stamp_s
            # Per-object e2e
            self.e2e_latencies.append(t - stamp_s)
            # Per-frame: collect recv times by stamp; close older frames when we see a newer stamp
            if stamp_s not in self.frame_recv_times:
                self.frame_recv_times[stamp_s] = []
            self.frame_recv_times[stamp_s].append(t)
            # Close frames with stamp < this stamp (we've moved to a later frame)
            for old_stamp in list(self.frame_recv_times.keys()):
                if old_stamp < stamp_s:
                    recv_times = self.frame_recv_times.pop(old_stamp)
                    frame_latency_s = max(recv_times) - old_stamp
                    self.frame_latencies.append(frame_latency_s)
            # Limit open frames to avoid memory growth
            if len(self.frame_recv_times) > 50:
                for old_stamp in sorted(self.frame_recv_times.keys())[:-25]:
                    recv_times = self.frame_recv_times.pop(old_stamp, None)
                    if recv_times is not None:
                        frame_latency_s = max(recv_times) - old_stamp
                        self.frame_latencies.append(frame_latency_s)
        except Exception:
            pass

    def _cb_duration(self, node_name: str, duration: float):
        if node_name in self.durations:
            self.durations[node_name].append(duration)

    def _freq_stats(self, times_deque):
        """Return (mean_period_s, std_period_s, mean_hz, std_hz, effective_hz) or None.
        mean_hz = mean(1/period) — high when messages come in bursts.
        effective_hz = 1/mean(period) — overall message rate (messages per second).
        """
        if len(times_deque) < 2:
            return None
        times = np.array(times_deque, dtype=float)
        periods = np.diff(times)
        mean_p = float(np.mean(periods))
        std_p = float(np.std(periods)) if len(periods) > 1 else 0.0
        # mean_hz = mean(1/period) — dominated by short intervals (bursts)
        hz = 1.0 / np.maximum(periods, 1e-6)
        mean_hz = float(np.mean(hz))
        std_hz = float(np.std(hz)) if len(hz) > 1 else 0.0
        # effective_hz = 1/mean(period) — overall rate
        effective_hz = 1.0 / mean_p if mean_p > 0 else 0.0
        return mean_p, std_p, mean_hz, std_hz, effective_hz

    def _duration_stats(self, dur_deque):
        """Return (mean_s, std_s) or None."""
        if len(dur_deque) < 1:
            return None
        d = np.array(dur_deque, dtype=float)
        return float(np.mean(d)), float(np.std(d)) if len(d) > 1 else 0.0

    def _build_report_dict(self):
        """Build a dict suitable for JSON and for later averaging over runs."""
        now = datetime.utcnow()
        if self.run_start_time is None:
            self.run_start_time = now
            self.run_id = self.run_start_time.strftime("%Y%m%d_%H%M%S")
            self.report_path = os.path.join(
                self.report_dir, f"timing_run_{self.run_id}.json"
            )

        topic_frequencies = {}
        for topic in self.recv_times:
            stats = self._freq_stats(self.recv_times[topic])
            n = len(self.recv_times[topic])
            if stats is not None:
                mean_p, std_p, mean_hz, std_hz, effective_hz = stats
                topic_frequencies[topic] = {
                    "mean_period_ms": round(mean_p * 1000, 3),
                    "std_period_ms": round(std_p * 1000, 3),
                    "mean_hz": round(mean_hz, 3),
                    "std_hz": round(std_hz, 3),
                    "effective_hz": round(effective_hz, 3),
                    "n": n,
                }
            else:
                topic_frequencies[topic] = {"n": n, "mean_period_ms": None, "std_period_ms": None, "mean_hz": None, "std_hz": None, "effective_hz": None}
        # Add "how long it takes" (latency) per topic where we have stamp-based duration (StringIdPose = e2e)
        if "/usd/StringIdPose" in topic_frequencies and len(self.e2e_latencies) >= 1:
            arr_ms = np.array(self.e2e_latencies, dtype=float) * 1000
            topic_frequencies["/usd/StringIdPose"]["mean_latency_ms"] = round(float(np.mean(arr_ms)), 3)
            topic_frequencies["/usd/StringIdPose"]["std_latency_ms"] = (
                round(float(np.std(arr_ms)), 3) if len(arr_ms) > 1 else 0.0
            )
            topic_frequencies["/usd/StringIdPose"]["n_latency"] = len(arr_ms)

        node_durations_ms = {}
        for node_name, dur_deque in self.durations.items():
            stats = self._duration_stats(dur_deque)
            n = len(dur_deque)
            if stats is not None:
                mean_s, std_s = stats
                node_durations_ms[node_name] = {
                    "mean_ms": round(mean_s * 1000, 3),
                    "std_ms": round(std_s * 1000, 3),
                    "n": n,
                }
            else:
                node_durations_ms[node_name] = {"n": n, "mean_ms": None, "std_ms": None}

        e2e = None
        if len(self.e2e_latencies) >= 1:
            arr_ms = np.array(self.e2e_latencies, dtype=float) * 1000
            mean_ms = float(np.mean(arr_ms))
            hz_per_sample = 1000.0 / np.maximum(arr_ms, 1e-6)
            mean_hz = float(np.mean(hz_per_sample))
            std_hz = float(np.std(hz_per_sample)) if len(hz_per_sample) > 1 else 0.0
            e2e = {
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(float(np.std(arr_ms)), 3) if len(arr_ms) > 1 else 0.0,
                "effective_hz": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
                "mean_hz": round(mean_hz, 3),
                "std_hz": round(std_hz, 3),
                "n": len(arr_ms),
            }

        frame_latency_ms = None
        if len(self.frame_latencies) >= 1:
            arr_ms = np.array(self.frame_latencies, dtype=float) * 1000
            mean_ms = float(np.mean(arr_ms))
            hz_per_sample = 1000.0 / np.maximum(arr_ms, 1e-6)
            mean_hz = float(np.mean(hz_per_sample))
            std_hz = float(np.std(hz_per_sample)) if len(hz_per_sample) > 1 else 0.0
            frame_latency_ms = {
                "mean_ms": round(mean_ms, 3),
                "std_ms": round(float(np.std(arr_ms)), 3) if len(arr_ms) > 1 else 0.0,
                "effective_hz": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
                "mean_hz": round(mean_hz, 3),
                "std_hz": round(std_hz, 3),
                "n": len(arr_ms),
            }

        string_id_pose_order = {
            "total_received": self.string_id_pose_total_count,
            "out_of_order_count": self.string_id_pose_out_of_order_count,
        }

        # "How long it takes" (duration/latency in ms) in one place; period = time between messages
        durations_ms = {
            "per_object_e2e": e2e,
            "per_frame": frame_latency_ms,
            "per_node": node_durations_ms,
        }
        if "/usd/StringIdPose" in topic_frequencies and "mean_latency_ms" in topic_frequencies["/usd/StringIdPose"]:
            durations_ms["per_topic_StringIdPose"] = {
                "mean_ms": topic_frequencies["/usd/StringIdPose"]["mean_latency_ms"],
                "std_ms": topic_frequencies["/usd/StringIdPose"]["std_latency_ms"],
                "n": topic_frequencies["/usd/StringIdPose"]["n_latency"],
            }

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "start_time_iso": self.run_start_time.isoformat() + "Z",
            "last_update_time_iso": now.isoformat() + "Z",
            "durations_ms": durations_ms,
            "topic_frequencies": topic_frequencies,
            "node_durations_ms": node_durations_ms,
            "e2e_latency_ms": e2e,
            "frame_latency_ms": frame_latency_ms,
            "string_id_pose_order_check": string_id_pose_order,
            "_interpretation": (
                "durations_ms = how long it takes (per object, per frame, per node, per topic where available). "
                "topic_frequencies[*].mean_latency_ms = how long from message start (stamp) to receive, when available. "
                "topic_frequencies[*].mean_period_ms = time between consecutive messages (throughput), not duration."
            ),
        }

    def write_timing_report(self):
        """Write current stats to JSON file (one file per run, overwritten each report)."""
        d = self._build_report_dict()
        if self.report_path is None:
            return  # no run_id yet (should not happen after _build_report_dict)
        try:
            with open(self.report_path, "w") as f:
                json.dump(d, f, indent=2)
            self.get_logger().info(f"Timing report written to {self.report_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to write timing report: {e}")

    def report_callback(self):
        self.get_logger().info("========== Timing report ==========")
        self.get_logger().info("--- How long it takes (duration / latency ms) ---")

        # End-to-end latency (per object: stamp → StringIdPose received)
        if len(self.e2e_latencies) >= 1:
            e2e_arr = np.array(self.e2e_latencies, dtype=float) * 1000
            mean_e2e_ms = float(np.mean(e2e_arr))
            std_e2e_ms = float(np.std(e2e_arr)) if len(e2e_arr) > 1 else 0.0
            self.get_logger().info(
                f"  per_object_e2e (CropImgDepth stamp -> StringIdPose received): {mean_e2e_ms:.1f} ± {std_e2e_ms:.1f} ms  n={len(e2e_arr)}"
            )
        else:
            self.get_logger().info("  per_object_e2e: (no samples)")

        # Per-frame latency
        if len(self.frame_latencies) >= 1:
            arr = np.array(self.frame_latencies, dtype=float) * 1000
            self.get_logger().info(
                f"  per_frame (frame stamp -> last pose for that frame): {float(np.mean(arr)):.1f} ± {float(np.std(arr)) if len(arr) > 1 else 0:.1f} ms  n={len(arr)}"
            )
        else:
            self.get_logger().info("  per_frame: (no samples)")

        # Per-node processing time
        for node_name, dur_deque in self.durations.items():
            stats = self._duration_stats(dur_deque)
            n = len(dur_deque)
            if stats is not None:
                mean_s, std_s = stats
                self.get_logger().info(
                    f"  {node_name}: {mean_s*1000:.1f} ± {std_s*1000:.1f} ms  n={n}"
                )
            else:
                self.get_logger().info(f"  {node_name}: (no samples)")

        self.get_logger().info("--- Topic throughput (time between consecutive messages) ---")
        for topic in self.recv_times:
            stats = self._freq_stats(self.recv_times[topic])
            n = len(self.recv_times[topic])
            if stats is not None:
                mean_p, std_p, mean_hz, std_hz, effective_hz = stats
                line = f"  {topic} (n={n}): period {mean_p*1000:.1f} ± {std_p*1000:.1f} ms, effective_hz {effective_hz:.2f}"
                if topic == "/usd/StringIdPose" and len(self.e2e_latencies) >= 1:
                    arr_ms = np.array(self.e2e_latencies, dtype=float) * 1000
                    line += f"  | latency (how long) {float(np.mean(arr_ms)):.1f} ± {float(np.std(arr_ms)) if len(arr_ms) > 1 else 0:.1f} ms"
                self.get_logger().info(line)
            else:
                self.get_logger().info(f"  {topic} (n={n}): (need ≥2 samples)")

        # StringIdPose order check (stamp older than previous message = out of order)
        self.get_logger().info(
            f"--- StringIdPose order: {self.string_id_pose_total_count} total, "
            f"{self.string_id_pose_out_of_order_count} out-of-order (stamp < previous) ---"
        )

        self.get_logger().info("====================================")
        self.write_timing_report()


def main():
    rclpy.init()
    node = TimingNode()
    try:
        rclpy.spin(node)
    finally:
        node.write_timing_report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
