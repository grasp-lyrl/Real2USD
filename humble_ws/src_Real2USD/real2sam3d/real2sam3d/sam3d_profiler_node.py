"""
SAM3D profiler: infers worker inference time from job enqueue → output.

Subscribes to /sam3d/job_enqueued (when job_writer writes a job) and to
/usd/SlotReady and /usd/Sam3dObjectForSlot (when injector/retrieval see output).
Publishes PipelineStepTiming to /pipeline/timings for node_name="sam3d_worker",
step_name="inference", duration_ms = completion_time - enqueue_time, so the
pipeline profiler strip shows SAM3D latency even though the worker runs outside ROS.

Also subscribes to /pipeline/object_in_buffer (ObjectInBufferMsg). "Done" for full-pipeline = object in buffer.
Uses wall clock: frame_wall_sec from Sam3dJobEnqueued and buffer stamp from ObjectInBufferMsg.

Publishes (all recorded in timing_summary.json):
  - inference: job enqueue → SlotReady (worker only; latency = queue + inference).
  - e2e_sam3d_only_ms: E2E from when SAM3D receives the job to when it finishes (ms per object) [latency].
  - e2e_full_pipeline_ms: E2E from frame (wall) to object in buffer (ms per object) [latency].
  - e2e_frame_ms: E2E from frame (wall) to last object from that frame in buffer (ms per frame).
  - e2e_frame_sam3d_only_ms: E2E from first job enqueue to last object from that frame done (ms per frame; SAM3D only).
  - inference_time_ms: worker-reported inference time per object (from pose.json); true inference, not latency.
  - inference_time_per_frame_ms: worker-reported sum of inference time per frame.

Note: inference and e2e_sam3d_only are LATENCY (enqueue → done), i.e. queue wait + inference. With a sequential
worker, later jobs wait for earlier ones, so mean latency grows as the run progresses. For per-object inference time
(roughly constant), use the min of these metrics or worker-side timing; timing_summary.json also reports
inference_approx_ms = min(inference) for this purpose.

Parameter debug_timing (default false): when true, logs each e2e_sam3d_only and e2e_full_pipeline with job_id and
timestamps so you can spot stale enqueue times or outliers (e.g. run with --ros-args -p debug_timing:=true).
"""

import collections
from typing import Optional

import rclpy
from rclpy.node import Node

from custom_message.msg import (
    ObjectInBufferMsg,
    PipelineStepTiming,
    Sam3dJobEnqueued,
    Sam3dObjectForSlotMsg,
    SlotReadyMsg,
)

MAX_ENQUEUE_ENTRIES = 2000  # cap memory; drop oldest by count
# job_id format: "{track_id}_{stamp.sec}_{stamp.nanosec}_{uuid8}"
MAX_FRAME_COMPLETIONS = 500  # cap frames we track for e2e_frame
MAX_SANE_E2E_FULL_MS = 3600.0 * 1000.0  # 1 hour; skip if e2e full-pipeline exceeds (sanity)
BUFFER_FRAME_ROUND_SEC = 1.0  # group buffer completions by frame_wall_sec rounded to this for e2e_frame_ms


def _stamp_to_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _frame_stamp_sec_from_job_id(job_id: str) -> Optional[float]:
    """Parse frame stamp (sec) from job_id. Returns None if malformed."""
    parts = job_id.split("_")
    if len(parts) < 4:
        return None
    try:
        sec = int(parts[1])
        nanosec = int(parts[2])
        return sec + nanosec * 1e-9
    except (ValueError, IndexError):
        return None


class Sam3dProfilerNode(Node):
    def __init__(self):
        super().__init__("sam3d_profiler_node")
        self.declare_parameter("debug_timing", False)
        self._debug_timing = self.get_parameter("debug_timing").value
        self._enqueue_stamp: dict[str, float] = {}
        self._enqueue_order: collections.deque = collections.deque(maxlen=MAX_ENQUEUE_ENTRIES)
        self._timing_sequence = 0
        # frame_stamp_sec -> list of (enqueue_sec, completion_sec) for e2e per frame SAM3D-only (SlotReady path)
        self._frame_completions: dict = {}
        self._frame_order: collections.deque = collections.deque(maxlen=MAX_FRAME_COMPLETIONS)
        # Wall clock: job_id -> frame_wall_sec (from Sam3dJobEnqueued) for full-pipeline E2E
        self._frame_wall_sec: dict[str, float] = {}
        # Buffer path: frame_id (rounded wall sec) -> list of (frame_wall_sec, buffer_stamp_sec, inference_ms) for e2e_frame_ms and inference_time_per_frame_ms
        self._buffer_frame_completions: dict = {}
        self._buffer_frame_order: collections.deque = collections.deque(maxlen=MAX_FRAME_COMPLETIONS)

        self.create_subscription(
            Sam3dJobEnqueued,
            "/sam3d/job_enqueued",
            self._on_job_enqueued,
            50,
        )
        self.create_subscription(
            ObjectInBufferMsg,
            "/pipeline/object_in_buffer",
            self._on_object_in_buffer,
            50,
        )
        self.create_subscription(
            SlotReadyMsg,
            "/usd/SlotReady",
            self._on_slot_ready,
            50,
        )
        self.create_subscription(
            Sam3dObjectForSlotMsg,
            "/usd/Sam3dObjectForSlot",
            self._on_object_for_slot,
            50,
        )
        self._pub_timing = self.create_publisher(PipelineStepTiming, "/pipeline/timings", 10)
        self.get_logger().info(
            "SAM3D profiler: correlate job_enqueued → SlotReady/ObjectForSlot, publish sam3d_worker/inference to /pipeline/timings"
        )

    def _on_job_enqueued(self, msg: Sam3dJobEnqueued):
        job_id = msg.job_id
        stamp_sec = _stamp_to_sec(msg.header)
        if job_id in self._enqueue_stamp:
            return  # avoid duplicates
        while len(self._enqueue_order) >= MAX_ENQUEUE_ENTRIES and self._enqueue_order:
            old_id = self._enqueue_order.popleft()
            self._enqueue_stamp.pop(old_id, None)
            self._frame_wall_sec.pop(old_id, None)
        self._enqueue_stamp[job_id] = stamp_sec
        self._enqueue_order.append(job_id)
        # Wall clock when pipeline had this frame (for full-pipeline E2E; "done" = object in buffer)
        fw = getattr(msg, "frame_wall_sec", None)
        if fw is not None:
            self._frame_wall_sec[job_id] = float(fw)

    def _publish_timing(self, node_name: str, step_name: str, duration_ms: float, completion_stamp_sec: float):
        msg = PipelineStepTiming()
        msg.header.frame_id = "map"
        msg.header.stamp.sec = int(completion_stamp_sec)
        msg.header.stamp.nanosec = int((completion_stamp_sec - int(completion_stamp_sec)) * 1e9)
        msg.node_name = node_name
        msg.step_name = step_name
        msg.duration_ms = duration_ms
        msg.sequence_id = self._timing_sequence
        self._timing_sequence += 1
        self._pub_timing.publish(msg)

    def _close_frame_e2e(
        self, frame_stamp_sec: float, enq_completion_pairs: list
    ) -> tuple:
        """Returns (e2e_frame_full_ms, e2e_frame_sam3d_only_ms) or (None, None). enq_completion_pairs = [(enqueue_sec, completion_sec), ...]."""
        if not enq_completion_pairs:
            return (None, None)
        completions = [c for _, c in enq_completion_pairs]
        enqueues = [e for e, _ in enq_completion_pairs]
        end_sec = max(completions)
        first_enq_sec = min(enqueues)
        e2e_full_ms = (end_sec - frame_stamp_sec) * 1000.0
        e2e_sam3d_only_ms = (end_sec - first_enq_sec) * 1000.0
        return (
            e2e_full_ms if e2e_full_ms >= 0 else None,
            e2e_sam3d_only_ms if e2e_sam3d_only_ms >= 0 else None,
        )

    def _publish_sam3d_timing(self, job_id: str, completion_stamp_sec: float):
        enq = self._enqueue_stamp.pop(job_id, None)
        if enq is None:
            return
        sam3d_only_ms = (completion_stamp_sec - enq) * 1000.0
        if sam3d_only_ms < 0:
            return
        if self._debug_timing:
            self.get_logger().info(
                f"[e2e_sam3d_only] job_id={job_id} enqueue_sec={enq:.3f} completion_sec={completion_stamp_sec:.3f} duration_ms={sam3d_only_ms:.1f}"
            )
        self._publish_timing("sam3d_worker", "inference", sam3d_only_ms, completion_stamp_sec)
        # E2E from when SAM3D only gets the job (enqueue → completion)
        self._publish_timing("sam3d_worker", "e2e_sam3d_only_ms", sam3d_only_ms, completion_stamp_sec)

        # E2E per frame (SAM3D only): close older frames and emit e2e_frame_sam3d_only_ms (full-pipeline uses buffer path)
        frame_stamp_sec = _frame_stamp_sec_from_job_id(job_id)
        if frame_stamp_sec is not None:
            for old_frame in sorted(self._frame_completions.keys()):
                if old_frame >= frame_stamp_sec:
                    break
                pairs = self._frame_completions.pop(old_frame, None)
                if pairs is not None:
                    end_sec = max(c for _, c in pairs)
                    _, e2e_sam3d_only_ms = self._close_frame_e2e(old_frame, pairs)
                    if e2e_sam3d_only_ms is not None:
                        self._publish_timing("sam3d_worker", "e2e_frame_sam3d_only_ms", e2e_sam3d_only_ms, end_sec)
            self._frame_completions.setdefault(frame_stamp_sec, []).append((enq, completion_stamp_sec))
            self._frame_order.append(frame_stamp_sec)
            while len(self._frame_completions) > MAX_FRAME_COMPLETIONS and self._frame_order:
                old = self._frame_order.popleft()
                self._frame_completions.pop(old, None)

    def _on_object_in_buffer(self, msg: ObjectInBufferMsg):
        """Full-pipeline E2E: wall clock (frame_wall_sec) → object in buffer. n = completed objects in buffer."""
        job_id = (getattr(msg, "job_id", None) or "").strip()
        if not job_id:
            return
        buffer_stamp_sec = _stamp_to_sec(msg.header)
        frame_wall_sec = self._frame_wall_sec.pop(job_id, None)
        if frame_wall_sec is None:
            return
        inference_ms = float(getattr(msg, "inference_ms", 0.0) or 0.0)
        if inference_ms > 0:
            self._publish_timing("sam3d_worker", "inference_time_ms", inference_ms, buffer_stamp_sec)
        full_pipeline_ms = (buffer_stamp_sec - frame_wall_sec) * 1000.0
        if full_pipeline_ms < 0 or full_pipeline_ms > MAX_SANE_E2E_FULL_MS:
            return
        if self._debug_timing:
            self.get_logger().info(
                f"[e2e_full_pipeline] job_id={job_id} frame_wall_sec={frame_wall_sec:.3f} buffer_sec={buffer_stamp_sec:.3f} duration_ms={full_pipeline_ms:.1f}"
            )
        self._publish_timing("sam3d_worker", "e2e_full_pipeline_ms", full_pipeline_ms, buffer_stamp_sec)
        # Per-frame full-pipeline: group by rounded frame_wall_sec; store (fw, bs, inference_ms) for inference_time_per_frame_ms
        frame_id = round(frame_wall_sec / BUFFER_FRAME_ROUND_SEC) * BUFFER_FRAME_ROUND_SEC
        for fid in sorted(self._buffer_frame_completions.keys()):
            if fid >= frame_id:
                break
            triples = self._buffer_frame_completions.pop(fid, None)
            if triples:
                fw_min = min(t[0] for t in triples)
                buf_max = max(t[1] for t in triples)
                e2e_frame_ms = (buf_max - fw_min) * 1000.0
                if 0 <= e2e_frame_ms <= MAX_SANE_E2E_FULL_MS:
                    self._publish_timing("sam3d_worker", "e2e_frame_ms", e2e_frame_ms, buf_max)
                total_inference_ms = sum(t[2] for t in triples)
                if total_inference_ms > 0:
                    self._publish_timing("sam3d_worker", "inference_time_per_frame_ms", total_inference_ms, buf_max)
        self._buffer_frame_completions.setdefault(frame_id, []).append((frame_wall_sec, buffer_stamp_sec, inference_ms))
        self._buffer_frame_order.append(frame_id)
        while len(self._buffer_frame_completions) > MAX_FRAME_COMPLETIONS and self._buffer_frame_order:
            old = self._buffer_frame_order.popleft()
            self._buffer_frame_completions.pop(old, None)

    def _on_slot_ready(self, msg: SlotReadyMsg):
        completion_sec = _stamp_to_sec(msg.header)
        self._publish_sam3d_timing(msg.job_id, completion_sec)

    def _on_object_for_slot(self, msg: Sam3dObjectForSlotMsg):
        completion_sec = _stamp_to_sec(msg.header)
        self._publish_sam3d_timing(msg.job_id, completion_sec)


def main():
    rclpy.init()
    node = Sam3dProfilerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
