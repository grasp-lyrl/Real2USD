"""
SAM3D profiler: infers worker inference time from job enqueue → output.

Subscribes to /sam3d/job_enqueued (when job_writer writes a job) and to
/usd/SlotReady and /usd/Sam3dObjectForSlot (when injector/retrieval see output).
Publishes PipelineStepTiming to /pipeline/timings for node_name="sam3d_worker",
step_name="inference", duration_ms = completion_time - enqueue_time, so the
pipeline profiler strip shows SAM3D latency even though the worker runs outside ROS.
"""

import collections

import rclpy
from rclpy.node import Node

from custom_message.msg import (
    PipelineStepTiming,
    Sam3dJobEnqueued,
    Sam3dObjectForSlotMsg,
    SlotReadyMsg,
)

MAX_ENQUEUE_ENTRIES = 2000  # cap memory; drop oldest by count


def _stamp_to_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Sam3dProfilerNode(Node):
    def __init__(self):
        super().__init__("sam3d_profiler_node")
        self._enqueue_stamp: dict[str, float] = {}
        self._enqueue_order: collections.deque = collections.deque(maxlen=MAX_ENQUEUE_ENTRIES)
        self._timing_sequence = 0

        self.create_subscription(
            Sam3dJobEnqueued,
            "/sam3d/job_enqueued",
            self._on_job_enqueued,
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
        self._enqueue_stamp[job_id] = stamp_sec
        self._enqueue_order.append(job_id)

    def _publish_sam3d_timing(self, job_id: str, completion_stamp_sec: float):
        enq = self._enqueue_stamp.pop(job_id, None)
        if enq is None:
            return
        duration_ms = (completion_stamp_sec - enq) * 1000.0
        if duration_ms < 0:
            return
        msg = PipelineStepTiming()
        msg.header.frame_id = "map"
        msg.header.stamp.sec = int(completion_stamp_sec)
        msg.header.stamp.nanosec = int((completion_stamp_sec - int(completion_stamp_sec)) * 1e9)
        msg.node_name = "sam3d_worker"
        msg.step_name = "inference"
        msg.duration_ms = duration_ms
        msg.sequence_id = self._timing_sequence
        self._timing_sequence += 1
        self._pub_timing.publish(msg)

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
