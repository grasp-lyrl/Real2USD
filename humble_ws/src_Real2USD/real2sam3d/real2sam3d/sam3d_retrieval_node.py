"""
Phase 3 retrieval node: for each slot (job_id + track_id + crop image), query the
SAM3D FAISS index (CLIP embeddings of object views) and publish the best-matching
object for registration. If FAISS is not ready or empty, use the candidate (newly
generated) object. Bridge subscribes to Sam3dObjectForSlot and runs registration.
"""

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from custom_message.msg import SlotReadyMsg, Sam3dObjectForSlotMsg

# CLIPUSDSearch from real2usd (same as index_sam3d_faiss and real2usd retrieval_node)
_script_dir = Path(__file__).resolve().parent
_repo_src = _script_dir.parents[1]  # real2sam3d -> src_Real2USD
_scripts_r2u = _repo_src / "real2usd" / "scripts_r2u"
if _scripts_r2u.is_dir() and str(_scripts_r2u) not in sys.path:
    sys.path.insert(0, str(_scripts_r2u))

TOPIC_OBJECT_FOR_SLOT = "/usd/Sam3dObjectForSlot"


def _get_clip_search(faiss_index_path: str):
    """Lazy load CLIPUSDSearch and load index. Index layout matches index_sam3d_faiss: <path>/faiss/index.faiss."""
    try:
        from clipusdsearch_cls import CLIPUSDSearch
    except ImportError as e:
        raise RuntimeError(
            "CLIPUSDSearch not found. Add real2usd/scripts_r2u to PYTHONPATH. CLIP/FAISS required."
        ) from e
    search = CLIPUSDSearch()
    p = Path(faiss_index_path).resolve()
    if p.suffix in (".faiss", ".pkl") or p.name.endswith("_indexed_jobs.txt"):
        base_dir = p.parent
    else:
        base_dir = p
    base = base_dir / "faiss" / "index"
    if (base.parent / "index.faiss").exists():
        search.load_index(str(base))
    return search


class Sam3dRetrievalNode(Node):
    def __init__(self):
        super().__init__("sam3d_retrieval_node")

        self.declare_parameter("queue_dir", "/data/sam3d_queue")
        self.declare_parameter("faiss_index_path", "/data/sam3d_faiss")

        self.queue_dir = Path(self.get_parameter("queue_dir").value)
        self.faiss_index_path = self.get_parameter("faiss_index_path").value

        self._clip_search = None  # lazy init on first slot

        self.sub = self.create_subscription(
            SlotReadyMsg,
            "/usd/SlotReady",
            self._on_slot_ready,
            10,
        )
        self.pub = self.create_publisher(
            Sam3dObjectForSlotMsg,
            TOPIC_OBJECT_FOR_SLOT,
            10,
        )
        self.get_logger().info(
            f"SAM3D retrieval: subscribe /usd/SlotReady, publish {TOPIC_OBJECT_FOR_SLOT} (faiss_index={self.faiss_index_path})"
        )

    def _ensure_clip_search(self):
        if self._clip_search is None:
            try:
                self._clip_search = _get_clip_search(self.faiss_index_path)
                n = len(self._clip_search.usd_paths) if self._clip_search.index is not None else 0
                self.get_logger().info(f"Loaded FAISS index: {n} object entries")
            except Exception as e:
                self.get_logger().error(f"Failed to load FAISS index: {e}")
                self._clip_search = "failed"

    def _on_slot_ready(self, msg: SlotReadyMsg):
        job_id = msg.job_id
        track_id = msg.track_id
        candidate_data_path = msg.candidate_data_path

        # Load crop image for this slot (output/<job_id>/rgb.png)
        output_dir = self.queue_dir / "output"
        job_dir = output_dir / job_id
        rgb_path = job_dir / "rgb.png"
        if not rgb_path.exists():
            job_dir = self.queue_dir / "input_processed" / job_id
            rgb_path = job_dir / "rgb.png"
        if not rgb_path.exists():
            self.get_logger().warn(
                f"[retrieval] job_id={job_id} track_id={track_id}: no rgb.png; using candidate (not FAISS)"
            )
            self._publish_object_for_slot(msg.header, job_id, track_id, candidate_data_path)
            return

        self._ensure_clip_search()
        if self._clip_search == "failed":
            self.get_logger().warn(
                f"[retrieval] job_id={job_id}: FAISS load failed; using candidate"
            )
            self._publish_object_for_slot(msg.header, job_id, track_id, candidate_data_path)
            return

        # Query FAISS with crop image (CLIP embedding)
        try:
            image = cv2.imread(str(rgb_path))
            if image is None:
                self.get_logger().warn(
                    f"[retrieval] job_id={job_id}: could not read rgb.png; using candidate"
                )
                self._publish_object_for_slot(msg.header, job_id, track_id, candidate_data_path)
                return
            embedding = self._clip_search.process_image(image)
            if self._clip_search.index is None or len(self._clip_search.usd_paths) == 0:
                best_data_path = candidate_data_path
                self.get_logger().info(
                    f"[retrieval] job_id={job_id}: FAISS index empty ({len(self._clip_search.usd_paths) if self._clip_search != 'failed' else 0} entries); using candidate"
                )
            else:
                urls, scores, _, _ = asyncio.run(
                    self._clip_search.call_search_post_api("", [embedding], limit=1, retrieval_mode="cosine")
                )
                if urls and len(urls) > 0:
                    best_data_path = urls[0]
                    score = float(scores[0]) if scores else 0.0
                    same_as_candidate = (
                        Path(best_data_path).resolve() == Path(candidate_data_path).resolve()
                    )
                    if same_as_candidate:
                        self.get_logger().info(
                            f"[retrieval] job_id={job_id} track_id={track_id}: FAISS best match is same as candidate (score={score:.3f}) — index may only have this run's objects"
                        )
                    else:
                        self.get_logger().info(
                            f"[retrieval] job_id={job_id}: FAISS best match {Path(best_data_path).name} (score={score:.3f}) diff from candidate"
                        )
                else:
                    best_data_path = candidate_data_path
                    self.get_logger().info(
                        f"[retrieval] job_id={job_id}: FAISS returned no urls; using candidate"
                    )
        except Exception as e:
            self.get_logger().warn(
                f"[retrieval] job_id={job_id}: exception {e}; using candidate"
            )
            best_data_path = candidate_data_path

        self._publish_object_for_slot(msg.header, job_id, track_id, best_data_path)

    def _publish_object_for_slot(self, header: Header, job_id: str, track_id: int, data_path: str):
        out = Sam3dObjectForSlotMsg()
        out.header = header
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "odom"
        out.job_id = job_id
        out.id = track_id
        out.data_path = data_path
        out.pose = Pose()
        out.pose.position.x = 0.0
        out.pose.position.y = 0.0
        out.pose.position.z = 0.0
        out.pose.orientation.x = 0.0
        out.pose.orientation.y = 0.0
        out.pose.orientation.z = 0.0
        out.pose.orientation.w = 1.0
        self.pub.publish(out)
        self.get_logger().info(f"Published object for slot: job_id={job_id} track_id={track_id} data_path={data_path}")


def main():
    rclpy.init()
    node = Sam3dRetrievalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
