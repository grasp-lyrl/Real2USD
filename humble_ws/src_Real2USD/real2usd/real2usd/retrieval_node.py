import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from custom_message.msg import UsdStringIdPCMsg
from custom_message.msg import CropImgDepthMsg
from std_msgs.msg import Header
from cv_bridge import CvBridge
import json, asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R

# from scripts_w2u.usdsearch_cls import USDSearch
from scripts_r2u.clipusdsearch_cls import CLIPUSDSearch
from scripts_r2u.gemini_usd_selector_cls import GeminiUSDSelector
from scripts_r2u.utils import ProjectionUtils

from ipdb import set_trace as st


"""
This node is used to retrieve the USD object from the database.
It is used to retrieve the USD object from the database based on the query image.

Updata the faiss index path to the correct path. based on your directory structure.
See Readme/CLIP_USD_SEARCH_README.md for more details.
"""

class RetrievalNode(Node):
    def __init__(self):
        super().__init__("retrieval_node")

        # Declare and get parameter for Gemini usage
        self.declare_parameter('use_gemini', False)
        self.declare_parameter('faiss_index_path', "/data/FAISS/FAISS")
        self.use_gemini = self.get_parameter('use_gemini').value
        self.faiss_index_path = self.get_parameter('faiss_index_path').value
        self.get_logger().info(f"Using Gemini for comparison: {self.use_gemini}")
        self.get_logger().info(f"Using FAISS index: {self.faiss_index_path}")

        self.bridge = CvBridge()

        # subscribers
        self.usd_sub = self.create_subscription(CropImgDepthMsg, "/usd/CropImgDepth", self.retrieval_callback, 50)

        # publishers
        self.usd_pc_pub = self.create_publisher(UsdStringIdPCMsg, "/usd/StringIdPC", 10)

        # debug publishers
        self.crop_pub = self.create_publisher(Image, "/segment/image_cropped", 10)
        self.seg_pc_pub = self.create_publisher(PointCloud2, "/segment/pointcloud", 10)
        self.usd_search_pub = self.create_publisher(Image, "/usd_search/image", 10)
        self.usd_search_result_pub = self.create_publisher(Image, "/usd_search/img_result", 10)

        self.ground_plane_height_threshold = 0.1  # Points below this height are considered ground

        self.limit = 3

        self.projection = ProjectionUtils(T=np.array([0.285, 0., 0.01]))

        self.usdsearch = CLIPUSDSearch()
        self.usdsearch.load_index(self.faiss_index_path)
        self.gemini_usd_selector = GeminiUSDSelector()

    def cam_info_callback(self, cam_msg):
        cam_info = {}
        cam_info["K"] = cam_msg.k.reshape(3, 3)
        cam_info["R"] = cam_msg.r
        cam_info["P"] = cam_msg.p.reshape(3, 4)
        cam_info["width"] = cam_msg.width
        cam_info["height"] = cam_msg.height
        return cam_info

    def odom_callback(self, odom_msg):
        t = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        eulerXYZ = R.from_quat(q).as_euler("xyz", degrees=True)
        odom_info = {}
        odom_info["t"] = t
        odom_info["q"] = q
        odom_info["eulerXYZ"] = eulerXYZ
        return odom_info

    def retrieval_callback(self, msg):
        """
        msg: CropImgDepthMsg
        """
        # decompose the msg
        imgs_crop = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding="bgr8")
        seg_pts = np.array(msg.seg_points).reshape(-1, 2)
        depth_image = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding="16UC1")
        cam_info = self.cam_info_callback(msg.camera_info)
        odom_info = self.odom_callback(msg.odometry)
        track_ids = msg.track_id
        labels = msg.label
        
        # compute 3d points in world and publish the point cloud
        lidar_world = self.projection.twoDtoThreeD(seg_pts, depth_image, cam_info, odom_info)
        lidar_world = self.filter_ground_plane(lidar_world)

        # publish for debugging
        self.crop_pub.publish(msg.rgb_image)
        segment_pc_msg = point_cloud2.create_cloud_xyz32(header=Header(frame_id="odom"), points=lidar_world)
        segment_pc_msg.header.stamp = msg.header.stamp
        self.seg_pc_pub.publish(segment_pc_msg)

        # Begin retrieval
        image_embedding = self.usdsearch.process_image(imgs_crop)

        # Extract label from the detected object
        detected_label = labels.lower()  # Convert to lowercase for matching

        urls, scores, image_paths, images = asyncio.run(
            self.usdsearch.call_search_post_api("", [image_embedding], limit=self.limit, retrieval_mode="cosine")
        )

        # visualize the cosine similarity, comment out if not needed
        highlight_indices = [self.usdsearch.image_paths.index(p) for p in image_paths if p in self.usdsearch.image_paths]
        self.usdsearch.visualize_cosine_similarity(
            test_embedding=image_embedding,
            highlight_indices=highlight_indices,
            label_mode='folder'
        )

        if urls is not None and images is not None:
            # Filter URLs and images based on label matching
            filtered_urls = []
            filtered_scores = []
            filtered_images = []
            
            for url, score, image in zip(urls, scores, images):
                # Extract label from USD path
                parts = url.split('/')
                """
                This is dependent on your database structure.
                We essentially want to use the file path to obtain a general label of the object.
                Like "chair". File directory could like like "/data/chairs/sofas/angled_sofa.usd"
                See Readme/CLIP_USD_SEARCH_README.md for more details.
                """
                if len(parts) >= 3:
                    usd_label = parts[-4].lower()  # Get fourth-to-last directory name
                    
                    # Check if labels match (allowing for partial matches)
                    if (detected_label in usd_label or usd_label in detected_label):
                        filtered_urls.append(url)
                        filtered_scores.append(score)
                        filtered_images.append(image)
            
            # If no matches found, use original results
            if not filtered_urls:
                return
            else:
                self.get_logger().info(f"Filtered {len(urls) - len(filtered_urls)} results based on label {detected_label}")
            
            # Update variables for Gemini comparison
            urls = filtered_urls
            scores = filtered_scores
            images = filtered_images

            # publish the usd search images and also checks if images are returned
            for image in images:
                image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                self.usd_search_pub.publish(image_msg)

            """
            This is an optional setup where we use Gemini to compare the query image with the reference images.
            Not used in the results of the paper
            """
            if self.use_gemini:
                try:
                    # Prepare images for Gemini comparison
                    query_image = imgs_crop
                    reference_images = images[:self.limit]  # Take first 3 images
                    
                    response = self.gemini_usd_selector.gemini_query(query_image, reference_images)
                    
                    # Process response
                    result = json.loads(response.text)
                    best_match_index = result['best_match_index']
                    confidence = result['confidence']
                    
                    # Use the best matching image's URL
                    url = urls[best_match_index]
                    scores = scores[best_match_index]
                    
                except Exception as e:
                    self.get_logger().error(f"Error in Gemini API call: {str(e)}")
                    # Continue with the first image if Gemini fails
                    url = urls[0]
                    scores = scores[0]
                    best_match_index = 0
                    confidence = None
            else:
                # When Gemini is disabled, use the highest scoring result
                best_match_index = np.argmax(scores)
                url = urls[best_match_index]
                scores = scores[best_match_index]
                confidence = None
        else:
            self.get_logger().info(f"{urls}")
            url = 'None'
            scores = None

            # move on to next object
            return

        # self.get_logger().info(f"label: {labels}, url: {url}, scores: {scores}")
        try:
            self.get_logger().info(f"url: {url}, best match index: {best_match_index}" + 
                                 (f", confidence: {confidence}" if confidence is not None else ""))

            image_msg = self.bridge.cv2_to_imgmsg(images[best_match_index], encoding="bgr8")
            self.usd_search_result_pub.publish(image_msg)
        except:
            self.get_logger().info(f"url: {url}")

        # replace 's3://usdsearch-whatchanged/' with local data path
        # fails when usd search could not find something, returns None
        try:
            url = url.replace("s3://usdsearch-whatchanged/", "/data/")

            # publish the usd path, track id, and point cloud
            msg = UsdStringIdPCMsg()
            msg.header = segment_pc_msg.header
            msg.data_path = url
            msg.id = track_ids
            msg.pc = segment_pc_msg
            self.usd_pc_pub.publish(msg)

        except:
            self.get_logger().info(f"failed to replace {url} with local data path")

    def filter_ground_plane(self, points):
        """
        Filter out points that are below the ground plane height threshold.
        Returns points that are above the ground plane.
        """
        # Simple height-based filtering
        return points[points[:, 2] > self.ground_plane_height_threshold]

def main():
    rclpy.init()
    retrieval_node = RetrievalNode()
    rclpy.spin(retrieval_node)
    retrieval_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
