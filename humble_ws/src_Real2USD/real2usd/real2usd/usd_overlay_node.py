import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from custom_message.msg import UsdBufferPoseMsg
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, Gf
from scripts_w2u.utils import ProjectionUtils
import json
import argparse

class USDOverlayNode(Node):
    """
    A ROS2 node that creates overlays of USD objects on camera images during bag replay.
    
    This node:
    1. Loads USD buffer data from a JSON file
    2. Subscribes to camera stream for visualization
    3. Projects USD objects into the image plane
    4. Creates visual overlays showing object locations
    5. Publishes the overlaid images
    """
    
    def __init__(self, json_path):
        super().__init__("usd_overlay_node")
        
        # Load USD buffer data from JSON
        self._load_usd_buffer(json_path)
        
        # Initialize subscribers
        self._init_subscribers()
        
        # Initialize publishers
        self._init_publishers()
        
        # Initialize data structures
        self._init_data_structures()
        
        # Initialize configuration
        self._init_config()

    def _load_usd_buffer(self, json_path):
        """Load USD buffer data from JSON file"""
        try:
            with open(json_path, 'r') as f:
                buffer_data = json.load(f)
            
            # Convert JSON data to our object format
            self.usd_objects = []
            for obj in buffer_data["objects"]:
                self.usd_objects.append({
                    'usd_path': obj["usd_path"],
                    'position': np.array(obj["position"]),
                    'quatWXYZ': np.array(obj["quatWXYZ"]),
                    'cluster_id': obj["cluster_id"],
                    'label': obj.get("label")  # Optional label
                })
            
            self.get_logger().info(f"Loaded {len(self.usd_objects)} objects from {json_path}")
        except Exception as e:
            self.get_logger().error(f"Error loading USD buffer from {json_path}: {e}")
            self.usd_objects = []

    def _init_subscribers(self):
        """Initialize ROS subscribers"""
        # Camera stream
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            10
        )
        self.rgb_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.rgb_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

    def _init_publishers(self):
        """Initialize ROS publishers"""
        self.overlay_pub = self.create_publisher(
            Image,
            "/usd/object_overlay",
            10
        )

    def _init_data_structures(self):
        """Initialize data structures"""
        self.bridge = CvBridge()
        self.cam_info = None
        self.rgb_image = None
        self.odom_info = {
            "t": np.array([0.0, 0.0, 0.0]),
            "q": np.array([0.0, 0.0, 0.0, 1.0]),
            "eulerXYZ": np.array([0.0, 0.0, 0.0])
        }
        self.usd_cache = {}  # Cache for USD vertices

    def _init_config(self):
        """Initialize configuration parameters"""
        # Camera offset in odom frame
        self.T_cam_in_odom = np.array([0.285, 0., 0.01])
        self.projection = ProjectionUtils(T=self.T_cam_in_odom)
        
        # Visualization parameters
        self.overlay_alpha = 0.3  # Transparency of overlay
        self.colors = {
            "Table": (0, 255, 0),    # Green
            "Chair": (255, 0, 0),    # Blue
            "Storage": (0, 0, 255),  # Red
            "Misc": (255, 255, 0)    # Cyan
        }

    def camera_info_callback(self, msg):
        """Handle incoming camera info messages"""
        self.cam_info = {
            "K": np.array(msg.k).reshape(3, 3),
            "R": np.array(msg.r),
            "P": np.array(msg.p).reshape(3, 4),
            "width": msg.width,
            "height": msg.height
        }

    def rgb_callback(self, msg):
        """Handle incoming RGB image messages"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Create and publish overlay if we have all required data
            if self.cam_info is not None and len(self.usd_objects) > 0:
                self._publish_object_overlay()
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def odom_callback(self, msg):
        """Handle incoming odometry messages"""
        self.odom_info["t"] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.odom_info["q"] = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.odom_info["eulerXYZ"] = R.from_quat(self.odom_info["q"]).as_euler("xyz", degrees=True)

    def get_usd_vertices(self, usd_path):
        """Get vertices from USD file and cache them"""
        if usd_path not in self.usd_cache:
            try:
                stage = Usd.Stage.Open(usd_path)
                if not stage:
                    self.get_logger().error(f"Could not open USD stage at {usd_path}")
                    return None

                meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
                vertices = []
                
                # Get all mesh vertices
                for prim in stage.TraverseAll():
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        points_attr = mesh.GetPointsAttr()
                        if points_attr:
                            points = points_attr.Get()
                            if points:
                                xform = UsdGeom.Xformable(prim)
                                matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                                for p in points:
                                    transformed_p = matrix.Transform(p)
                                    scaled_p = transformed_p * meters_per_unit
                                    vertices.append(scaled_p)

                vertices = np.array(vertices)
                if len(vertices) == 0:
                    self.get_logger().error(f"No vertices found in USD file: {usd_path}")
                    return None

                self.usd_cache[usd_path] = vertices
            except Exception as e:
                self.get_logger().error(f"Error loading USD file {usd_path}: {e}")
                return None

        return self.usd_cache[usd_path]

    def _project_points_to_image(self, points, position, orientation):
        """Project 3D points to image plane using camera parameters"""
        if len(points) == 0:
            return np.array([])

        # Transform points from world to camera frame
        R_world_from_odom = R.from_quat(self.odom_info["q"]).as_matrix()
        t_world_from_odom = self.odom_info["t"]
        T_world_from_odom = self.projection._create_homogeneous_matrix(R_world_from_odom, t_world_from_odom)
        
        # Transform points from object to world frame
        r = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        rot_matrix = r.as_matrix()
        points_world = np.dot(rot_matrix, points.T).T + position
        
        # Transform to camera frame
        T_cam_from_world = np.linalg.inv(self.projection.T_odom_from_cam) @ np.linalg.inv(T_world_from_odom)
        points_h = np.hstack((points_world, np.ones((points_world.shape[0], 1))))
        points_cam = (T_cam_from_world @ points_h.T).T
        points_cam = points_cam[:, :3] / points_cam[:, 3:]
        
        # Filter points behind camera
        points_cam = points_cam[points_cam[:, 2] > 0]
        if len(points_cam) == 0:
            return np.array([])
        
        # Project to image plane using camera matrix
        uv = (self.cam_info["K"] @ points_cam.T).T
        uv = uv[:, :2] / uv[:, 2:]
        
        # Filter points outside image bounds
        mask = (
            (uv[:, 0] >= 0) & 
            (uv[:, 0] < self.cam_info["width"]) & 
            (uv[:, 1] >= 0) & 
            (uv[:, 1] < self.cam_info["height"])
        )
        
        return uv[mask]

    def _get_object_color(self, usd_path):
        """Get color for object based on its path"""
        # Extract label from path (assuming path contains label)
        try:
            parts = usd_path.split('/')
            for part in reversed(parts):
                if part in self.colors:
                    return self.colors[part]
        except:
            pass
        return (128, 128, 128)  # Default gray color

    def _publish_object_overlay(self):
        """Create and publish overlay of USD objects on camera image"""
        if self.rgb_image is None or self.cam_info is None:
            return

        # Create a copy of the RGB image for overlay
        overlay = self.rgb_image.copy()
        
        # Project each USD object
        for obj in self.usd_objects:
            # Get object vertices
            vertices = self.get_usd_vertices(obj['usd_path'])
            if vertices is None:
                continue

            # Project vertices to image plane
            projected_points = self._project_points_to_image(
                vertices,
                obj['position'],
                obj['quatWXYZ']
            )
            
            if len(projected_points) > 0:
                # Get color for this object
                color = self._get_object_color(obj['usd_path'])
                
                # Create convex hull of projected points
                try:
                    hull = cv2.convexHull(projected_points.astype(np.int32))
                    # Draw filled polygon with semi-transparency
                    overlay = cv2.fillPoly(
                        overlay,
                        [hull],
                        color,
                        cv2.LINE_AA
                    )
                    
                    # Draw object ID
                    if len(hull) > 0:
                        # Convert center point to tuple of integers for cv2.putText
                        center = tuple(map(int, np.mean(hull, axis=0).flatten()))
                        cv2.putText(
                            overlay,
                            f"ID: {obj['cluster_id']}",
                            center,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                except Exception as e:
                    self.get_logger().warn(f"Error creating convex hull: {e}")

        # Blend overlay with original image
        blended = cv2.addWeighted(
            self.rgb_image,
            1 - self.overlay_alpha,
            overlay,
            self.overlay_alpha,
            0
        )
        
        # Publish overlay
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
            self.overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing overlay: {e}")

def main():
    """Main entry point for the USD overlay node"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='USD Overlay Node')
    parser.add_argument('json_path', type=str, help='Path to the USD buffer JSON file')
    args = parser.parse_args()
    
    rclpy.init()
    usd_overlay_node = USDOverlayNode(args.json_path)
    rclpy.spin(usd_overlay_node)
    usd_overlay_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main() 