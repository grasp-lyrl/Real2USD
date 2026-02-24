import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from custom_message.msg import UsdStringIdPCMsg
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from action_msgs.msg import GoalStatusArray, GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
from ipdb import set_trace as st
from google import genai
import os, time
import json
import pydantic
from typing import List, Optional, Dict, Any, Union
import math
import glob

def _load_gemini_key():
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory("real2sam3d")
        key_path = os.path.join(pkg_share, "config", "gemini_key.py")
        with open(key_path) as f:
            ns = {}
            exec(f.read(), ns)
            return ns.get("GEMINI_API_KEY", "")
    except Exception:
        try:
            from config.gemini_key import GEMINI_API_KEY
            return GEMINI_API_KEY
        except Exception:
            return ""
GEMINI_API_KEY = _load_gemini_key()

"""
This node listens to a query and generates waypoints for navigation.

load the usda by file name or it will automatically load the usda from todays date.

ros2 run rea2usd llm_navigator_node
=== or ===
ros2 run rea2usd llm_navigator_node --ros-args -p context_file:=/path/to/your/context.txt

ros2 topic pub --once /nav_query std_msgs/msg/String "{data: 'chair'}"
ros2 topic pub --once /nav_query std_msgs/msg/String "{data: 'plan a path that goes to each of the chairs'}"
"""

# Define Pydantic models for structured output
class Waypoint(pydantic.BaseModel):
    """A single waypoint with position and orientation"""
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    description: Optional[str] = None

class ObjectWaypoint(pydantic.BaseModel):
    """A waypoint associated with an object"""
    object_name: str
    waypoint: Waypoint
    object_id: Optional[str] = None

class PathPlan(pydantic.BaseModel):
    """A plan with multiple waypoints"""
    waypoints: List[Waypoint]
    description: Optional[str] = None

class ObjectPathPlan(pydantic.BaseModel):
    """A plan with multiple object waypoints"""
    object_waypoints: List[ObjectWaypoint]
    description: Optional[str] = None

class LlmNavigatorNode(Node):
    def __init__(self):
        super().__init__("llm_navigator_node")

        self.sub = self.create_subscription(String, "/nav_query", self.listener_callback, 10)
        
        # Create a publisher for the raw JSON response
        self.response_pub = self.create_publisher(String, '/llm_response', 10)
        
        # Create a publisher for waypoint markers
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        
        # Create action client for Nav2
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Marker ID counter
        self.marker_id = 0
        
        # Configure Gemini
        if not GEMINI_API_KEY:
            self.get_logger().warn("GEMINI_API_KEY is empty. Copy config/gemini_key_template.py to config/gemini_key.py and set your key.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.context_file = self.declare_parameter('context_file', '').get_parameter_value().string_value
        
        # Parameter for number of recent USD files to use
        self.num_recent_files = self.declare_parameter('num_recent_files', 3).get_parameter_value().integer_value

        self.context_text = ""
        if self.context_file:
            try:
                with open(self.context_file, 'r') as f:
                    self.context_text = f.read()
                self.get_logger().info(f"Context file '{self.context_file}' loaded.")
            except FileNotFoundError:
                self.get_logger().warn(f"Context file '{self.context_file}' not found.")
            except Exception as e:
                self.get_logger().warn(f"Error loading context file: {e}")

        # Define the base prompt
        self.base_prompt = """
        You are a navigation assistant that helps plan paths to objects in a scene.
        
        The context includes multiple USD files from recent observations, ordered from oldest to newest.
        Each file represents a snapshot of the scene at a specific point in time.
        
        Your task is to analyze these files and generate waypoints for navigation based on the query.
        
        For each waypoint, provide:
        - Position (x, y, z coordinates)
        - Orientation (quaternion: qx, qy, qz, qw)
        - A brief description of the waypoint
        
        If the query is about a specific object, identify that object in the scene and generate a waypoint to it.
        If the query is about planning a path to multiple objects, generate a sequence of waypoints.
        
        You MUST return your response in valid JSON format that matches the provided schema.
        """

        # Define custom schemas for Gemini API (without default values)
        self.waypoint_schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "qx": {"type": "number"},
                "qy": {"type": "number"},
                "qz": {"type": "number"},
                "qw": {"type": "number"},
                "description": {"type": "string"}
            },
            "required": ["x", "y", "z", "qx", "qy", "qz", "qw"]
        }
        
        self.object_waypoint_schema = {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "waypoint": self.waypoint_schema,
                "object_id": {"type": "string"}
            },
            "required": ["object_name", "waypoint"]
        }
        
        self.path_plan_schema = {
            "type": "object",
            "properties": {
                "waypoints": {
                    "type": "array",
                    "items": self.waypoint_schema
                },
                "description": {"type": "string"}
            },
            "required": ["waypoints"]
        }
        
        self.object_path_plan_schema = {
            "type": "object",
            "properties": {
                "object_waypoints": {
                    "type": "array",
                    "items": self.object_waypoint_schema
                },
                "description": {"type": "string"}
            },
            "required": ["object_waypoints"]
        }
        
        # Navigation state tracking
        self.current_schema_type = None
        self.current_path_plan = None
        self.current_waypoint_index = 0
        self.is_navigating = False
        self.navigation_start_time = None
        self.navigation_timeout = 60.0  # 60 seconds timeout for navigation

    def publish_waypoint_markers(self, waypoints, is_object_waypoints=False):
        """Publish markers for all waypoints in the path"""
        marker_array = MarkerArray()
        
        # Create markers for each waypoint
        for i, waypoint in enumerate(waypoints):
            # Get the actual waypoint object
            wp = waypoint.waypoint if is_object_waypoints else waypoint
            
            # Create sphere marker for waypoint position
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "map"
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.ns = "waypoints"
            sphere_marker.id = i * 3  # Use consistent IDs
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = Point(x=float(wp.x), y=float(wp.y), z=float(wp.z))
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.2  # 20cm diameter
            sphere_marker.scale.y = 0.2
            sphere_marker.scale.z = 0.2
            sphere_marker.color.r = 0.0
            sphere_marker.color.g = 1.0
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 0.8
            marker_array.markers.append(sphere_marker)
            
            # Create text marker for waypoint label
            text_marker = Marker()
            text_marker.header = sphere_marker.header
            text_marker.ns = "waypoint_labels"
            text_marker.id = i * 3 + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = Point(x=float(wp.x), y=float(wp.y), z=float(wp.z) + 0.3)  # Offset above sphere
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2  # Text size
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            # Create label text
            if is_object_waypoints:
                label = f"{i+1}: {waypoint.object_name}"
                if waypoint.object_id:
                    label += f" ({waypoint.object_id})"
            else:
                label = f"{i+1}"
                if hasattr(wp, 'description') and wp.description:
                    label += f": {wp.description}"
            
            text_marker.text = label
            marker_array.markers.append(text_marker)
            
            # Create arrow marker for orientation
            arrow_marker = Marker()
            arrow_marker.header = sphere_marker.header
            arrow_marker.ns = "waypoint_orientations"
            arrow_marker.id = i * 3 + 2
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.pose.position = Point(x=float(wp.x), y=float(wp.y), z=float(wp.z))
            arrow_marker.pose.orientation.x = float(wp.qx)
            arrow_marker.pose.orientation.y = float(wp.qy)
            arrow_marker.pose.orientation.z = float(wp.qz)
            arrow_marker.pose.orientation.w = float(wp.qw)
            arrow_marker.scale.x = 0.3  # Shaft length
            arrow_marker.scale.y = 0.05  # Shaft diameter
            arrow_marker.scale.z = 0.05  # Head diameter
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 1.0
            arrow_marker.color.a = 0.8
            marker_array.markers.append(arrow_marker)
        
        # Add a clear marker at the start
        clear_marker = Marker()
        clear_marker.header.frame_id = "map"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.insert(0, clear_marker)
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(waypoints)} waypoint markers")
        
        # Debug print the first marker
        if marker_array.markers:
            self.get_logger().info(f"First marker: type={marker_array.markers[1].type}, action={marker_array.markers[1].action}, pos=({marker_array.markers[1].pose.position.x}, {marker_array.markers[1].pose.position.y}, {marker_array.markers[1].pose.position.z})")

    def listener_callback(self, msg):
        self.get_logger().info("Received: %s" % msg.data)

        # load the usda data based on the date saved by usd_builder.py
        if not self.context_file:
            base_path = "/data/SimIsaacData/usda"
            current_date = time.strftime("%Y%m%d")
            
            # Find all state files for today
            file_pattern = os.path.join(base_path, f"state_{current_date}_*.usda")
            state_files = glob.glob(file_pattern)
            
            # Sort files by number (oldest first) to show progression over time
            state_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Take the most recent files up to the specified limit
            recent_files = state_files[-self.num_recent_files:] if len(state_files) > self.num_recent_files else state_files
            
            if recent_files:
                try:
                    # Combine the content of all recent files
                    combined_context = ""
                    for file_path in recent_files:
                        with open(file_path, 'r') as f:
                            file_content = f.read()
                            # Add a separator and file name to distinguish between files
                            combined_context += f"\n\n=== USD File: {os.path.basename(file_path)} ===\n\n"
                            combined_context += file_content
                    
                    self.context_text = combined_context
                    self.get_logger().info(f"Loaded context from {len(recent_files)} recent files")
                except Exception as e:
                    self.get_logger().warn(f"Error loading context files: {e}")
            else:
                self.get_logger().warn(f"No state files found for today ({current_date})")
        
        # Determine which schema to use based on the query
        if "path" in msg.data.lower() or "plan" in msg.data.lower():
            if "each" in msg.data.lower() or "multiple" in msg.data.lower():
                schema_json = json.dumps(self.object_path_plan_schema, indent=2)
                self.current_schema_type = "object_path_plan"
            else:
                schema_json = json.dumps(self.path_plan_schema, indent=2)
                self.current_schema_type = "path_plan"
        else:
            if "each" in msg.data.lower() or "multiple" in msg.data.lower():
                schema_json = json.dumps(self.object_waypoint_schema, indent=2)
                self.current_schema_type = "object_waypoint"
            else:
                schema_json = json.dumps(self.waypoint_schema, indent=2)
                self.current_schema_type = "waypoint"
        
        # Create the full prompt with schema
        full_prompt = f"{self.base_prompt}\n\nUse this JSON schema:\n{schema_json}\n\nQuery: {msg.data}"
        
        # Generate content with the schema
        try:
            self.get_logger().info("Sending request to Gemini API...")
            # Configure generation with schema
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[self.context_text, full_prompt],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': json.loads(schema_json)
                }
            )
            
            # Extract the response text
            response_text = response.text
            self.get_logger().info(f"Received response from Gemini API: {response_text[:200]}...")
            
            # Try to parse the response as JSON
            try:
                # Find JSON in the response (in case there's additional text)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    self.get_logger().info(f"Extracted JSON: {json_str[:200]}...")
                    parsed_json = json.loads(json_str)
                    
                    # Validate against the schema
                    self.get_logger().info("Validating against schema...")
                    
                    if self.current_schema_type == "waypoint":
                        validated = Waypoint.model_validate(parsed_json)
                        self.publish_waypoint_markers([validated])
                        self.publish_waypoint(validated)
                    elif self.current_schema_type == "object_waypoint":
                        validated = ObjectWaypoint.model_validate(parsed_json)
                        self.publish_waypoint_markers([validated], is_object_waypoints=True)
                        self.publish_waypoint(validated.waypoint)
                    elif self.current_schema_type == "path_plan":
                        validated = PathPlan.model_validate(parsed_json)
                        self.current_path_plan = validated
                        self.current_waypoint_index = 0
                        self.publish_waypoint_markers(validated.waypoints)
                        if validated.waypoints:
                            self.publish_waypoint(validated.waypoints[0])
                    elif self.current_schema_type == "object_path_plan":
                        validated = ObjectPathPlan.model_validate(parsed_json)
                        self.current_path_plan = validated
                        self.current_waypoint_index = 0
                        self.publish_waypoint_markers(validated.object_waypoints, is_object_waypoints=True)
                        if validated.object_waypoints:
                            self.publish_waypoint(validated.object_waypoints[0].waypoint)
                    
                    self.get_logger().info(f'Validated response: {json.dumps(parsed_json, indent=2)}')
                    
                    # Publish the validated response
                    self.publish_response(parsed_json)
                else:
                    self.get_logger().warn(f'No JSON found in response: {response_text}')
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Failed to parse JSON: {e}')
                self.get_logger().error(f'Response text: {response_text}')
            except pydantic.ValidationError as e:
                self.get_logger().error(f'Schema validation failed: {e}')
                self.get_logger().error(f'Parsed JSON: {parsed_json}')
        except Exception as e:
            self.get_logger().error(f'Error generating content: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
    
    def publish_response(self, response_data):
        """Publish the raw JSON response"""
        response_msg = String()
        response_msg.data = json.dumps(response_data)
        self.response_pub.publish(response_msg)
        self.get_logger().info(f'Published response: {response_msg.data[:200]}...')
    
    def publish_waypoint(self, waypoint):
        """Send a waypoint to Nav2 using the action client"""
        if not self._action_client.server_is_ready():
            self.get_logger().warn("Action server /navigate_to_pose not ready, waiting...")
            if not self._action_client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error("Action server /navigate_to_pose not available after waiting. Aborting navigation.")
                return

        # Create a PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        pose_msg.pose.position.x = waypoint.x
        pose_msg.pose.position.y = waypoint.y
        pose_msg.pose.position.z = waypoint.z
        
        # Set orientation
        pose_msg.pose.orientation.x = waypoint.qx
        pose_msg.pose.orientation.y = waypoint.qy
        pose_msg.pose.orientation.z = waypoint.qz
        pose_msg.pose.orientation.w = waypoint.qw
        
        # Create the goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg
        
        # Send the goal
        self.get_logger().info(f'Sending navigation goal: x={waypoint.x}, y={waypoint.y}, z={waypoint.z}')
        
        # Send goal and wait for result
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
        # Set navigation state
        self.is_navigating = True
        self.navigation_start_time = time.time()
    
    def goal_response_callback(self, future):
        """Handle the response from the action server"""
        goal_handle = future.result()
        if not goal_handle:
            self.get_logger().error(f"Goal handle was invalid: {future.exception()}")
            self.is_navigating = False
            self.publish_next_waypoint()
            return
        if not goal_handle.accepted:
            self.get_logger().error("Navigation goal rejected")
            self.is_navigating = False
            self.publish_next_waypoint()
            return

        self.get_logger().info("Navigation goal accepted")
        
        # Get the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """Handle the result from the action server"""
        self.is_navigating = False
        
        try:
            result = future.result()
            status = result.status
            
            self.get_logger().info(f"Navigation goal finished with status: {status}")
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Navigation goal succeeded!")
                self.publish_next_waypoint()
            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().error("Navigation goal aborted")
                self.publish_next_waypoint()
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn("Navigation goal canceled")
                self.publish_next_waypoint()
            else:
                self.get_logger().error(f"Navigation goal failed with unknown status: {status}")
                self.publish_next_waypoint()
        except Exception as e:
            self.get_logger().error(f"Error getting navigation result: {e}")
            self.publish_next_waypoint()
    
    def publish_next_waypoint(self):
        """Publish the next waypoint in the path plan"""
        if not self.current_path_plan:
            return
        
        self.current_waypoint_index += 1
        
        if self.current_schema_type == "path_plan":
            if self.current_waypoint_index < len(self.current_path_plan.waypoints):
                self.get_logger().info(f"Moving to next waypoint {self.current_waypoint_index + 1} of {len(self.current_path_plan.waypoints)}")
                self.publish_waypoint(self.current_path_plan.waypoints[self.current_waypoint_index])
            else:
                self.get_logger().info("Completed all waypoints in path plan")
                self.current_path_plan = None
        elif self.current_schema_type == "object_path_plan":
            if self.current_waypoint_index < len(self.current_path_plan.object_waypoints):
                self.get_logger().info(f"Moving to next object waypoint {self.current_waypoint_index + 1} of {len(self.current_path_plan.object_waypoints)}")
                self.publish_waypoint(self.current_path_plan.object_waypoints[self.current_waypoint_index].waypoint)
            else:
                self.get_logger().info("Completed all waypoints in object path plan")
                self.current_path_plan = None


def main(args=None):
    rclpy.init(args=args)
    llm_node = LlmNavigatorNode()
    llm_node.get_logger().info("LLM Navigator node started")
    try:
        rclpy.spin(llm_node)
    except KeyboardInterrupt:
        llm_node.get_logger().info("KeyboardInterrupt received, shutting down.")
        pass
    except Exception as e:
        llm_node.get_logger().error(f"Unhandled exception during spin: {e}")
    finally:
        llm_node.get_logger().info("Shutting down LLM Navigator node.")
        if rclpy.ok():
            llm_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
