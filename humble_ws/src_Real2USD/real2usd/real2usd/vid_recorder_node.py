import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import subprocess
from datetime import datetime
from collections import defaultdict
import numpy as np

class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__("video_recorder_node")
        
        # Configuration - Edit these as needed
        self.topics_config = {
            "/camera/image_raw": {
                "output_dir": "/home/me/video_frames/rgb",
                "frame_rate": 20.0,
                "save_every_n_frames": 1,
                "enabled": True
            },
            "/segment/image_segmented": {
                "output_dir": "/home/me/video_frames/segmented",
                "frame_rate": 20.0,
                "save_every_n_frames": 1,
                "enabled": True
            },
            # "/depth_image/rgbd": {
            #     "output_dir": "/home/me/video_frames/rgbd",
            #     "frame_rate": 10.0,
            #     "save_every_n_frames": 1,
            #     "enabled": True
            # },
            "/segment/image_cropped": {
                "output_dir": "/home/me/video_frames/image_cropped",
                "frame_rate": 20.0,
                "save_every_n_frames": 1,
                "enabled": True
            },
            "/usd_search/image": {
                "output_dir": "/home/me/video_frames/usd_search",
                "frame_rate": 1.0,
                "save_every_n_frames": 1,
                "enabled": True
            },
            "/usd_search/img_result": {
                "output_dir": "/home/me/video_frames/usd_search_result",
                "frame_rate": 1.0,
                "save_every_n_frames": 1,
                "enabled": True
            }
        }
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Frame counters and timing for each topic
        self.frame_counters = defaultdict(int)
        self.last_save_times = defaultdict(float)
        
        # Video writers for each topic
        self.video_writers = {}
        
        # Special handling for cropped images (they come individually per object from retrieval_node.py)
        self.cropped_image_buffer = {}  # Buffer to accumulate cropped images
        
        # Create subscribers for each enabled topic
        self.image_subscriptions = {}
        for topic, config in self.topics_config.items():
            if config["enabled"]:
                # Create output directory
                os.makedirs(config["output_dir"], exist_ok=True)
                
                # Create subscription
                self.image_subscriptions[topic] = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, t=topic: self.image_callback(msg, t),
                    10
                )
                
                self.get_logger().info(f"Subscribing to: {topic}")
                self.get_logger().info(f"  Output: {config['output_dir']}")
                self.get_logger().info(f"  Frame rate: {config['frame_rate']} fps")
        
        self.get_logger().info(f"Video recorder initialized with {len(self.image_subscriptions)} topics")
        
        # Timer for periodic status updates
        self.status_timer = self.create_timer(10.0, self.status_callback)
    
    def image_callback(self, msg, topic):
        """Callback function for incoming image messages"""
        try:
            config = self.topics_config[topic]
            
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Special handling for cropped images from retrieval_node.py
            if topic == "/segment/image_cropped":
                self.handle_cropped_image(cv_image, topic)
                return
            
            # Check if we should save this frame based on frame rate and save interval
            current_time = time.time()
            time_since_last_save = current_time - self.last_save_times[topic]
            
            # Event-driven topics (USD search results) - save every frame regardless of timing
            event_driven_topics = ["/usd_search/image", "/usd_search/img_result"]
            
            if topic in event_driven_topics:
                # For event-driven topics, save every frame regardless of timing
                should_save = True
            else:
                # For regular video streams, apply frame rate filtering
                should_save = (self.frame_counters[topic] % config["save_every_n_frames"] == 0 and 
                              time_since_last_save >= 1.0 / config["frame_rate"])
            
            if should_save:
                # Initialize video writer if not already done
                if topic not in self.video_writers:
                    self.initialize_video_writer(topic, cv_image)
                
                # Write frame to video
                if topic in self.video_writers and self.video_writers[topic] is not None:
                    self.video_writers[topic].write(cv_image)
                
                self.last_save_times[topic] = current_time
                
                # Log every 100 frames to avoid spam
                if self.frame_counters[topic] % 100 == 0:
                    self.get_logger().info(f"[{topic}] Wrote frame {self.frame_counters[topic]} to video")
            
            self.frame_counters[topic] += 1
            
        except Exception as e:
            self.get_logger().error(f"Error processing image from {topic}: {str(e)}")
    
    def handle_cropped_image(self, cv_image, topic):
        """Handle cropped images by resizing them to standard size and writing as individual frames"""
        try:
            config = self.topics_config[topic]
            
            # Resize image to standard size
            standard_width = 400
            standard_height = 400
            resized_image = cv2.resize(cv_image, (standard_width, standard_height))
            
            # Check if we should save this frame based on frame rate
            current_time = time.time()
            time_since_last_save = current_time - self.last_save_times.get(topic, 0)
            
            if time_since_last_save >= 1.0 / config["frame_rate"]:
                # Initialize video writer if not already done
                if topic not in self.video_writers:
                    self.initialize_video_writer(topic, resized_image)
                
                # Write resized frame to video
                if topic in self.video_writers and self.video_writers[topic] is not None:
                    self.video_writers[topic].write(resized_image)
                
                self.last_save_times[topic] = current_time
                
                # Log every 10 frames to avoid spam
                if self.frame_counters[topic] % 10 == 0:
                    self.get_logger().info(f"[{topic}] Wrote crop frame {self.frame_counters[topic]} to video")
            
            self.frame_counters[topic] += 1
            
        except Exception as e:
            self.get_logger().error(f"Error handling cropped image from {topic}: {str(e)}")
    
    def create_combined_cropped_frame(self, topic):
        """This method is no longer used - keeping for compatibility"""
        pass
    
    def initialize_video_writer(self, topic, cv_image):
        """Initialize video writer for a topic"""
        try:
            config = self.topics_config[topic]
            
            # Get image dimensions
            height, width = cv_image.shape[:2]
            
            # Create video filename
            topic_name = topic.replace("/", "_").replace("_", "").lstrip("_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{topic_name}_{timestamp}.mp4"
            video_path = os.path.join(config["output_dir"], video_filename)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, 
                fourcc, 
                config["frame_rate"], 
                (width, height)
            )
            
            if video_writer.isOpened():
                self.video_writers[topic] = video_writer
                self.get_logger().info(f"Initialized video writer for {topic}: {video_path}")
            else:
                self.get_logger().error(f"Failed to initialize video writer for {topic}")
                self.video_writers[topic] = None
                
        except Exception as e:
            self.get_logger().error(f"Error initializing video writer for {topic}: {str(e)}")
            self.video_writers[topic] = None
    
    def status_callback(self):
        """Periodic status update"""
        self.get_logger().info("=== Status Update ===")
        for topic, config in self.topics_config.items():
            if config["enabled"]:
                frames_processed = self.frame_counters[topic]
                frames_written = frames_processed // config["save_every_n_frames"]
                self.get_logger().info(f"[{topic}] Processed: {frames_processed}, Written: {frames_written}")
    
    def cleanup_video_writers(self):
        """Release all video writers"""
        self.get_logger().info("=== Cleaning up video writers ===")
        for topic, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
                self.get_logger().info(f"Released video writer for {topic}")
        self.video_writers.clear()
    
    def add_topic(self, topic, output_dir, frame_rate=30.0, save_every_n_frames=1):
        """Dynamically add a new topic to record"""
        if topic not in self.topics_config:
            self.topics_config[topic] = {
                "output_dir": output_dir,
                "frame_rate": frame_rate,
                "save_every_n_frames": save_every_n_frames,
                "enabled": True
            }
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create subscription
            self.image_subscriptions[topic] = self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self.image_callback(msg, t),
                10
            )
            
            self.get_logger().info(f"Added new topic: {topic}")
            self.get_logger().info(f"  Output: {output_dir}")
            self.get_logger().info(f"  Frame rate: {frame_rate} fps")

def main():
    rclpy.init()
    video_recorder_node = VideoRecorderNode()
    
    try:
        rclpy.spin(video_recorder_node)
    except KeyboardInterrupt:
        video_recorder_node.get_logger().info("Shutting down video recorder...")
        # Clean up video writers
        video_recorder_node.cleanup_video_writers()
    finally:
        video_recorder_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
