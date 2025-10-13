#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import sqlite3
from pathlib import Path

def find_rosbags_convert():
    """
    Find the rosbags-convert command in common installation locations.
    
    Returns:
        str: Path to rosbags-convert command, or None if not found
    """
    # Common locations where pip installs scripts
    possible_paths = [
        "rosbags-convert",  # Try PATH first
        os.path.expanduser("~/.local/bin/rosbags-convert"),
        os.path.expanduser("~/tmp/.local/bin/rosbags-convert"),
        "/usr/local/bin/rosbags-convert",
        "/usr/bin/rosbags-convert"
    ]
    
    for path in possible_paths:
        if path == "rosbags-convert":
            # Check if it's in PATH
            try:
                result = subprocess.run(["which", "rosbags-convert"], capture_output=True, text=True)
                if result.returncode == 0:
                    return path
            except:
                continue
        elif os.path.isfile(path):
            return path
    
    return None

def check_compatible_topics(ros2_bag_path):
    """
    Check which topics in the ROS2 bag are compatible with ROS1.
    
    Args:
        ros2_bag_path: Path to the ROS2 bag directory or .db3 file
    
    Returns:
        tuple: (compatible_topics, incompatible_topics)
    """
    
    # Known compatible ROS1 message types
    compatible_types = {
        'sensor_msgs/msg/Image': 'sensor_msgs/Image',
        'sensor_msgs/msg/CameraInfo': 'sensor_msgs/CameraInfo',
        'geometry_msgs/msg/PoseStamped': 'geometry_msgs/PoseStamped',
        'geometry_msgs/msg/TwistStamped': 'geometry_msgs/TwistStamped',
        'std_msgs/msg/String': 'std_msgs/String',
        'std_msgs/msg/Header': 'std_msgs/Header',
        'nav_msgs/msg/OccupancyGrid': 'nav_msgs/OccupancyGrid',
        'tf2_msgs/msg/TFMessage': 'tf2_msgs/TFMessage'
    }
    
    # Determine the actual database file path
    db_path = ros2_bag_path
    
    # If it's a directory, check if it's a ROS2 bag directory
    if os.path.isdir(ros2_bag_path):
        # Check if it's a ROS2 bag directory (has metadata.yaml)
        metadata_path = os.path.join(ros2_bag_path, 'metadata.yaml')
        if os.path.exists(metadata_path):
            # This is a proper ROS2 bag directory, look for .db3 file
            db_files = []
            for item in os.listdir(ros2_bag_path):
                if item.endswith('.db3'):
                    db_files.append(os.path.join(ros2_bag_path, item))
            
            if not db_files:
                print("Error: No .db3 files found in ROS2 bag directory")
                return [], []
            
            db_path = db_files[0]
        else:
            # Look for .db3 files in the directory
            db_files = []
            for item in os.listdir(ros2_bag_path):
                if item.endswith('.db3'):
                    db_files.append(os.path.join(ros2_bag_path, item))
            
            if not db_files:
                print("Error: No .db3 files found in ROS2 bag directory")
                return [], []
            
            db_path = db_files[0]
    
    # Connect to ROS2 bag database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    
    compatible_topics = []
    incompatible_topics = []
    
    for topic_id, topic_name, topic_type in topics:
        if topic_type in compatible_types:
            compatible_topics.append((topic_id, topic_name, topic_type))
        else:
            incompatible_topics.append((topic_id, topic_name, topic_type))
    
    conn.close()
    
    return compatible_topics, incompatible_topics

def convert_ros2_to_ros1_final(ros2_bag_path, output_path=None, force=False):
    """
    Convert ROS2 bag to ROS1 format using the rosbags-convert command.
    
    Args:
        ros2_bag_path: Path to the ROS2 bag directory or .db3 file
        output_path: Output path for ROS1 bag file
        force: Force conversion even if incompatible topics exist
    """
    
    if not os.path.exists(ros2_bag_path):
        print(f"Error: ROS2 bag path not found: {ros2_bag_path}")
        return False
    
    # Find the rosbags-convert command
    rosbags_convert_cmd = find_rosbags_convert()
    if not rosbags_convert_cmd:
        print("✗ Error: rosbags-convert command not found.")
        print("Make sure rosbags is installed: pip install rosbags")
        print("If installed with --user, try adding ~/.local/bin to your PATH")
        return False
    
    # Determine the actual bag path
    actual_bag_path = ros2_bag_path
    
    # If it's a directory, check if it's a ROS2 bag directory
    if os.path.isdir(ros2_bag_path):
        # Check if it's a ROS2 bag directory (has metadata.yaml)
        metadata_path = os.path.join(ros2_bag_path, 'metadata.yaml')
        if os.path.exists(metadata_path):
            # This is a proper ROS2 bag directory
            actual_bag_path = ros2_bag_path
            print(f"Found ROS2 bag directory: {ros2_bag_path}")
        else:
            # Look for .db3 files in the directory
            db_files = []
            for item in os.listdir(ros2_bag_path):
                if item.endswith('.db3'):
                    db_files.append(os.path.join(ros2_bag_path, item))
            
            if not db_files:
                print(f"Error: No .db3 files found in ROS2 bag directory: {ros2_bag_path}")
                return False
            
            if len(db_files) > 1:
                print(f"Warning: Multiple .db3 files found in {ros2_bag_path}:")
                for db_file in db_files:
                    print(f"  - {db_file}")
                print(f"Using the first one: {db_files[0]}")
            
            actual_bag_path = db_files[0]
    
    # Determine output path
    if output_path is None:
        if os.path.isdir(actual_bag_path):
            bag_name = Path(actual_bag_path).name
        else:
            bag_name = Path(actual_bag_path).stem
        output_path = f"{bag_name}_ros1.bag"
    
    print(f"Converting ROS2 bag: {actual_bag_path}")
    print(f"Output ROS1 bag: {output_path}")
    
    # Check topic compatibility
    print("\n=== Checking topic compatibility ===")
    compatible_topics, incompatible_topics = check_compatible_topics(actual_bag_path)
    
    print(f"✓ Compatible topics ({len(compatible_topics)}):")
    for topic_id, topic_name, topic_type in compatible_topics:
        print(f"  - {topic_name} ({topic_type})")
    
    if incompatible_topics:
        print(f"\n⚠ Incompatible topics ({len(incompatible_topics)}):")
        for topic_id, topic_name, topic_type in incompatible_topics:
            print(f"  - {topic_name} ({topic_type})")
        
        if not force:
            print(f"\n⚠ Warning: {len(incompatible_topics)} incompatible topics found.")
            print("These topics may cause conversion to fail or be skipped.")
            response = input("Continue with conversion? (y/N): ")
            if response.lower() != 'y':
                print("Conversion cancelled.")
                return False
    
    if not compatible_topics:
        print("✗ No compatible topics found for ROS1 conversion.")
        return False
    
    try:
        # Use the official rosbags-convert command
        cmd = [
            rosbags_convert_cmd,
            "--src", actual_bag_path,
            "--dst", output_path,
            "--src-typestore", "ros2_humble"  # Specify the source typestore for ROS2
        ]
        
        print(f"\n=== Converting bag file ===")
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully converted bag file to: {output_path}")
            
            # Verify the output file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ Output file size: {file_size} bytes")
                
                # Check if we can read the ROS1 bag
                try:
                    info_cmd = ["rosbag", "info", output_path]
                    info_result = subprocess.run(info_cmd, capture_output=True, text=True)
                    if info_result.returncode == 0:
                        print("✓ ROS1 bag file is valid and readable")
                    else:
                        print("⚠ ROS1 bag file created but may have issues")
                except FileNotFoundError:
                    print("⚠ rosbag command not available for verification")
            else:
                print("⚠ Output file was not created")
                return False
                
            return True
        else:
            print(f"✗ Conversion failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ Error: rosbags-convert command not found.")
        print("Make sure rosbags is installed: pip install rosbags")
        return False
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag files to ROS1 format using rosbags-convert')
    parser.add_argument('input', help='ROS2 bag directory path or .db3 file')
    parser.add_argument('-o', '--output', help='Output path for ROS1 bag file')
    parser.add_argument('--force', action='store_true', help='Force conversion even with incompatible topics')
    parser.add_argument('--check-only', action='store_true', help='Only check compatibility, do not convert')
    
    args = parser.parse_args()
    
    if args.check_only:
        # Only check compatibility
        compatible_topics, incompatible_topics = check_compatible_topics(args.input)
        print(f"\nCompatibility Summary:")
        print(f"✓ Compatible topics: {len(compatible_topics)}")
        print(f"⚠ Incompatible topics: {len(incompatible_topics)}")
        return 0
    
    success = convert_ros2_to_ros1_final(args.input, args.output, args.force)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())