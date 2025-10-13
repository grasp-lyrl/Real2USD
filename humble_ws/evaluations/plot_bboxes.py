import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os
from datetime import datetime
from parse_usda import parse_usda_data
from parse_clio import parse_clio_data
from parse_supervisely_bbox_json import parse_cuboid_data
from scipy.spatial.transform import Rotation
from ipdb import set_trace as st
import matplotlib.transforms as transforms
import seaborn as sns
import pickle
sns.set_theme()
fsz = 24
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.7*fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})

def plot_boxes_xy(usda_data=None, clio_data=None, supervisely_data=None, odom_pkl_file=None, output_dir='./evaluations/results'):
    """
    Plot XY positions of USDA (estimated), CLIO (estimated), and Supervisely (ground truth) boxes.
    
    Args:
        usda_data (dict, optional): USDA box data
        clio_data (dict, optional): CLIO box data
        supervisely_data (dict, optional): Supervisely box data
        odom_pkl_file (str, optional): Path to odometry pickle file
        output_dir (str): Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Track which labels we've already added to the legend
    added_labels = set()
    
    # Initialize min and max coordinates
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    # Plot odometry trajectory if pickle file is provided
    if odom_pkl_file is not None and os.path.exists(odom_pkl_file):
        print(f"\nLoading odometry data from: {odom_pkl_file}")
        try:
            with open(odom_pkl_file, 'rb') as f:
                odom_buffer = pickle.load(f)
            
            print(f"Loaded {len(odom_buffer)} odometry entries")
            # Extract positions and plot trajectory
            positions = []
            orientations = []
            
            for odom_entry in odom_buffer:
                if odom_entry["t"] is not None and odom_entry["q"] is not None:
                    positions.append(odom_entry["t"][:2])  # Only x, y coordinates
                    # Convert quaternion to yaw angle for arrow direction
                    rot = Rotation.from_quat(odom_entry["q"])
                    yaw = rot.as_euler('xyz')[2]  # Get yaw angle in radians
                    orientations.append(yaw)

            # Plot trajectory line
            if positions:
                positions = np.array(positions)
                orientations = np.array(orientations)
                
                # Update min/max coordinates to include trajectory
                min_x = min(min_x, np.min(positions[:, 0]))
                max_x = max(max_x, np.max(positions[:, 0]))
                min_y = min(min_y, np.min(positions[:, 1]))
                max_y = max(max_y, np.max(positions[:, 1]))
                
                # Plot trajectory line
                ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.7, linewidth=2, label='Robot Trajectory')
                
                # Plot arrows for every 10th entry
                arrow_step = 5
                for i in range(0, len(positions), arrow_step):
                    if i < len(positions) and i < len(orientations):
                        x, y = positions[i]
                        yaw = orientations[i]
                        
                        # Calculate arrow direction (unit vector)
                        dx = 0.5 * np.cos(yaw)  # Arrow length of 0.1 units
                        dy = 0.5 * np.sin(yaw)
                        
                        # Create arrow
                        arrow = FancyArrowPatch(
                            (x, y), (x + dx, y + dy),
                            arrowstyle='->', mutation_scale=15,
                            color='black', alpha=0.8, linewidth=2
                        )
                        ax.add_patch(arrow)
                
                print(f"Plotted trajectory with {len(positions)} points and arrows every {arrow_step} points")
                
        except Exception as e:
            print(f"Error loading odometry pickle file: {e}")
    

    # Plot USDA (estimated) boxes if data is provided
    # Plot USDA (estimated) boxes if data is provided
    if usda_data is not None:
        print("\nPlotting USDA boxes...")
        for label, boxes in usda_data.items():
            for box in boxes:
                bbox_min = box['bbox_min']
                bbox_max = box['bbox_max']
                
                # Update min/max coordinates using actual bounding box values
                min_x = min(min_x, bbox_min[0])
                max_x = max(max_x, bbox_max[0])
                min_y = min(min_y, bbox_min[1])
                max_y = max(max_y, bbox_max[1])
                
                # Create rectangle for XY view using bbox min/max (ignoring Z)
                rect = Rectangle(
                    (bbox_min[0], bbox_min[1]),  # bottom left corner
                    bbox_max[0] - bbox_min[0],  # width
                    bbox_max[1] - bbox_min[1],  # height
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    alpha=0.8,
                    label=f'USDA ({label})' if label not in added_labels else None
                )
                
                if label not in added_labels:
                    added_labels.add(label)
                ax.add_patch(rect)
    
    # Plot CLIO (estimated) boxes if data is provided
    if clio_data is not None:
        print("\nPlotting CLIO boxes...")
        for label, boxes in clio_data.items():
            for box in boxes:
                pos = box['position']
                dim = box['dimensions']
                rotation = box['rotation']
                
                # Convert rotation matrix to yaw angle
                if isinstance(rotation, list) and len(rotation) == 3 and all(isinstance(r, list) for r in rotation):
                    rot = Rotation.from_matrix(rotation)
                    yaw = rot.as_euler('xyz')[2]  # Get yaw angle in radians
                else:
                    yaw = 0.0
                
                # Update min/max coordinates (approximate for rotated box)
                min_x = min(min_x, pos[0] - dim[0]/2)
                max_x = max(max_x, pos[0] + dim[0]/2)
                min_y = min(min_y, pos[1] - dim[1]/2)
                max_y = max(max_y, pos[1] + dim[1]/2)
                
                # Create rectangle for XY view (ignoring Z)
                rect = Rectangle(
                    (pos[0] - dim[0]/2, pos[1] - dim[1]/2),  # bottom left corner
                    dim[0],  # width
                    dim[1],  # height
                    linewidth=2,
                    edgecolor='g',
                    facecolor='none',
                    alpha=0.8,
                    label=f'CLIO ({label})' if label not in added_labels else None
                )
                
                # Set the rotation point to the center of the rectangle
                rotation = transforms.Affine2D().rotate_around(pos[0], pos[1], yaw) + ax.transData
                rect.set_transform(rotation)
                
                if label not in added_labels:
                    added_labels.add(label)
                ax.add_patch(rect)
                # Add label text
                # ax.text(pos[0], pos[1], f"{box['id']}: {label}", fontsize=8, ha='center', va='center', color='g')
    
    # Plot Supervisely (ground truth) boxes if data is provided
    if supervisely_data is not None:
        print("\nPlotting Supervisely boxes...")
        for label, boxes in supervisely_data.items():
            for box in boxes:
                pos = box['position']
                dim = box['dimensions']
                rotation = box['rotation']  # Euler angles [x, y, z]
                
                # Convert Euler angles to rotation matrix and extract yaw angle
                if isinstance(rotation, dict):
                    # If rotation is a dictionary, extract the values
                    rotation = [rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)]
                rot = Rotation.from_euler('xyz', rotation)
                yaw = rot.as_euler('xyz')[2]  # Get yaw angle in radians
                
                # Update min/max coordinates (approximate for rotated box)
                min_x = min(min_x, pos[0] - dim[0]/2)
                max_x = max(max_x, pos[0] + dim[0]/2)
                min_y = min(min_y, pos[1] - dim[1]/2)
                max_y = max(max_y, pos[1] + dim[1]/2)
                
                # Create rectangle for XY view (ignoring Z)
                rect = Rectangle(
                    (pos[0] - dim[0]/2, pos[1] - dim[1]/2),  # bottom left corner
                    dim[0],  # width
                    dim[1],  # height
                    linewidth=2,
                    edgecolor='b',
                    facecolor='none',
                    alpha=0.8,
                    label=f'Supervisely ({label})' if label not in added_labels else None
                )
                
                # Set the rotation point to the center of the rectangle
                rotation = transforms.Affine2D().rotate_around(pos[0], pos[1], yaw) + ax.transData
                rect.set_transform(rotation)
                
                if label not in added_labels:
                    added_labels.add(label)
                ax.add_patch(rect)
                # Add label text
                ax.text(pos[0], pos[1], label, fontsize=12, ha='center', va='center', color='b')
    
    # Add padding to the plot limits
    padding = 0.0  # 10% padding
    x_range = max_x - min_x
    y_range = max_y - min_y
    ax.set_xlim(min_x - x_range * padding, max_x + x_range * padding)
    ax.set_ylim(min_y - y_range * padding, max_y + y_range * padding)
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title('Box Positions (XY View)')
    # ax.grid(True)
    # ax.legend()

    # Add legend entries for each data source
    legend_elements = []
    # Add legend entry for USDA (red boxes)
    legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Ours (USD)'))
    # Add legend entry for CLIO (green boxes)
    legend_elements.append(plt.Line2D([0], [0], color='green', lw=2, label='CLIO'))
    # Add legend entry for Supervisely (blue boxes)
    legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Ground Truth'))
    # Add legend entry for robot trajectory
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, label='Robot Trajectory'))
    
    # Display the legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    

    # Equal aspect ratio
    ax.set_aspect('equal')
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    # Save the plot
    output_file = os.path.join(output_dir, f'box_positions.png')
    # plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
    #maybe this is the right way to do it so pdf is type42
    fig.savefig(output_file, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    
    # Show the plot
    plt.show()

def main():
    # Example usage
    # usda_file = './evaluations/usda/hallway1_20250419.usda'
    # usda_file = '/data/SimIsaacData/usda/0_5edit_hallway_01_09082025.usda'
    # clio_file = './evaluations/clio/hallway_01.graphml'
    # supervisely_file = './evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json'
    # odom_pkl_file = './evaluations/odom/hallway_odom_buffer.pkl'

    # usda_file = '/data/SimIsaacData/usda/smalloffice0_edit05_09082025.usda'
    # clio_file = './evaluations/clio/smalloffice00.graphml'
    # supervisely_file = './evaluations/supervisely/smalloffice-0_voxel_pointcloud.pcd.json'
    # odom_pkl_file = './evaluations/odom/smalloffice0_odom_buffer.pkl'

    # usda_file = '/data/SimIsaacData/usda/lounge_09082025.usda'
    usda_file = '/data/SimIsaacData/usda/lounge_05_09082025.usda'
    clio_file = './evaluations/clio/lounge_0.graphml'
    supervisely_file = './evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json'
    odom_pkl_file = './evaluations/odom/lounge_odom_buffer.pkl'

    print(f"Attempting to read USDA file (estimated objects): {usda_file}")
    print(f"Attempting to read CLIO file (estimated objects): {clio_file}")
    print(f"Attempting to read Supervisely file (ground truth): {supervisely_file}")
    
    try:
        # Initialize data variables
        usda_data = None
        clio_data = None
        supervisely_data = None
        
        # Parse files if they exist
        if os.path.exists(usda_file):
            print("\nParsing USDA file (estimated objects)...")
            usda_data = parse_usda_data(usda_file)
        if os.path.exists(clio_file):
            print("\nParsing CLIO file (estimated objects)...")
            clio_data = parse_clio_data(clio_file)
        
        if os.path.exists(supervisely_file):
            print("\nParsing Supervisely file (ground truth)...")
            supervisely_data = parse_cuboid_data(supervisely_file)
        
        # Plot the boxes
        plot_boxes_xy(usda_data, clio_data, supervisely_data, odom_pkl_file)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 