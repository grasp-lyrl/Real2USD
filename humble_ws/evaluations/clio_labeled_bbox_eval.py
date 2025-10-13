import numpy as np
from scipy.spatial.transform import Rotation
from parse_clio import parse_clio_data
from parse_supervisely_bbox_json import parse_cuboid_data
import json
import os
from datetime import datetime
from ipdb import set_trace as st
from shapely.geometry import Polygon
from label_mapped_metrics import calculate_mean_metrics, print_metrics


def get_yaw_from_box(box):
    """
    Extract yaw angle from either CLIO or Supervisely box format.
    
    Args:
        box (dict): Bounding box data
        
    Returns:
        float: Yaw angle in radians
    """
    if 'rotation' in box:
        # CLIO or Supervisely format
        rotation = box['rotation']
        if isinstance(rotation, dict):
            # If rotation is a dictionary, extract the values
            rotation = [rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)]
        elif isinstance(rotation, list) and len(rotation) == 3 and all(isinstance(r, list) for r in rotation):
            # If rotation is a 3x3 matrix (CLIO format), convert to Euler angles
            rot = Rotation.from_matrix(rotation)
            return rot.as_euler('xyz')[2]  # Get yaw angle in radians
        rot = Rotation.from_euler('xyz', rotation)
        return rot.as_euler('xyz')[2]  # Get yaw angle in radians
    else:
        # Default to 0 if no rotation information
        return 0.0

def get_box_corners(position, dimensions, yaw):
    """
    Get the corners of a rotated 2D box.
    
    Args:
        position (np.array): Center position [x, y, z]
        dimensions (np.array): Box dimensions [width, height, depth]
        yaw (float): Rotation angle in radians
        
    Returns:
        np.array: 4x2 array of corner coordinates (only using x,y coordinates)
    """
    # Get half dimensions (only using x and y dimensions)
    half_w = dimensions[0] / 2  # width
    half_h = dimensions[1] / 2  # height
    
    # Define corners in local coordinate system (only x,y coordinates)
    corners = np.array([
        [-half_w, -half_h],  # bottom-left
        [half_w, -half_h],   # bottom-right
        [half_w, half_h],    # top-right
        [-half_w, half_h]    # top-left
    ])
    
    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    
    # Rotate corners
    rotated_corners = np.dot(corners, rot_matrix.T)
    
    # Ensure position is 2D for translation (only using x and y coordinates)
    pos_2d = np.array(position[:2])
    
    # Translate to world coordinates (only x,y coordinates)
    world_corners = rotated_corners + pos_2d
    
    return world_corners

def compute_iou(corners1, corners2):
    """
    Compute the Intersection over Union (IoU) of two rotated rectangles.
    
    Args:
        corners1 (np.array): 4x2 array of corner coordinates for first box
        corners2 (np.array): 4x2 array of corner coordinates for second box
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Create polygons from corners
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    
    # Compute intersection and union
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    # Return IoU
    return intersection / union if union > 0 else 0.0


def compute_bbox_overlap(bbox1, bbox2):
    """
    Compute if two rotated bounding boxes overlap in 2D (XY plane).
    
    Args:
        bbox1 (dict): First bounding box with position, dimensions, and rotation
        bbox2 (dict): Second bounding box with position, dimensions, and rotation
        
    Returns:
        tuple: (overlap, strict_accuracy, relaxed_accuracy, iou)
            - overlap: True if boxes overlap in 2D
            - strict_accuracy: True if both centroids are contained in each other's boxes
            - relaxed_accuracy: True if at least one centroid is contained in the other's box
            - iou: Intersection over Union value
    """
    # Get yaw angles for both boxes
    yaw1 = get_yaw_from_box(bbox1)
    yaw2 = get_yaw_from_box(bbox2)
    
    # Get box corners
    corners1 = get_box_corners(bbox1['position'], bbox1['dimensions'], yaw1)
    corners2 = get_box_corners(bbox2['position'], bbox2['dimensions'], yaw2)
    
    # Compute IoU
    iou = compute_iou(corners1, corners2)
    
    # Check for overlap using Separating Axis Theorem (SAT)
    def project_corners(corners, axis):
        return np.dot(corners, axis)
    
    def check_separation(corners1, corners2, axis):
        proj1 = project_corners(corners1, axis)
        proj2 = project_corners(corners2, axis)
        return (np.max(proj1) < np.min(proj2)) or (np.max(proj2) < np.min(proj1))
    
    # Get normals for each edge of box1
    edges1 = np.roll(corners1, -1, axis=0) - corners1
    normals1 = np.array([-edges1[:, 1], edges1[:, 0]]).T
    
    # Get normals for each edge of box2
    edges2 = np.roll(corners2, -1, axis=0) - corners2
    normals2 = np.array([-edges2[:, 1], edges2[:, 0]]).T
    
    # Check for separation along all axes
    for normal in normals1:
        if check_separation(corners1, corners2, normal):
            return False, False, False, 0.0
    
    for normal in normals2:
        if check_separation(corners1, corners2, normal):
            return False, False, False, 0.0
    
    # If no separation found, boxes overlap
    overlap = True
    
    # Check centroid containment
    def point_in_rotated_box(point, box_corners):
        # Use ray casting algorithm
        inside = False
        for i in range(len(box_corners)):
            j = (i + 1) % len(box_corners)
            if ((box_corners[i][1] > point[1]) != (box_corners[j][1] > point[1])) and \
               (point[0] < (box_corners[j][0] - box_corners[i][0]) * (point[1] - box_corners[i][1]) / 
                (box_corners[j][1] - box_corners[i][1]) + box_corners[i][0]):
                inside = not inside
        return inside
    
    centroid1 = bbox1['position'][:2]
    centroid2 = bbox2['position'][:2]
    
    centroid1_in_bbox2 = point_in_rotated_box(centroid1, corners2)
    centroid2_in_bbox1 = point_in_rotated_box(centroid2, corners1)
    
    strict_accuracy = centroid1_in_bbox2 and centroid2_in_bbox1
    relaxed_accuracy = centroid1_in_bbox2 or centroid2_in_bbox1
    
    # Print comparison information
    print(f"\nComparing boxes:")
    print(f"  Box 1 (ID: {bbox1['id']}):")
    print(f"    Position: {bbox1['position']}")
    print(f"    Dimensions: {bbox1['dimensions']}")
    print(f"    Yaw: {np.degrees(yaw1):.2f}°")
    print(f"  Box 2 (ID: {bbox2['id']}):")
    print(f"    Position: {bbox2['position']}")
    print(f"    Dimensions: {bbox2['dimensions']}")
    print(f"    Yaw: {np.degrees(yaw2):.2f}°")
    print(f"  Overlap: {'Yes' if overlap else 'No'}")
    print(f"  IoU: {iou:.3f}")
    print(f"  Centroid 1 in Box 2: {'Yes' if centroid1_in_bbox2 else 'No'}")
    print(f"  Centroid 2 in Box 1: {'Yes' if centroid2_in_bbox1 else 'No'}")
    print(f"  Strict Accuracy: {'Yes' if strict_accuracy else 'No'}")
    print(f"  Relaxed Accuracy: {'Yes' if relaxed_accuracy else 'No'}")
    
    return overlap, strict_accuracy, relaxed_accuracy, iou

def evaluate_bboxes(clio_data, supervisely_data):
    """
    Evaluate bounding boxes between CLIO (estimated) and Supervisely (ground truth) formats
    using the MIT Clio evaluation methodology.
    
    Args:
        clio_data (dict): Parsed CLIO data (estimated objects)
        supervisely_data (dict): Parsed Supervisely data (ground truth)
        
    Returns:
        dict: Evaluation metrics including box counts per label and overlap details
    """
    # Initialize metrics
    metrics = {
        'total_estimated_boxes': 0,
        'total_ground_truth_boxes': 0,
        'strict_matches': 0,  # Number of strict matches
        'relaxed_matches': 0,  # Number of relaxed matches
        'relevant_detections': 0,  # Number of detections with cosine similarity >= 0.9
        'boxes_per_label': {
            'clio': {},
            'ground_truth': {}
        },
        'overlap_details': [],
        'unmatched_ground_truth_details': [],
        'iou_values': [],
    }
    
    # Count boxes per label in CLIO data (estimated)
    for label, boxes in clio_data.items():
        metrics['boxes_per_label']['clio'][label] = len(boxes)
        metrics['total_estimated_boxes'] += len(boxes)
    
    # Count boxes per label in Supervisely data (ground truth)
    for label, boxes in supervisely_data.items():
        metrics['boxes_per_label']['ground_truth'][label] = len(boxes)
        metrics['total_ground_truth_boxes'] += len(boxes)
    
    # Create a set to track which ground truth objects have been matched
    matched_ground_truth = set()
    
    # Compare CLIO boxes with ground truth
    print("\nStarting CLIO box comparisons...")
    for clio_label, clio_boxes in clio_data.items():
        for clio_box in clio_boxes:
            overlap_found = False
            max_cosine_sim = 0
            best_match = None
            
            # Track if this detection is relevant (cosine similarity >= 0.9)
            is_relevant_detection = False
            
            for supervisely_label, supervisely_boxes in supervisely_data.items():
                for supervisely_box in supervisely_boxes:
                    
                    overlap, strict_acc, relaxed_acc, iou = compute_bbox_overlap(clio_box, supervisely_box)
                    if overlap:
                        overlap_found = True
                        matched_ground_truth.add((supervisely_label, supervisely_box['id']))
                        
                        # Update match counts
                        if strict_acc:
                            metrics['strict_matches'] += 1
                        if relaxed_acc:
                            metrics['relaxed_matches'] += 1
                        
                        metrics['iou_values'].append(iou)
                        
                        metrics['overlap_details'].append({
                            'estimated_id': clio_box['id'],
                            'estimated_label': clio_label,
                            'ground_truth_id': supervisely_box['id'],
                            'ground_truth_label': supervisely_label,
                            'estimated_position': clio_box['position'],
                            'ground_truth_position': supervisely_box['position'],
                            'strict_accuracy': strict_acc,
                            'relaxed_accuracy': relaxed_acc,
                            'iou': iou,
                        })
            
            # Count relevant detections
            if is_relevant_detection:
                metrics['relevant_detections'] += 1
            
            if not overlap_found:
                metrics['overlap_details'].append({
                    'estimated_id': clio_box['id'],
                    'estimated_label': clio_label,
                    'ground_truth_id': None,
                    'ground_truth_label': None,
                    'estimated_position': clio_box['position'],
                    'ground_truth_position': None,
                    'strict_accuracy': False,
                    'relaxed_accuracy': False,
                    'iou': 0.0,
                })
    
    # Find unmatched ground truth objects
    for supervisely_label, supervisely_boxes in supervisely_data.items():
        for supervisely_box in supervisely_boxes:
            if (supervisely_label, supervisely_box['id']) not in matched_ground_truth:
                metrics['unmatched_ground_truth_details'].append({
                    'ground_truth_id': supervisely_box['id'],
                    'ground_truth_label': supervisely_label,
                    'ground_truth_position': supervisely_box['position']
                })
    
    return metrics

def save_evaluation_metrics(metrics, task, output_dir='./evaluations/results'):
    """
    Save evaluation metrics to a JSON file with a timestamp.
    
    Args:
        metrics (dict): The evaluation metrics to save
        output_dir (str): Directory to save the results in
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'clio_{task}_evaluation_metrics_{timestamp}.json')
    
    # Convert numpy arrays and booleans to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Convert metrics to JSON-serializable format
    json_metrics = convert_numpy(metrics)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"\nMetrics saved to: {output_file}")
    return output_file



def main():
    # Example usage
    clio_file = './evaluations/clio/hallway_01.graphml'
    supervisely_file = './evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json'
    task = 'hallway'

    # clio_file = './evaluations/clio/lounge_0.graphml'
    # supervisely_file = './evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json'
    # task = 'lounge'

    # clio_file = './evaluations/clio/smalloffice00.graphml'
    # supervisely_file = './evaluations/supervisely/smalloffice-0_voxel_pointcloud.pcd.json'
    # task = 'small_office'

    print(f"Attempting to read CLIO file (estimated objects): {clio_file}")
    print(f"Attempting to read Supervisely file (ground truth): {supervisely_file}")
    
    try:
        # Parse all files
        print("\nParsing CLIO file (estimated objects)...")
        clio_data = parse_clio_data(clio_file)
        print("\nParsing Supervisely file (ground truth)...")
        supervisely_data = parse_cuboid_data(supervisely_file)
        
        # Continue with evaluation using parsed data
        metrics = evaluate_bboxes(clio_data, supervisely_data)
        
        # Calculate additional metrics
        overlaps_per_label = {}
        distances = []
        total_estimated_boxes = metrics['total_estimated_boxes']
        total_ground_truth_boxes = metrics['total_ground_truth_boxes']
        
        # Process USDA overlaps
        for detail in metrics['overlap_details']:
            if detail['ground_truth_id'] is not None:
                estimated_label = detail['estimated_label']
                ground_truth_label = detail['ground_truth_label']
                key = f"{estimated_label} -> {ground_truth_label}"
                overlaps_per_label[key] = overlaps_per_label.get(key, 0) + 1
                
                pos1 = detail['estimated_position']
                pos2 = detail['ground_truth_position']
                distance = np.linalg.norm(pos1 - pos2)
                distances.append(distance)
        
        # Calculate metrics according to MIT Clio methodology
        strict_accuracy = (metrics['strict_matches'] / total_ground_truth_boxes * 100) if total_ground_truth_boxes > 0 else 0
        relaxed_accuracy = (metrics['relaxed_matches'] / total_ground_truth_boxes * 100) if total_ground_truth_boxes > 0 else 0
        
        # Calculate IoU-based F1 score
        mean_iou = float(np.mean(metrics['iou_values'])) if metrics['iou_values'] else 0
        iou_f1_score = 2 * (relaxed_accuracy * mean_iou) / (relaxed_accuracy + mean_iou) if (relaxed_accuracy + mean_iou) > 0 else 0
        
        # Add additional metrics to the results
        metrics['overlap_statistics'] = {
            'strict_accuracy': strict_accuracy,
            'relaxed_accuracy': relaxed_accuracy,
            'iou_f1_score': iou_f1_score,
            'mean_iou': mean_iou,
            'median_iou': float(np.median(metrics['iou_values'])) if metrics['iou_values'] else 0,
            'min_iou': float(np.min(metrics['iou_values'])) if metrics['iou_values'] else 0,
            'max_iou': float(np.max(metrics['iou_values'])) if metrics['iou_values'] else 0,
            'average_distance': float(np.mean(distances)) if distances else 0,
            'min_distance': float(np.min(distances)) if distances else 0,
            'max_distance': float(np.max(distances)) if distances else 0,
            'overlaps_per_label': overlaps_per_label,
            'percentage_unmatched_ground_truth': (len(metrics['unmatched_ground_truth_details']) / total_ground_truth_boxes * 100) if total_ground_truth_boxes > 0 else 0
        }
        
        # Calculate and print label-mapped metrics
        mean_metrics = calculate_mean_metrics(metrics, 'clio', task)
        print_metrics(mean_metrics)

        # append mean_metrics to metrics
        metrics['mean_metrics'] = mean_metrics

        save_evaluation_metrics(metrics, task)
        


    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
