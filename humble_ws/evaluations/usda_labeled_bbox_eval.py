import numpy as np
from scipy.spatial.transform import Rotation
from parse_usda import parse_usda_data
from parse_supervisely_bbox_json import parse_cuboid_data
import json
import os
from datetime import datetime
from ipdb import set_trace as st
from shapely.geometry import Polygon
from scipy.spatial.distance import cosine
from label_mapped_metrics import calculate_mean_metrics, print_metrics


def compute_bbox_overlap(bbox1, bbox2):
    """
    Compute if two axis-aligned bounding boxes overlap in 2D (XY plane).
    
    Args:
        bbox1 (dict): First bounding box (USDA format with bbox_min/max)
        bbox2 (dict): Second bounding box (Supervisely format with position/dimensions)
        
    Returns:
        tuple: (overlap, strict_accuracy, relaxed_accuracy, iou)
            - overlap: True if boxes overlap in 2D
            - strict_accuracy: True if both centroids are contained in each other's boxes
            - relaxed_accuracy: True if at least one centroid is contained in the other's box
            - iou: Intersection over Union value
    """
    
    # For USDA boxes, use bbox_min and bbox_max directly
    if 'bbox_min' in bbox1 and 'bbox_max' in bbox1:
        # USDA format - use axis-aligned bounding box
        min1 = np.array(bbox1['bbox_min'][:2])  # Only x,y coordinates
        max1 = np.array(bbox1['bbox_max'][:2])
        centroid1 = (min1 + max1) / 2
    else:
        # Supervisely format - compute axis-aligned bounding box from position and dimensions
        pos1 = np.array(bbox1['position'][:2])
        dim1 = np.array(bbox1['dimensions'][:2])
        min1 = pos1 - dim1 / 2
        max1 = pos1 + dim1 / 2
        centroid1 = pos1
    
    # For Supervisely boxes, compute axis-aligned bounding box
    if 'bbox_min' in bbox2 and 'bbox_max' in bbox2:
        # USDA format - use axis-aligned bounding box
        min2 = np.array(bbox2['bbox_min'][:2])  # Only x,y coordinates
        max2 = np.array(bbox2['bbox_max'][:2])
        centroid2 = (min2 + max2) / 2
    else:
        # Supervisely format - compute axis-aligned bounding box from position and dimensions
        pos2 = np.array(bbox2['position'][:2])
        dim2 = np.array(bbox2['dimensions'][:2])
        min2 = pos2 - dim2 / 2
        max2 = pos2 + dim2 / 2
        centroid2 = pos2
    
    # Compute axis-aligned bounding box IoU
    def compute_axis_aligned_iou(min1, max1, min2, max2):
        # Calculate intersection
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        # Check if there's an intersection
        if np.any(intersection_min >= intersection_max):
            return 0.0
        
        # Calculate intersection area
        intersection_area = np.prod(intersection_max - intersection_min)
        
        # Calculate union area
        area1 = np.prod(max1 - min1)
        area2 = np.prod(max2 - min2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    # Compute IoU
    iou = compute_axis_aligned_iou(min1, max1, min2, max2)
    
    # Check for overlap (simple axis-aligned overlap check)
    overlap = not (max1[0] < min2[0] or max2[0] < min1[0] or 
                   max1[1] < min2[1] or max2[1] < min1[1])
    
    # Check centroid containment for axis-aligned boxes
    def point_in_axis_aligned_box(point, box_min, box_max):
        return (box_min[0] <= point[0] <= box_max[0] and 
                box_min[1] <= point[1] <= box_max[1])
    
    centroid1_in_bbox2 = point_in_axis_aligned_box(centroid1, min2, max2)
    centroid2_in_bbox1 = point_in_axis_aligned_box(centroid2, min1, max1)
    
    strict_accuracy = centroid1_in_bbox2 and centroid2_in_bbox1
    relaxed_accuracy = centroid1_in_bbox2 or centroid2_in_bbox1
    
    # Print comparison information
    print(f"\nComparing boxes:")
    print(f"  Box 1 (ID: {bbox1['id']}):")
    if 'bbox_min' in bbox1:
        print(f"    BBox Min: {bbox1['bbox_min']}")
        print(f"    BBox Max: {bbox1['bbox_max']}")
    else:
        print(f"    Position: {bbox1['position']}")
        print(f"    Dimensions: {bbox1['dimensions']}")
    print(f"  Box 2 (ID: {bbox2['id']}):")
    if 'bbox_min' in bbox2:
        print(f"    BBox Min: {bbox2['bbox_min']}")
        print(f"    BBox Max: {bbox2['bbox_max']}")
    else:
        print(f"    Position: {bbox2['position']}")
        print(f"    Dimensions: {bbox2['dimensions']}")
    print(f"  Overlap: {'Yes' if overlap else 'No'}")
    print(f"  IoU: {iou:.3f}")
    print(f"  Centroid 1 in Box 2: {'Yes' if centroid1_in_bbox2 else 'No'}")
    print(f"  Centroid 2 in Box 1: {'Yes' if centroid2_in_bbox1 else 'No'}")
    print(f"  Strict Accuracy: {'Yes' if strict_accuracy else 'No'}")
    print(f"  Relaxed Accuracy: {'Yes' if relaxed_accuracy else 'No'}")
    
    return overlap, strict_accuracy, relaxed_accuracy, iou


def evaluate_bboxes(usda_data, supervisely_data):
    """
    Evaluate bounding boxes between USDA (estimated) and Supervisely (ground truth) formats
    using the MIT Clio evaluation methodology.
    
    Args:
        usda_file (str): Path to USDA file (estimated objects)
        supervisely_file (str): Path to Supervisely JSON file (ground truth)
        
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
            'usda': {},
            'ground_truth': {}
        },
        'overlap_details': [],
        'unmatched_ground_truth_details': [],
        'iou_values': [],
    }
    
    # Count boxes per label in USDA data (estimated)
    for label, boxes in usda_data.items():
        metrics['boxes_per_label']['usda'][label] = len(boxes)
        metrics['total_estimated_boxes'] += len(boxes)
    
    # Count boxes per label in Supervisely data (ground truth)
    for label, boxes in supervisely_data.items():
        metrics['boxes_per_label']['ground_truth'][label] = len(boxes)
        metrics['total_ground_truth_boxes'] += len(boxes)
    
    # Create a set to track which ground truth objects have been matched
    matched_ground_truth = set()
    
    # Compare USDA boxes with ground truth
    print("\nStarting USDA box comparisons...")
    for usda_label, usda_boxes in usda_data.items():
        for usda_box in usda_boxes:
            overlap_found = False
            
            # Track if this detection is relevant (cosine similarity >= 0.9)
            is_relevant_detection = False
            
            for supervisely_label, supervisely_boxes in supervisely_data.items():
                for supervisely_box in supervisely_boxes:
                    
                    overlap, strict_acc, relaxed_acc, iou = compute_bbox_overlap(usda_box, supervisely_box)
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
                            'estimated_id': usda_box['id'],
                            'estimated_label': usda_label,
                            'ground_truth_id': supervisely_box['id'],
                            'ground_truth_label': supervisely_label,
                            'estimated_position': usda_box['position'],
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
                    'estimated_id': usda_box['id'],
                    'estimated_label': usda_label,
                    'ground_truth_id': None,
                    'ground_truth_label': None,
                    'estimated_position': usda_box['position'],
                    'ground_truth_position': None,
                    'strict_accuracy': 0.0,
                    'relaxed_accuracy': 0.0,
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
    output_file = os.path.join(output_dir, f'usda_{task}_evaluation_metrics_{timestamp}.json')
    
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
    # usda_file = './evaluations/usda/hallway1_20250419.usda'
    # usda_file = '/data/SimIsaacData/usda/0_5edit_hallway_01_09082025.usda'
    # supervisely_file = './evaluations/supervisely/hallway-1_voxel_pointcloud.pcd.json'
    # task = 'hallway'

    # usda_file = '/data/SimIsaacData/usda/lounge_09082025.usda'
    usda_file = '/data/SimIsaacData/usda/lounge_05_09082025.usda'
    supervisely_file = './evaluations/supervisely/lounge-0_voxel_pointcloud.pcd.json'
    task = 'lounge'

    # usda_file = '/data/SimIsaacData/usda/smalloffice0_09082025.usda'
    # usda_file = '/data/SimIsaacData/usda/smalloffice0_edit05_09082025.usda'
    # supervisely_file = './evaluations/supervisely/smalloffice-0_voxel_pointcloud.pcd.json'
    # task = 'small_office'

    print(f"Attempting to read USDA file (estimated objects): {usda_file}")
    print(f"Attempting to read Supervisely file (ground truth): {supervisely_file}")
    
    try:
        # Parse all files
        print("\nParsing USDA file (estimated objects)...")
        usda_data = parse_usda_data(usda_file)
        print("\nParsing Supervisely file (ground truth)...")
        supervisely_data = parse_cuboid_data(supervisely_file)
        
        # Continue with evaluation using parsed data
        metrics = evaluate_bboxes(usda_data, supervisely_data)
        
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
            'overlaps_per_label': overlaps_per_label,
            'percentage_unmatched_ground_truth': (len(metrics['unmatched_ground_truth_details']) / total_ground_truth_boxes * 100) if total_ground_truth_boxes > 0 else 0
        }
        
        # Calculate and print label-mapped metrics
        mean_metrics = calculate_mean_metrics(metrics, 'usda', task)
        print_metrics(mean_metrics)

        metrics['mean_metrics'] = mean_metrics
        save_evaluation_metrics(metrics, task)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
