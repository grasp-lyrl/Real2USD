"""
Evaluation metrics for bounding box detection with label mapping.

For each class i, we calculate:

Strict Accuracy (SA):
    SA_i = |{boxes where both centroids contained}| / |{ground truth boxes}_i|

Relaxed Accuracy (RA):
    RA_i = |{boxes where at least one centroid contained}| / |{ground truth boxes}_i|

IoU for each matched pair of boxes:
    IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)
    where B_pred is predicted box and B_gt is ground truth box

We calculate several types of mean metrics:

1. Unweighted Mean Metrics:
    - Unweighted Mean IoU: mean(IoU_values) across all classes
    - Unweighted Strict Accuracy: mean(SA_i) across all classes
    - Unweighted Relaxed Accuracy: mean(RA_i) across all classes

2. Weighted Mean Metrics (weighted by class frequency):
    - Weighted Mean IoU: ∑(w_i * mean_IoU_i) for i in classes
    - Weighted Strict Accuracy: ∑(w_i * SA_i) for i in classes
    - Weighted Relaxed Accuracy: ∑(w_i * RA_i) for i in classes
    where w_i = |{ground truth boxes}_i| / ∑|{ground truth boxes}|

3. Frequency-weighted Mean IoU (f-mIoU):
    f-mIoU = ∑(w_i * mean_IoU_i) for i in classes
    where w_i = |{ground truth boxes}_i| / ∑|{ground truth boxes}|
    and mean_IoU_i is the mean IoU for class i

F1-scores based on IoU:
    - Unweighted F1: 2 * (RA * mean_IoU) / (RA + mean_IoU)
    - Weighted F1: 2 * (weighted_RA * weighted_mean_IoU) / (weighted_RA + weighted_mean_IoU)
    where RA is relaxed accuracy and mean_IoU is mean IoU across all matches
"""

import numpy as np
from collections import defaultdict
from mappings import *





def map_labels(data, label_map):
    """
    Map labels in the data according to the provided mapping.
    
    Args:
        data (dict): Dictionary of labels to boxes or counts
        label_map (dict): Mapping from original labels to new labels
        
    Returns:
        dict: New dictionary with mapped labels
    """
    mapped_data = defaultdict(int)
    for label, value in data.items():
        new_label = label_map.get(label, label)
        if isinstance(value, (list, tuple)):
            # If value is a list of boxes, count them
            mapped_data[new_label] += len(value)
        else:
            # If value is already a count, just add it
            mapped_data[new_label] += value
    return dict(mapped_data)

def calculate_class_metrics(metrics, class_name, dataset_type, task):
    """
    Calculate metrics for a specific class.
    
    Args:
        metrics (dict): Evaluation metrics from evaluate_bboxes
        class_name (str): Name of the class to calculate metrics for
        dataset_type (str): Type of dataset ('usda' or 'clio')
        task (str): Task type ('hallway' or 'small_office')
        
    Returns:
        dict: Metrics for the specific class
    """
    class_metrics = {
        'total_estimated': 0,
        'total_ground_truth': 0,
        'strict_matches': 0,
        'relaxed_matches': 0,
        'iou_values': [],
        'strict_accuracies': [],
        'relaxed_accuracies': []
    }
    
    # Get the appropriate label maps based on task
    if task == 'hallway':
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_HALLWAY
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_HALLWAY
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_HALLWAY
    elif task == 'small_office':  # small_office
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_SMALL_OFFICE
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_SMALL_OFFICE
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_SMALL_OFFICE
    elif task == 'lounge':
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_LOUNGE
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_LOUNGE
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_LOUNGE
    
    if dataset_type == 'usda':
        # Count total boxes for this class
        class_metrics['total_estimated'] = metrics['boxes_per_label']['usda'].get(class_name, 0)
        class_metrics['total_ground_truth'] = metrics['boxes_per_label']['ground_truth'].get(class_name, 0)
    
    elif dataset_type == 'clio':
        # Count total boxes for this class
        class_metrics['total_estimated'] = metrics['boxes_per_label']['clio'].get(class_name, 0)
        class_metrics['total_ground_truth'] = metrics['boxes_per_label']['ground_truth'].get(class_name, 0)
    
    # Initialize accuracy lists with zeros for all ground truth boxes
    class_metrics['strict_accuracies'] = [0.0] * class_metrics['total_ground_truth']
    class_metrics['relaxed_accuracies'] = [0.0] * class_metrics['total_ground_truth']
    class_metrics['iou_values'] = [0.0] * class_metrics['total_ground_truth']
    
    # Create a mapping of ground truth box IDs to their indices
    gt_id_to_index = {}
    current_index = 0
    
    # First map all ground truth boxes from unmatched_ground_truth_details
    for unmatched in metrics['unmatched_ground_truth_details']:
        if ground_truth_map.get(unmatched['ground_truth_label'], unmatched['ground_truth_label']) == class_name:
            gt_id_to_index[unmatched['ground_truth_id']] = current_index
            current_index += 1
    
    # Then map all ground truth boxes from overlap_details
    for detail in metrics['overlap_details']:
        gt_id = detail['ground_truth_id']
        if gt_id not in gt_id_to_index and ground_truth_map.get(detail['ground_truth_label'], detail['ground_truth_label']) == class_name:
            gt_id_to_index[gt_id] = current_index
            current_index += 1
    
    # Track which ground truth boxes have been matched
    matched_gt_ids = set()
    
    # Process overlap details for this class
    for detail in metrics['overlap_details']:
        # Get mapped labels for both estimated and ground truth
        estimated_label = estimated_map.get(detail['estimated_label'], detail['estimated_label'])
        ground_truth_label = ground_truth_map.get(detail['ground_truth_label'], detail['ground_truth_label'])
        
        if estimated_label == class_name and ground_truth_label == class_name:
            gt_id = detail['ground_truth_id']
            if gt_id in gt_id_to_index and gt_id not in matched_gt_ids:
                matched_gt_ids.add(gt_id)
                gt_index = gt_id_to_index[gt_id]
                class_metrics['strict_matches'] += 1 if detail['strict_accuracy'] else 0
                class_metrics['relaxed_matches'] += 1 if detail['relaxed_accuracy'] else 0
                class_metrics['iou_values'][gt_index] = detail['iou']
                class_metrics['strict_accuracies'][gt_index] = 1.0 if detail['strict_accuracy'] else 0.0
                class_metrics['relaxed_accuracies'][gt_index] = 1.0 if detail['relaxed_accuracy'] else 0.0
    
    return class_metrics

def calculate_mean_metrics(metrics, dataset_type, task):
    """
    Calculate mean metrics across all classes.
    
    Args:
        metrics (dict): Evaluation metrics from evaluate_bboxes
        dataset_type (str): Type of dataset ('usda' or 'clio')
        task (str): Task type ('hallway' or 'small_office', or 'lounge')
        
    Returns:
        dict: Mean metrics including both weighted and unweighted versions
    """
    # Get the appropriate label maps based on task
    if task == 'hallway':
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_HALLWAY
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_HALLWAY
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_HALLWAY
    elif task == 'small_office':  # small_office
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_SMALL_OFFICE
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_SMALL_OFFICE
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_SMALL_OFFICE
    elif task == 'lounge':
        ground_truth_map = GROUND_TRUTH_LABEL_MAP_LOUNGE
        if dataset_type == 'usda':
            estimated_map = USDA_LABEL_MAP_LOUNGE
        else:  # clio
            estimated_map = CLIO_LABEL_MAP_LOUNGE

    if dataset_type == 'usda':
        # Map labels
        mapped_metrics = metrics.copy()
        mapped_metrics['boxes_per_label']['usda'] = map_labels(metrics['boxes_per_label']['usda'], estimated_map)
        mapped_metrics['boxes_per_label']['ground_truth'] = map_labels(metrics['boxes_per_label']['ground_truth'], ground_truth_map)
    
        # Calculate metrics for each class
        class_metrics = {}
        for class_name in set(list(mapped_metrics['boxes_per_label']['usda'].keys()) + 
                            list(mapped_metrics['boxes_per_label']['ground_truth'].keys())):
            class_metrics[class_name] = calculate_class_metrics(mapped_metrics, class_name, dataset_type, task)
        
        # Calculate mean metrics
        mean_metrics = {
            'unweighted_mean_iou': 0.0,
            'weighted_mean_iou': 0.0,
            'frequency_weighted_iou': 0.0,
            'unweighted_strict_accuracy': 0.0,
            'weighted_strict_accuracy': 0.0,
            'unweighted_relaxed_accuracy': 0.0,
            'weighted_relaxed_accuracy': 0.0,
            'class_metrics': class_metrics,
            'per_class_metrics': {}  # New field for per-class mean metrics
        }
    elif dataset_type == 'clio':
        # Map labels
        mapped_metrics = metrics.copy()
        mapped_metrics['boxes_per_label']['clio'] = map_labels(metrics['boxes_per_label']['clio'], estimated_map)
        mapped_metrics['boxes_per_label']['ground_truth'] = map_labels(metrics['boxes_per_label']['ground_truth'], ground_truth_map)
        
        # Calculate metrics for each class
        class_metrics = {}
        for class_name in set(list(mapped_metrics['boxes_per_label']['clio'].keys()) + 
                            list(mapped_metrics['boxes_per_label']['ground_truth'].keys())):
            class_metrics[class_name] = calculate_class_metrics(mapped_metrics, class_name, dataset_type, task)
        
        # Calculate mean metrics
        mean_metrics = {
            'unweighted_mean_iou': 0.0,
            'weighted_mean_iou': 0.0,
            'frequency_weighted_iou': 0.0,
            'unweighted_strict_accuracy': 0.0,
            'weighted_strict_accuracy': 0.0,
            'unweighted_relaxed_accuracy': 0.0,
            'weighted_relaxed_accuracy': 0.0,
            'class_metrics': class_metrics,
            'per_class_metrics': {}  # New field for per-class mean metrics
        }
    
    total_ground_truth = sum(m['total_ground_truth'] for m in class_metrics.values())
    total_classes = len(class_metrics)
    
    # Initialize lists for unweighted metrics
    all_iou_values = []
    all_strict_accuracies = []
    all_relaxed_accuracies = []
    
    # Calculate per-class mean metrics
    for class_name, metrics in class_metrics.items():
        if metrics['total_ground_truth'] > 0:
            # Calculate class-specific metrics
            class_iou = np.mean(metrics['iou_values']) if metrics['iou_values'] else 0.0
            class_strict_acc = np.mean(metrics['strict_accuracies']) if metrics['strict_accuracies'] else 0.0
            class_relaxed_acc = np.mean(metrics['relaxed_accuracies']) if metrics['relaxed_accuracies'] else 0.0
            
            # Store per-class mean metrics
            mean_metrics['per_class_metrics'][class_name] = {
                'mean_iou': float(class_iou),
                'mean_strict_accuracy': float(class_strict_acc),
                'mean_relaxed_accuracy': float(class_relaxed_acc)
            }
            
            # Collect values for unweighted metrics
            all_iou_values.append(class_iou)
            all_strict_accuracies.append(class_strict_acc)
            all_relaxed_accuracies.append(class_relaxed_acc)
            
            # Update weighted metrics
            weight = metrics['total_ground_truth'] / total_ground_truth
            mean_metrics['weighted_strict_accuracy'] += class_strict_acc * weight
            mean_metrics['weighted_relaxed_accuracy'] += class_relaxed_acc * weight
            mean_metrics['weighted_mean_iou'] += class_iou * weight
            mean_metrics['frequency_weighted_iou'] += class_iou * weight
    
    # Calculate unweighted metrics
    mean_metrics['unweighted_mean_iou'] = np.mean(all_iou_values) if all_iou_values else 0.0
    mean_metrics['unweighted_strict_accuracy'] = np.mean(all_strict_accuracies) if all_strict_accuracies else 0.0
    mean_metrics['unweighted_relaxed_accuracy'] = np.mean(all_relaxed_accuracies) if all_relaxed_accuracies else 0.0

    # Calculate F1 scores
    mean_metrics['unweighted_f1'] = 2 * (mean_metrics['unweighted_relaxed_accuracy'] * mean_metrics['unweighted_mean_iou']) / \
                   (mean_metrics['unweighted_relaxed_accuracy'] + mean_metrics['unweighted_mean_iou']) \
                   if (mean_metrics['unweighted_relaxed_accuracy'] + mean_metrics['unweighted_mean_iou']) > 0 else 0
    mean_metrics['weighted_f1'] = 2 * (mean_metrics['weighted_relaxed_accuracy'] * mean_metrics['weighted_mean_iou']) / \
                 (mean_metrics['weighted_relaxed_accuracy'] + mean_metrics['weighted_mean_iou']) \
                 if (mean_metrics['weighted_relaxed_accuracy'] + mean_metrics['weighted_mean_iou']) > 0 else 0

    return mean_metrics

def print_metrics(mean_metrics):
    """
    Print the calculated metrics in a readable format.
    
    Args:
        mean_metrics (dict): Mean metrics from calculate_mean_metrics
    """
    print("\n=== Label-Mapped Metrics ===")
    print(f"\nOverall Metrics:")
    print(f"  Weighted Mean IoU: {mean_metrics['weighted_mean_iou']:.3f}")
    print(f"  Unweighted Mean IoU: {mean_metrics['unweighted_mean_iou']:.3f}")
    print(f"  Frequency-Weighted Mean IoU: {mean_metrics['frequency_weighted_iou']:.3f}")
    print(f"  Unweighted Strict Accuracy: {mean_metrics['unweighted_strict_accuracy']:.3f}")
    print(f"  Weighted Strict Accuracy: {mean_metrics['weighted_strict_accuracy']:.3f}")
    print(f"  Unweighted Relaxed Accuracy: {mean_metrics['unweighted_relaxed_accuracy']:.3f}")
    print(f"  Weighted Relaxed Accuracy: {mean_metrics['weighted_relaxed_accuracy']:.3f}")
    print(f"  Unweighted F1 Score: {mean_metrics['unweighted_f1']:.3f}")
    print(f"  Weighted F1 Score: {mean_metrics['weighted_f1']:.3f}")

    
    print("\nClass-Specific Metrics:")
    for class_name, metrics in mean_metrics['class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  Total Estimated: {metrics['total_estimated']}")
        print(f"  Total Ground Truth: {metrics['total_ground_truth']}")
        print(f"  Strict Matches: {metrics['strict_matches']}")
        print(f"  Relaxed Matches: {metrics['relaxed_matches']}")
        if metrics['iou_values']:
            print(f"  Mean IoU: {np.mean(metrics['iou_values']):.3f}")
            print(f"  Median IoU: {np.median(metrics['iou_values']):.3f}")
        if metrics['strict_accuracies']:
            print(f"  Mean Strict Accuracy: {np.mean(metrics['strict_accuracies']):.3f}")
        if metrics['relaxed_accuracies']:
            print(f"  Mean Relaxed Accuracy: {np.mean(metrics['relaxed_accuracies']):.3f}") 