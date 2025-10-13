import re
import json
import numpy as np
from scipy.spatial.transform import Rotation
import os

def quaternion_to_euler(orientation):
    """
    Convert quaternion orientation to Euler angles in radians, but only consider yaw rotation.
    
    Args:
        orientation (list): Quaternion [x, y, z, w]
        
    Returns:
        float: Yaw angle in radians (third Euler angle)
    """
    rot = Rotation.from_quat(orientation)
    euler = rot.as_euler('xyz')
    # Only use the yaw component (third Euler angle)
    return euler[0]

def parse_usda_data(usda_file, target_object_id=None):
    """
    Parses a .usda file and extracts object information.

    Args:
        usda_file (str): The path to the .usda file.
        target_object_id (str, optional): If specified, only returns data for this object ID.

    Returns:
        dict: A dictionary containing all objects, organized by semantic label
    """

    with open(usda_file, 'r') as f:
        data = f.read()

    # Pre-compile all regex patterns
    object_pattern = re.compile(r'def Xform "object_(\d+)"\s*\([^)]*\)\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', re.DOTALL)
    
    # Property patterns
    semantic_pattern = re.compile(r'string semantic:Semantics:params:semanticData = "([^"]*)"')
    orient_pattern = re.compile(r'quatd?\s*xformOp:orient\s*=\s*\(([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+)\)')
    translate_pattern = re.compile(r'double3 xformOp:translate = \(([\d\.\-]+), ([\d\.\-]+), ([\d\.\-]+)\)')
    bbox_max_pattern = re.compile(r'custom float3 boundingBoxMax = \(([\d\.\-]+), ([\d\.\-]+), ([\d\.\-]+)\)')
    bbox_min_pattern = re.compile(r'custom float3 boundingBoxMin = \(([\d\.\-]+), ([\d\.\-]+), ([\d\.\-]+)\)')

    result = {}

    # Find all object definitions in the file
    all_object_defs = re.finditer(r'def Xform "object_(\d+)"\s*\([^)]*\)\s*\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', data, re.DOTALL)
    
    for match in all_object_defs:
        obj_num = match.group(1)
        obj_content = match.group(2)
        
        if target_object_id and obj_num != target_object_id:
            continue

        # Extract object properties
        semantic_match = semantic_pattern.search(obj_content)
        orient_match = orient_pattern.search(obj_content)
        translate_match = translate_pattern.search(obj_content)
        bbox_max_match = bbox_max_pattern.search(obj_content)
        bbox_min_match = bbox_min_pattern.search(obj_content)

        if all([semantic_match, orient_match, translate_match, bbox_max_match, bbox_min_match]):
            semantic_label = semantic_match.group(1)
            orientation = [float(x) for x in orient_match.groups()]
            position = [float(x) for x in translate_match.groups()]
            bbox_max = np.array([float(x) for x in bbox_max_match.groups()])
            bbox_min = np.array([float(x) for x in bbox_min_match.groups()])
            dimensions = bbox_max - bbox_min
            yaw = quaternion_to_euler(orientation)

            if semantic_label not in result:
                result[semantic_label] = []

            result[semantic_label].append({
                'id': obj_num,
                'position': position,
                'orientation': orientation,
                'yaw': yaw,
                'dimensions': dimensions.tolist(),
                'bbox_max': bbox_max.tolist(),
                'bbox_min': bbox_min.tolist()
            })
            print(f"Found object: {obj_num} with label {semantic_label}")
        else:
            missing_props = []
            if not semantic_match:
                missing_props.append("semantic")
            if not orient_match:
                missing_props.append("orientation")
                print(f"Debug - Object {obj_num} content:")
                print(obj_content)
                print(f"Debug - Orientation pattern: {orient_pattern.pattern}")
                print(f"Debug - Found orientation: {orient_pattern.findall(obj_content)}")
            if not translate_match:
                missing_props.append("translation")
            if not bbox_max_match:
                missing_props.append("bounding box max")
            if not bbox_min_match:
                missing_props.append("bounding box min")
            print(f"Object {obj_num} skipped - missing properties: {', '.join(missing_props)}")

    return result

def main():
    # Example Usage
    # usda_file_path = './evaluations/usda/smalloffice0_20250414.usda'  # Replace with your .usda file path
    usda_file_path = './evaluations/usda/hallway1_20250419.usda'
    # usda_file_path = './evaluations/usda/lounge-0_20250414.usda'
    result = parse_usda_data(usda_file_path)
    
    # Print all objects for verification
    print("All objects found:")
    for semantic_label, objects in result.items():
        for obj in objects:
            print(f"\nObject ID: {obj['id']}")
            print(f"Semantic Label: {semantic_label}")
            print(f"Position: {obj['position']}")
            print(f"Orientation: {obj['orientation']}")
            print(f"Yaw: {obj['yaw']} radians ({np.degrees(obj['yaw'])} degrees)")
            print(f"Dimensions: {obj['dimensions']}")

    # Create parsed directory if it doesn't exist
    parsed_dir = './evaluations/usda/parsed'
    os.makedirs(parsed_dir, exist_ok=True)

    # Save the parsed dictionary to a text file
    usda_filename = os.path.basename(usda_file_path)
    base_filename = os.path.splitext(usda_filename)[0]
    
    # Save objects data
    objects_output = os.path.join(parsed_dir, f"{base_filename}_objects.txt")
    with open(objects_output, 'w') as f:
        json.dump(result, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f"\nObjects data has been saved to {objects_output}")

if __name__ == "__main__":
    main()