import json
import numpy as np
from ipdb import set_trace as st


def parse_cuboid_data(json_file):
    """
    Parses a JSON file containing cuboid data and organizes it into a dictionary.

    Args:
        json_file (str): The path to the JSON file.

    Returns:
        dict: A dictionary where keys are class titles and values are lists of 
              dictionaries containing 'id', 'position' (numpy array), 'rotation' (Euler angles),
              and 'dimensions' (numpy array).
    """

    with open(json_file, 'r') as f:
        data = json.load(f)

    cuboid_dict = {}

    # First, process the objects to get the class titles and their corresponding IDs
    for obj in data['objects']:
        class_title = obj['classTitle']
        obj_id = obj['id']
        if class_title not in cuboid_dict:
            cuboid_dict[class_title] = []

    # Next, process the figures to extract cuboid geometry and associate it with the object IDs
    for figure in data['figures']:
        obj_id = figure['objectId']
        geometry = figure['geometry']

        # Find the class title associated with the objectId
        class_title = None
        for obj in data['objects']:
            if obj['id'] == obj_id:
                class_title = obj['classTitle']
                break

        if class_title:
            # Convert to numpy arrays
            position = np.array([
                geometry['position']['x'],
                geometry['position']['y'],
                geometry['position']['z']
            ])
            dimensions = np.array([
                geometry['dimensions']['x'],
                geometry['dimensions']['y'],
                geometry['dimensions']['z']
            ])
            
            cuboid_dict[class_title].append({
                'id': figure['id'],
                'position': position,
                'rotation': geometry['rotation'],
                'dimensions': dimensions
            })

    return cuboid_dict



def main():
    # Example Usage
    json_file_path = './evaluations/supervisely/hallway-agh-1_dataset 2025-04-08 16-49-12_voxel_pointcloud.pcd.json'  # Replace with your JSON file path
    result = parse_cuboid_data(json_file_path)
    # print(json.dumps(result, indent=4))

    st()

if __name__ == "__main__":
    main()