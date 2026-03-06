import networkx as nx
import numpy as np
import ast

def parse_clio_data(graphml_file):
    """
    Parses a GraphML file containing object data and organizes it into a dictionary.

    Args:
        graphml_file (str): The path to the GraphML file.

    Returns:
        dict: A dictionary where keys are semantic labels and values are lists of 
              dictionaries containing 'id', 'position' (numpy array), 'rotation' (from bbox_orientation),
              and 'dimensions' (numpy array).
    """
    # Read the GraphML file using networkx
    G = nx.read_graphml(graphml_file)
    
    cuboid_dict = {}
    
    # Iterate through nodes to find objects
    for node in G.nodes(data=True):
        node_id, data = node
        
        # Skip non-object nodes (like rooms)
        if data.get('node_type') != 'object':
            continue
            
        # Get the semantic label (task name)
        semantic_label = data.get('name', '').replace('find the ', '')
        if not semantic_label:
            continue
            
        if semantic_label not in cuboid_dict:
            cuboid_dict[semantic_label] = []
            
        # Parse position
        position_str = data.get('position', '')
        if position_str:
            position = np.array(ast.literal_eval(position_str))
        else:
            continue
            
        # Parse dimensions
        dimensions_str = data.get('bbox_dim', '')
        if dimensions_str:
            dimensions = np.array(ast.literal_eval(dimensions_str))
        else:
            continue
            
        # Parse orientation (rotation matrix)
        orientation_str = data.get('bbox_orientation', '')
        if orientation_str:
            orientation = ast.literal_eval(orientation_str)
        else:
            orientation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # Identity matrix
            
        cuboid_dict[semantic_label].append({
            'id': data.get('obj_id', ''),
            'position': position,
            'rotation': orientation,
            'dimensions': dimensions
        })
    
    return cuboid_dict

def main():
    # Example Usage
    graphml_file_path = './evaluations/clio/hallway_rs_01.graphml'  # Replace with your GraphML file path
    result = parse_clio_data(graphml_file_path)
    
    # Print results for verification
    for semantic_label, objects in result.items():
        print(f"\nSemantic Label: {semantic_label}")
        for obj in objects:
            print(f"\nObject ID: {obj['id']}")
            print(f"Position: {obj['position']}")
            print(f"Rotation: {obj['rotation']}")
            print(f"Dimensions: {obj['dimensions']}")

if __name__ == "__main__":
    main()