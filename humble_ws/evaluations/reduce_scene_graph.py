import json
import numpy as np
import uuid
import os
import torch
import re # Import the regular expression library
from transformers import CLIPModel, CLIPTokenizer
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CLIP Model Setup ---
# *** IMPORTANT: This setup requires 'torch' and 'transformers' libraries. ***
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

CLIP_MODEL = None
CLIP_TOKENIZER = None

def initialize_clip_model():
    """Initializes the global CLIP model and tokenizer."""
    global CLIP_MODEL, CLIP_TOKENIZER
    print(f"Loading CLIP model '{MODEL_NAME}' on device: {DEVICE}...")
    try:
        CLIP_MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading CLIP model. Please ensure you have 'torch' and 'transformers' installed.")
        print(f"Error: {e}")
        # Fallback to the simple string match if CLIP fails to load
        CLIP_MODEL = False
        CLIP_TOKENIZER = False


def get_text_embedding(text: str) -> torch.Tensor:
    """Computes the CLIP text embedding for a given string."""
    if not CLIP_MODEL:
        raise RuntimeError("CLIP Model not initialized.")

    cleaned_text = clean_label(text)
    inputs = CLIP_TOKENIZER([cleaned_text], padding=True, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Get the text features (embeddings)
        text_features = CLIP_MODEL.get_text_features(**inputs)
    
    # Normalize the features to unit length (standard for cosine similarity)
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True)


def clean_label(label: str) -> str:
    """
    Cleans labels by converting to lowercase and removing common suffixes 
    like '_1', '_2', '_A', '_B', or simple sequential numbers.
    
    Example: 'chair_1' -> 'chair'
             'table_A' -> 'table'
             'Couch 1' -> 'couch'
    """
    label = label.lower().strip()
    
    # Regex 1: Remove '_number' or ' number' at the end (e.g., 'chair_1', 'couch 2')
    # Captures: (_|\s)\d+$ 
    #   (_|\s): a preceding underscore or space
    #   \d+    : one or more digits
    #   $      : end of string
    label = re.sub(r'(_|\s)\d+$', '', label)
    
    # Regex 2: Remove '_char' or ' char' (single uppercase or lowercase letter) at the end 
    # Captures: (_|\s)[a-zA-Z]$
    # This is useful for formats like 'table_A'
    label = re.sub(r'(_|\s)[a-zA-Z]$', '', label)
    
    return label.strip()

def are_labels_similar_clip(label1: str, label2: str, similarity_threshold: float) -> bool:
    """
    Determines if two semantic labels are similar using CLIP cosine similarity.
    """
    if not CLIP_MODEL:
        # Fallback if initialization failed
        print("CLIP model not available, falling back to exact string match.")
        return label1.lower() == label2.lower()

    # 1. Get embeddings
    embedding1 = get_text_embedding(label1)
    embedding2 = get_text_embedding(label2)

    # 2. Calculate cosine similarity
    # Cosine similarity is a dot product of two normalized vectors
    similarity = torch.matmul(embedding1, embedding2.T).item()
    
    # 3. Compare with threshold
    return similarity >= similarity_threshold

# --- Scene graph format ---
# Objects-list format: {"objects": [{"id", "job_id", "data_path", "position", "orientation", "track_id", "label"}, ...]}
# Step-wise format: {step_id: {label: {"obj_in_world_info": {"aabb_bounds": [...]}, ...}}, ...}
DEFAULT_POSITION_BOX_HALF = 0.5  # half-extent (m) for synthetic AABB when only position is available


def is_objects_list_format(data: dict) -> bool:
    """True if data is {"objects": [...]} with a list of object dicts."""
    if not isinstance(data, dict) or "objects" not in data:
        return False
    objs = data["objects"]
    return isinstance(objs, list) and len(objs) > 0 and isinstance(objs[0], dict)


def position_to_aabb(position: list, half: float = None):
    """Build synthetic AABB [min, max] around a position (for IoU when no bounds in JSON)."""
    if half is None:
        half = DEFAULT_POSITION_BOX_HALF
    p = np.array(position, dtype=float)
    return [list(p - half), list(p + half)]


def get_aabb_from_object(obj: dict, is_objects_list: bool) -> list:
    """Return AABB for an object (from aabb_bounds or from position)."""
    if is_objects_list:
        return position_to_aabb(obj["position"])
    return obj["obj_in_world_info"]["aabb_bounds"]


# --- Helper Functions (From Previous Script) ---

def calculate_3d_iou(bbox1_bounds, bbox2_bounds):
    """Calculates the 3D Intersection over Union (IoU) of two AABBs."""
    min_A = np.array(bbox1_bounds[0])
    max_A = np.array(bbox1_bounds[1])
    min_B = np.array(bbox2_bounds[0])
    max_B = np.array(bbox2_bounds[1])

    intersect_min = np.maximum(min_A, min_B)
    intersect_max = np.minimum(max_A, max_B)

    intersect_dims = np.maximum(0, intersect_max - intersect_min)
    intersect_volume = np.prod(intersect_dims)

    volume_A = np.prod(max_A - min_A)
    volume_B = np.prod(max_B - min_B)

    if volume_A <= 1e-6 and volume_B <= 1e-6:
        return 1.0
    if volume_A <= 1e-6 or volume_B <= 1e-6:
        return 0.0

    union_volume = volume_A + volume_B - intersect_volume
    
    if union_volume == 0:
        return 0.0

    iou = intersect_volume / union_volume
    return iou

def check_overlaps(scene_graph: dict, iou_threshold: float, position_box_half: float = None) -> list:
    """Checks for overlaps between unique objects in the scene graph.
    Supports both step-wise format (dict id -> data with aabb_bounds) and
    objects-list format (dict with 'objects' list; bounds from position).
    """
    if position_box_half is None:
        position_box_half = DEFAULT_POSITION_BOX_HALF
    overlaps = []
    if is_objects_list_format(scene_graph):
        objs = scene_graph["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                o1, o2 = objs[i], objs[j]
                bounds1 = position_to_aabb(o1["position"], half=position_box_half)
                bounds2 = position_to_aabb(o2["position"], half=position_box_half)
                iou = calculate_3d_iou(bounds1, bounds2)
                if iou > iou_threshold:
                    overlaps.append({
                        "obj1": o1.get("id", i),
                        "obj2": o2.get("id", j),
                        "iou": iou
                    })
        return overlaps
    objects = list(scene_graph.items())
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            id1, data1 = objects[i]
            id2, data2 = objects[j]
            bounds1 = data1["obj_in_world_info"]["aabb_bounds"]
            bounds2 = data2["obj_in_world_info"]["aabb_bounds"]
            iou = calculate_3d_iou(bounds1, bounds2)
            if iou > iou_threshold:
                overlaps.append({
                    "obj1": id1,
                    "obj2": id2,
                    "iou": iou
                })
    return overlaps

# --- Main Deduplication Function (Updated to use CLIP) ---

def _flatten_objects_list(input_data: dict, position_box_half: float = None) -> list:
    """Build all_objects list from {"objects": [...]} format."""
    if position_box_half is None:
        position_box_half = DEFAULT_POSITION_BOX_HALF
    all_objects = []
    for obj in input_data["objects"]:
        position = obj["position"]
        all_objects.append({
            "original_label": obj["label"],
            "confidence": 0.0,
            "aabb_bounds": position_to_aabb(position, half=position_box_half),
            "data": dict(obj),
        })
    return all_objects


def _flatten_step_wise(input_data: dict) -> list:
    """Build all_objects list from step-wise {step_id: {label: data}} format."""
    all_objects = []
    for step_id, objects_in_step in input_data.items():
        for label, data in objects_in_step.items():
            obj_data = {
                "step_id": step_id,
                "original_label": label,
                "confidence": data.get("confidence", 0.0),
                "aabb_bounds": data["obj_in_world_info"]["aabb_bounds"],
                "data": data
            }
            all_objects.append(obj_data)
    return all_objects


def deduplicate_scene_graph(
    input_data: dict,
    iou_threshold: float,
    clip_threshold: float,
    position_box_half: float = None,
) -> dict:
    """
    Deduplicates objects based on CLIP label similarity and 3D IoU overlap.

    Supports two input formats:
    - Objects-list: {"objects": [{"id", "job_id", "data_path", "position", "orientation", "track_id", "label"}, ...]}
    - Step-wise: {step_id: {label: {"obj_in_world_info": {"aabb_bounds": [...]}, ...}}, ...}

    Output format matches input: objects-list -> {"objects": [...]}, step-wise -> {id: data}.
    position_box_half: half-extent (m) for synthetic AABB in objects-list format; uses DEFAULT_POSITION_BOX_HALF if None.
    """
    if position_box_half is None:
        position_box_half = DEFAULT_POSITION_BOX_HALF
    use_objects_list = is_objects_list_format(input_data)

    # 1. Flatten into a list of all detected objects
    if use_objects_list:
        all_objects = _flatten_objects_list(input_data, position_box_half=position_box_half)
    else:
        all_objects = _flatten_step_wise(input_data)

    # 2. Group objects into unique instances (label similarity + IoU)
    unique_groups = []
    for current_obj in all_objects:
        is_matched = False
        for group in unique_groups:
            representative = group[0]
            label_match = are_labels_similar_clip(
                current_obj["original_label"],
                representative["original_label"],
                clip_threshold
            )
            if label_match:
                iou = calculate_3d_iou(current_obj["aabb_bounds"], representative["aabb_bounds"])
                if iou >= iou_threshold:
                    group.append(current_obj)
                    if current_obj["confidence"] > representative["confidence"]:
                        group.insert(0, group.pop(-1))
                    is_matched = True
                    break
        if not is_matched:
            unique_groups.append([current_obj])

    # 3. Build output in same shape as input
    if use_objects_list:
        out_objects = []
        for group in unique_groups:
            best = group[0]["data"]
            final_label = best["label"].replace(" ", "_").lower()
            out_obj = dict(best)
            out_obj["label"] = final_label
            out_objects.append(out_obj)
        return {"objects": out_objects}
    else:
        consolidated_scene_graph = {}
        for group in unique_groups:
            best_obj = group[0]
            final_label = best_obj["original_label"].replace(" ", "_").lower()
            unique_id = f"{final_label}_{uuid.uuid4().hex[:4]}"
            consolidated_scene_graph[unique_id] = best_obj["data"]
            consolidated_scene_graph[unique_id]["label"] = final_label
        return consolidated_scene_graph

def _get_labels_from_dedup(deduplicated_data: dict) -> list:
    """Extract label list from deduplicated output (objects-list or step-wise)."""
    if is_objects_list_format(deduplicated_data):
        return [obj.get("label", "unknown") for obj in deduplicated_data["objects"]]
    return [v.get("label", k.split("_")[0]) for k, v in deduplicated_data.items()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Deduplicate a scene graph by CLIP label similarity and 3D IoU. "
        "Supports objects-list JSON (objects with id, position, orientation, label) or step-wise format."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input scene graph JSON path (default: <dataset_dir>/scene_graph.json)",
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        nargs="?",
        default=None,
        help="Dataset directory (used when input/output not given; then input=<dataset_dir>/scene_graph.json, output=<dataset_dir>/scene_graph_reduced.json)",
    )
    parser.add_argument(
        "lidar_dir",
        type=str,
        nargs="?",
        default=None,
        help="Optional lidar dir for accum_pc (only used with step-wise format)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    parser.add_argument("--iou_threshold", type=float, default=0.2, help="IoU threshold for deduplication")
    parser.add_argument("--clip_threshold", type=float, default=0.8, help="CLIP similarity threshold for deduplication")
    parser.add_argument(
        "--position_box_half",
        type=float,
        default=DEFAULT_POSITION_BOX_HALF,
        help="Half-extent (m) for synthetic AABB around position when using objects-list format (default: 0.5)",
    )
    args = parser.parse_args()

    # Resolve input/output paths
    if args.input is not None:
        input_file = args.input
        output_file = args.output or input_file.replace(".json", "_reduced.json")
        if output_file == input_file:
            output_file = input_file.rsplit(".", 1)[0] + "_reduced.json"
        dataset_dir = os.path.dirname(os.path.abspath(input_file))
    elif args.dataset_dir is not None:
        dataset_dir = args.dataset_dir
        input_file = os.path.join(dataset_dir, "scene_graph.json")
        output_file = args.output or os.path.join(dataset_dir, "scene_graph_reduced.json")
    else:
        parser.error("Provide either input JSON path or dataset_dir")

    position_box_half = args.position_box_half

    with open(input_file, "r") as f:
        scene_data = json.load(f)

    if is_objects_list_format(scene_data):
        print(f"Detected objects-list format ({len(scene_data['objects'])} objects).")
    else:
        print("Detected step-wise scene graph format.")

    initialize_clip_model()

    print(f"Deduplicating with IoU threshold: {args.iou_threshold}, CLIP threshold: {args.clip_threshold}...")
    deduplicated_data = deduplicate_scene_graph(
        scene_data, args.iou_threshold, args.clip_threshold, position_box_half=position_box_half
    )

    n_out = len(deduplicated_data["objects"]) if is_objects_list_format(deduplicated_data) else len(deduplicated_data)
    n_in = len(scene_data["objects"]) if is_objects_list_format(scene_data) else sum(len(s) for s in scene_data.values() if isinstance(s, dict))
    print(f"Reduced to {n_out} unique objects (from {n_in}).")

    with open(output_file, "w") as f:
        json.dump(deduplicated_data, f, indent=2)
    print(f"Wrote {output_file}.")

    overlaps = check_overlaps(deduplicated_data, args.iou_threshold, position_box_half=position_box_half)
    if overlaps:
        print(f"Found {len(overlaps)} overlaps > IoU threshold {args.iou_threshold}:")
        for o in overlaps:
            print(f"  {o['obj1']} and {o['obj2']}: IoU {o['iou']:.3f}")
    else:
        print("No overlaps above threshold in deduplicated scene graph.")

    if CLIP_MODEL:
        try:
            labels = _get_labels_from_dedup(deduplicated_data)
            if labels:
                embeddings = [get_text_embedding(label) for label in labels]
                emb_tensor = torch.stack(embeddings).squeeze(1).cpu().numpy()
                pca = PCA(n_components=2)
                emb_2d = pca.fit_transform(emb_tensor)
                plt.figure(figsize=(10, 8))
                for i, label in enumerate(labels):
                    plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
                    plt.annotate(label, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)
                plt.title(f"CLIP Embeddings PCA (Threshold: {args.clip_threshold})")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plot_path = os.path.join(dataset_dir, "clip_embeddings_plot.png")
                plt.savefig(plot_path)
                print(f"Saved CLIP embeddings plot to {plot_path}")
        except Exception as e:
            print(f"Failed to plot embeddings: {e}")
    else:
        print("CLIP model not available, skipping plot.")

    if args.lidar_dir and not is_objects_list_format(deduplicated_data):
        try:
            from demo_pc_rs import accum_pc
            accum_pc(dataset_dir, os.path.basename(output_file), args.lidar_dir)
        except ImportError:
            print("demo_pc_rs not found, skipping accum_pc.")