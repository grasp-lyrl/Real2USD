"""
Resolve paths from the current run JSON written by real2sam3d launch when use_run_subdir=true.
Other scripts (worker, indexer) can use --use-current-run to read queue_dir and faiss_index_path
without passing paths manually.
"""
import json
import os
from pathlib import Path


def get_current_run_path():
    """Path to current_run.json (base queue dir)."""
    base = os.environ.get("SAM3D_QUEUE_BASE", "/data/sam3d_queue")
    return Path(base).resolve() / "current_run.json"


def load_current_run():
    """
    Load current_run.json. Returns dict with queue_dir, base_dir, faiss_index_path, created_at, run_name
    or None if file missing/invalid.
    """
    path = get_current_run_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data.get("queue_dir"):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def resolve_queue_and_index(use_current_run: bool, queue_dir_arg: str, index_path_arg: str = None):
    """
    If use_current_run, load from JSON and return (queue_dir Path, index_path Path or None).
    Otherwise return (Path(queue_dir_arg), Path(index_path_arg) if index_path_arg else None).
    Exits with message if use_current_run but JSON missing.
    """
    if not use_current_run:
        q = Path(queue_dir_arg).resolve()
        idx = Path(index_path_arg).resolve() if index_path_arg else None
        return q, idx
    data = load_current_run()
    if not data:
        print("Error: current_run.json not found. Start the pipeline (ros2 launch) first so it creates the run dir and writes current_run.json.", file=__import__("sys").stderr)
        print(f"  Looked at: {get_current_run_path()}", file=__import__("sys").stderr)
        print("  Use --no-current-run and pass --queue-dir (and --index-path for indexer) to run without current run.", file=__import__("sys").stderr)
        __import__("sys").exit(1)
    q = Path(data["queue_dir"]).resolve()
    idx = Path(data["faiss_index_path"]).resolve() if data.get("faiss_index_path") else None
    print(f"Using current run: queue_dir={q} (from {get_current_run_path()})", file=__import__("sys").stderr)
    return q, idx
