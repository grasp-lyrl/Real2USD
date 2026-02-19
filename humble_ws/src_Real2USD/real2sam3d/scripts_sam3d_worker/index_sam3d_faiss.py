#!/usr/bin/env python3
"""
FAISS indexer for SAM3D-generated objects: watches queue_dir/output/ for new job
results, adds each object's image(s) + path to a CLIP+FAISS index so they can be
retrieved by visual similarity (Phase 3 hybrid retrieval).

Uses CLIPUSDSearch from real2usd (scripts_r2u). Run where PyTorch/CLIP/FAISS are
available (e.g. host or a conda env). Same queue dir as the worker.

Usage:
  python index_sam3d_faiss.py --queue-dir /data/sam3d_queue --index-path /data/sam3d_faiss [--watch-interval 5] [--once]

--queue-dir: Base queue directory (output/ will be queue_dir/output).
--index-path: Directory for the FAISS index; a 'faiss' subdir is created here and all outputs go there (index.faiss, index.pkl, indexed_jobs.txt).
--watch-interval: Seconds between scans for new jobs (default 5).
--once: Index any unindexed jobs and exit (no continuous watch).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# CLIPUSDSearch is imported only when indexing (so tests can import helpers without CLIP/FAISS)
_script_dir = Path(__file__).resolve().parent
_repo_src = _script_dir.parents[1]  # real2sam3d -> src_Real2USD
_scripts_r2u = _repo_src / "real2usd" / "scripts_r2u"


def _get_clip_search():
    """Lazy import of CLIPUSDSearch so helpers are testable without CLIP/FAISS."""
    if _scripts_r2u.is_dir() and str(_scripts_r2u) not in sys.path:
        sys.path.insert(0, str(_scripts_r2u))
    try:
        from clipusdsearch_cls import CLIPUSDSearch
        return CLIPUSDSearch
    except ImportError as e:
        print("Error: Could not import CLIPUSDSearch. Add real2usd/scripts_r2u to PYTHONPATH. CLIP/FAISS required.", file=sys.stderr)
        raise SystemExit(1) from e


def _faiss_dir_and_paths(index_path: Path):
    """
    Resolve index_path to a directory, create faiss/ inside it, and return
    (faiss_dir, index_base, state_path). All outputs go in faiss_dir.
    """
    p = index_path.resolve()
    if p.suffix in (".faiss", ".pkl") or p.name.endswith("_indexed_jobs.txt"):
        base_dir = p.parent
    else:
        base_dir = p
    faiss_dir = base_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    index_base = faiss_dir / "index"  # index.faiss, index.pkl
    state_path = faiss_dir / "indexed_jobs.txt"
    return faiss_dir, index_base, state_path


def _load_indexed_jobs(state_path: Path) -> set:
    if not state_path.exists():
        return set()
    try:
        return set(line.strip() for line in state_path.read_text().splitlines() if line.strip())
    except Exception:
        return set()


def _save_indexed_job(state_path: Path, job_id: str) -> None:
    try:
        with open(state_path, "a") as f:
            f.write(job_id + "\n")
    except Exception as e:
        print(f"[WARN] Could not append to indexed state {state_path}: {e}", file=sys.stderr)


def _completed_job_dirs(output_dir: Path):
    """Yield (job_path, object_path) for each completed job (pose + object.ply or object.usd)."""
    if not output_dir.exists():
        return
    for job_path in output_dir.iterdir():
        if not job_path.is_dir():
            continue
        pose_path = job_path / "pose.json"
        object_ply = job_path / "object.ply"
        object_usd = job_path / "object.usd"
        if not pose_path.exists():
            continue
        object_path = None
        if object_usd.exists():
            object_path = str(object_usd.resolve())
        elif object_ply.exists():
            object_path = str(object_ply.resolve())
        if object_path is None:
            continue
        yield job_path, object_path


def _image_paths_for_job(job_path: Path):
    """Yield image paths to index for this job: views/*.png if present, else rgb.png."""
    views_dir = job_path / "views"
    if views_dir.is_dir():
        view_files = sorted(views_dir.glob("*.png"))
        if view_files:
            for p in view_files:
                yield str(p)
            return
    rgb = job_path / "rgb.png"
    if rgb.exists():
        yield str(rgb)


def index_pending_jobs(
    output_dir: Path,
    index_path: Path,
    clip_search,  # CLIPUSDSearch instance
    indexed: set,
    state_path: Path,
) -> int:
    """Index all completed jobs not yet in indexed. Returns number newly indexed."""
    count = 0
    for job_path, object_path in _completed_job_dirs(output_dir):
        job_id = job_path.name
        if job_id in indexed:
            continue
        image_paths = list(_image_paths_for_job(job_path))
        if not image_paths:
            print(f"[SKIP] {job_id}: no rgb.png or views/*.png", file=sys.stderr)
            continue
        try:
            for img_path in image_paths:
                clip_search.add_image_to_index(img_path, object_path)
            _save_indexed_job(state_path, job_id)
            indexed.add(job_id)
            count += 1
            print(f"[OK] Indexed {job_id} ({len(image_paths)} image(s))", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Indexing {job_id}: {e}", file=sys.stderr)
    if count > 0:
        try:
            clip_search.save_index(str(index_path))
            print(f"[OK] Saved index to {index_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Saving index: {e}", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description="FAISS indexer for SAM3D output")
    parser.add_argument("--queue-dir", type=str, default="/data/sam3d_queue", help="Queue base directory")
    parser.add_argument("--index-path", type=str, default="/data/sam3d_faiss", help="Directory for index; outputs go in <path>/faiss/")
    parser.add_argument("--watch-interval", type=float, default=5.0, help="Seconds between scans when watching")
    parser.add_argument("--once", action="store_true", help="Index once and exit")
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir).resolve()
    index_path_arg = Path(args.index_path).resolve()
    faiss_dir, index_base, state_path = _faiss_dir_and_paths(index_path_arg)
    output_dir = queue_dir / "output"

    indexed = _load_indexed_jobs(state_path)
    print(f"Loaded {len(indexed)} already-indexed job(s) from {state_path}", file=sys.stderr)

    # Load or create index (files: faiss/index.faiss, faiss/index.pkl)
    CLIPUSDSearch = _get_clip_search()
    clip_search = CLIPUSDSearch()
    faiss_file = Path(str(index_base) + ".faiss")
    if faiss_file.exists():
        clip_search.load_index(str(index_base))
        print(f"Loaded existing index: {index_base} ({len(clip_search.image_paths)} entries)", file=sys.stderr)
    else:
        print(f"No existing index; will create on first add in {faiss_dir}", file=sys.stderr)

    if args.once:
        n = index_pending_jobs(output_dir, index_base, clip_search, indexed, state_path)
        print(f"Indexed {n} new job(s).", file=sys.stderr)
        return

    print(f"Watching {output_dir} every {args.watch_interval}s. Index in {faiss_dir}. Ctrl+C to stop.", file=sys.stderr)
    while True:
        try:
            index_pending_jobs(output_dir, index_base, clip_search, indexed, state_path)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
