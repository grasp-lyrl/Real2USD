#!/usr/bin/env python3
"""
Setup script for CLIP-based USD search replacement.
This script helps create the initial directory structure and provides utilities
for building the FAISS index from USD object images.
"""

import os
import argparse
from pathlib import Path
import json
# from humble_ws.src_whatchanged.world2usd.scripts_w2u.clipusdsearch_cls import CLIPUSDSearch
from clipusdsearch_cls import CLIPUSDSearch



def build_index_from_directory(usd_images_dir: str, save_index_path: str = None):
    """
    Build FAISS index from USD images directory.
    
    Args:
        usd_images_dir: Directory containing USD object images
        save_index_path: Path to save the built index (optional)
    """
    print(f"Building FAISS index from {usd_images_dir}")
    
    # Initialize CLIP search (progress bar is handled inside the class)
    clip_search = CLIPUSDSearch()
    clip_search.build_index(usd_images_dir)
    
    if clip_search.index is not None:
        print(f"✓ Successfully built index with {len(clip_search.image_paths)} images")
        
        # Save index if path provided
        if save_index_path:
            print(f"Saving index to {save_index_path}...")
            clip_search.save_index(save_index_path)
            print(f"✓ Index saved to {save_index_path}")
    else:
        print("✗ Failed to build index. Please check the directory structure and image files.")


def add_single_image(image_path: str, usd_path: str, index_path: str):
    """
    Add a single image to an existing FAISS index.
    
    Args:
        image_path: Path to the image file
        usd_path: Corresponding USD file path
        index_path: Path to the existing FAISS index
    """
    # Load existing index
    clip_search = CLIPUSDSearch("")  # Empty directory since we're loading existing index
    clip_search.load_index(index_path)
    
    # Add new image
    clip_search.add_image_to_index(image_path, usd_path)
    
    # Save updated index
    clip_search.save_index(index_path)
    print(f"Added {image_path} to index")


def main():
    parser = argparse.ArgumentParser(description="Setup CLIP-based USD search")
    parser.add_argument("--action", choices=["build_index", "add_image"], 
                       required=True, help="Action to perform")
    parser.add_argument("--usd_images_dir", default="/data/usd_images", 
                       help="Directory containing USD object images")
    parser.add_argument("--save_index_path", help="Path to save/load FAISS index")
    parser.add_argument("--image_path", help="Path to image file (for add_image action)")
    parser.add_argument("--usd_path", help="USD file path (for add_image action)")
    
    args = parser.parse_args()
    
    if args.action == "build_index":
        build_index_from_directory(args.usd_images_dir, args.save_index_path)
    elif args.action == "add_image":
        if not args.image_path or not args.usd_path or not args.save_index_path:
            print("Error: --image_path, --usd_path, and --save_index_path are required for add_image action")
            return
        add_single_image(args.image_path, args.usd_path, args.save_index_path)


if __name__ == "__main__":
    main() 