#!/usr/bin/env python3
"""
Utility script for managing pre-processed USD data
Created by: Christopher Hsu, chsu8@seas.upenn.edu
Date: 2/17/25

This script provides utilities for:
1. Creating USD file lists from directories
2. Checking pre-processed data integrity
3. Managing the pre-processed dataset
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import glob


def find_usd_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all USD files in a directory"""
    usd_extensions = ['*.usd', '*.usda', '*.usdc', '*.usdz']
    usd_files = []
    
    if recursive:
        for ext in usd_extensions:
            usd_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    else:
        for ext in usd_extensions:
            usd_files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(usd_files)


def create_usd_list(directory: str, output_file: str, recursive: bool = True):
    """Create a list of USD files from a directory"""
    usd_files = find_usd_files(directory, recursive)
    
    with open(output_file, 'w') as f:
        for usd_file in usd_files:
            f.write(f"{usd_file}\n")
    
    print(f"Created USD list with {len(usd_files)} files: {output_file}")
    return usd_files


def check_preprocessed_data(data_dir: str) -> Dict:
    """Check integrity of pre-processed data"""
    data_path = Path(data_dir)
    results_file = data_path / "processing_results.json"
    
    if not results_file.exists():
        print(f"Error: No processing results found at {results_file}")
        return {}
    
    with open(results_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} pre-processed objects")
    
    # Check each object
    valid_objects = []
    missing_files = []
    corrupted_files = []
    
    for obj in metadata:
        if obj['status'] == 'success':
            usd_filename = Path(obj['usd_path']).stem
            data_file = data_path / f"{usd_filename}_id_{obj['object_id']}.pkl"
            
            if not data_file.exists():
                missing_files.append(obj)
                continue
            
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Verify data structure
                required_keys = ['usd_path', 'object_id', 'point_cloud', 'sensor_translations', 'timestamp']
                if all(key in data for key in required_keys):
                    # Check point cloud shape
                    if data['point_cloud'].shape[1] == 3:  # Should be Nx3
                        valid_objects.append(obj)
                    else:
                        corrupted_files.append(obj)
                else:
                    corrupted_files.append(obj)
                    
            except Exception as e:
                print(f"Error reading {data_file}: {e}")
                corrupted_files.append(obj)
        else:
            print(f"Object {obj['usd_path']} (ID: {obj['object_id']}) failed during processing")
    
    # Print summary
    print(f"\nData Integrity Check Summary:")
    print(f"Valid objects: {len(valid_objects)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    
    if missing_files:
        print(f"\nMissing files:")
        for obj in missing_files:
            print(f"  - {obj['usd_path']} (ID: {obj['object_id']})")
    
    if corrupted_files:
        print(f"\nCorrupted files:")
        for obj in corrupted_files:
            print(f"  - {obj['usd_path']} (ID: {obj['object_id']})")
    
    return {
        'valid_objects': valid_objects,
        'missing_files': missing_files,
        'corrupted_files': corrupted_files,
        'total_objects': len(metadata)
    }


def list_available_objects(data_dir: str):
    """List all available pre-processed objects"""
    data_path = Path(data_dir)
    results_file = data_path / "processing_results.json"
    
    if not results_file.exists():
        print(f"Error: No processing results found at {results_file}")
        return
    
    with open(results_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Available pre-processed objects ({len(metadata)} total):")
    print("-" * 80)
    
    for obj in metadata:
        status = "✓" if obj['status'] == 'success' else "✗"
        if obj['status'] == 'success':
            shape_info = f"shape: {obj['point_cloud_shape']}"
        else:
            shape_info = f"error: {obj.get('error', 'unknown')}"
        
        print(f"{status} {obj['usd_path']} (ID: {obj['object_id']}) - {shape_info}")


def create_sample_usd_list():
    """Create a sample USD list file"""
    sample_usd_files = [
        "/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Seating/Steelbook.usd",
        "/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Seating/Chair_Office.usd",
        "/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Tables/Table_Office.usd",
        "/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Storage/Cabinet_Office.usd",
        "/data/SimReadyPacks/Commercial_NVD/Assets/ArchVis/Commercial/Lighting/Lamp_Desk.usd"
    ]
    
    output_file = "sample_usd_list.txt"
    with open(output_file, 'w') as f:
        for usd_file in sample_usd_files:
            f.write(f"{usd_file}\n")
    
    print(f"Created sample USD list: {output_file}")
    print("Note: Update the paths in this file to match your actual USD file locations")


def main():
    parser = argparse.ArgumentParser(description="Manage pre-processed USD data")
    parser.add_argument("--action", choices=["create_list", "check_data", "list_objects", "sample"], 
                       required=True, help="Action to perform")
    parser.add_argument("--directory", type=str, help="Directory to scan for USD files")
    parser.add_argument("--output", type=str, help="Output file for USD list")
    parser.add_argument("--data_dir", type=str, default="/data/preprocessed_usd_data",
                       help="Directory containing pre-processed data")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Recursively search for USD files")
    
    args = parser.parse_args()
    
    if args.action == "create_list":
        if not args.directory or not args.output:
            print("Error: --directory and --output are required for create_list action")
            return
        
        create_usd_list(args.directory, args.output, args.recursive)
    
    elif args.action == "check_data":
        check_preprocessed_data(args.data_dir)
    
    elif args.action == "list_objects":
        list_available_objects(args.data_dir)
    
    elif args.action == "sample":
        create_sample_usd_list()


if __name__ == "__main__":
    main() 