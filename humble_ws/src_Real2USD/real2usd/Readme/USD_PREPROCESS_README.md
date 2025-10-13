# USD Dataset Pre-processing System

This system allows you to pre-process USD objects to generate point cloud data, eliminating the need for real-time simulation during the full pipeline execution.

## Overview

The pre-processing system consists of three main components:

1. **`preprocess_usd_dataset.py`** - Main script to process USD objects and generate point cloud data
2. **`isaac_lidar_node_preprocessed.py`** - Modified node that uses pre-processed data instead of real-time simulation
3. **`manage_preprocessed_data.py`** - Utility script for managing pre-processed data

## Workflow

### Step 1: Prepare USD File List

First, create a list of USD files to process:

# Or create a list from a directory
python manage_preprocessed_data.py --action create_list --directory /data/SimReadyAssets/Curated_SimSearch --output usd_list.txt
```

# and then i filtered out the files I did not want to include, i.e. the ref usd of the main usd.

### Step 2: Pre-process USD Dataset

Run the pre-processing script to generate point cloud data:

```bash
source humble/install/setup.bash

~/isaacsim/python.sh humble_ws/src_Real2USD/scripts_isaacsim/preprocess_usd_dataset.py --usd_list humble_ws/src_Real2USD/real2usd/config/usd_list.txt --output_dir /data/preprocessed_usd_data --start_id 0
```

This will:
- Load each USD object into Isaac Sim
- Generate point cloud data using the same LiDAR setup as `multi_rtx_lidar_standalone_node.py`
- Save the data to pickle files
- Create a metadata file with processing results

### Step 3: Use Pre-processed Data in Pipeline

Replace the real-time simulation nodes with the pre-processed version:

```bash
# Instead of running multi_rtx_lidar_standalone_node.py and isaac_lidar_node.py
# Use the pre-processed version:
ros2 run real2usd isaac_lidar_node_preprocessed
```