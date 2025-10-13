# Build FAISS index
# See CLIP_USD_SEARCH_README.md for more details


USD_DATA_DIR=/data/SimReadyAssets/Curated_SimSearch
FAISS_INDEX_PATH=/data/FAISS/FAISS  
python3 src_Real2USD/real2usd/scripts_r2u/setup_clip_search.py --action build_index --usd_images_dir $USD_DATA_DIR --save_index_path $FAISS_INDEX_PATH

# Preprocess USD dataset
# See USD_PREPROCESS_README.md for more details
# This first step is to create a list of USD files to process
USD_LIST_PATH=src_Real2USD/real2usd/config/usd_list.txt
python3 src_Real2USD/real2usd/scripts_r2u/manage_preprocessed_data.py --action create_list --directory $USD_DATA_DIR --output $USD_LIST_PATH