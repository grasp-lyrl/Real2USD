# CLIP-based USD Search Replacement

This implementation replaces the unreliable NVIDIA USD Search API with a local, controllable solution using CLIP (Contrastive Language-Image Pre-training) and FAISS for similarity search.

## Features

- **Local Control**: No dependency on external APIs that can crash
- **CLIP-based Similarity**: Uses OpenAI's CLIP model for robust image similarity
- **FAISS Indexing**: Fast similarity search using Facebook's FAISS library
- **Flexible Directory Structure**: Supports various ways to organize USD object images
- **Metadata Support**: Can include JSON metadata files with USD paths
- **Easy Integration**: Drop-in replacement for the original USDSearch class

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support, install FAISS with CUDA:
```bash
pip install faiss-gpu
```

## Directory Structure

The system expects USD object images to be organized in a directory structure. Here's the recommended layout:

```
/data/SimReadyAssets/
├── Chair/
│   ├── Chair/
│   │   ├── Danny/
|   │   │   ├── Textures/
|   │   │   ├── Danny.usd
|   │   │   └── danny_1.png
|   │   │   └── Screenshot_123.png
│   │   ├── Rave/
|   │   │   ├── Rave.usd
|   │   │   └── Rave_1.png
|   │   │   └── Screenshot_123.png
│   └── Deskchair/
│   │   ├── Steelbook/
|   │   │   ├── Textures/
|   │   │   ├── steelbook.usd
|   │   │   └── steelbook_1.png
|   │   │   └── steelbook_2.png
├── Table/
│   ├── Roundtable/
│   │   ├── HallwayRoundtable/
|   │   │   ├── Davis_Mez.usd
|   │   │   └── screenshot_1.png
|   │   │   └── Screenshot_2.png
```
### 3. Build FAISS Index

```bash
python3 src_Real2USD/real2usd/scripts_r2u/setup_clip_search.py --action build_index --usd_images_dir /data/SimReadyAssets/Curated_SimSearch --save_index_path /data/FAISS/FAISS
```

## Usage

### Basic Usage

```bash
python3 src_Real2USD/real2usd/scripts_r2u/clipusdsearch_cly.py
```

### Integration with Retrieval Node

The `retrieval_node.py` has been updated to use the new CLIP-based search. The interface remains the same, as usdsearch.


## Performance

- **Index Building**: ~1-2 seconds per image (depends on GPU/CPU)
- **Search Speed**: ~1-10ms per query (depends on index size)
- **Memory Usage**: ~512 bytes per image embedding
- **Accuracy**: CLIP provides state-of-the-art image similarity

## Configuration

### Model Selection

You can choose different CLIP models:

```python
# Use a larger model for better accuracy (slower)
usd_search = CLIPUSDSearch(model_name="ViT-L/14")

# Use a smaller model for faster inference
usd_search = CLIPUSDSearch(model_name="ViT-B/32")
```

### GPU vs CPU

The system automatically detects and uses GPU if available. For better performance:

```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install GPU version of FAISS
pip install faiss-gpu
```

## Troubleshooting

### Common Issues

1. **"No image files found"**
   - Check that your directory structure is correct
   - Ensure images have supported extensions (.jpg, .png, etc.)

2. **"FAISS index not built"**
   - Run the setup script to create the index
   - Check that the USD images directory exists and contains images

3. **"CUDA out of memory"**
   - Use a smaller CLIP model (ViT-B/32 instead of ViT-L/14)
   - Process images in smaller batches
   - Use CPU instead of GPU

4. **"No similar objects found"**
   - Add more diverse images to your index
   - Check that the query image is clear and well-lit
   - Consider using a larger CLIP model

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

All method calls remain the same:
- `process_image()`
- `call_search_post_api()`

## Advanced Features

### Custom Preprocessing

You can customize image preprocessing:

```python
def custom_preprocess(image):
    # Your custom preprocessing here
    return processed_image

usd_search.preprocess = custom_preprocess
```

### Multiple Search Strategies

```python
# Search with different limits
urls, scores, images = await usd_search.call_search_post_api(
    image_string=[embedding], limit=10
)

# Filter by category (implement custom filtering)
filtered_urls = [url for url in urls if "Furniture" in url]
```

### Index Management

```python
# Save index for later use
usd_search.save_index("/data/usd_index")

# Load existing index
usd_search.load_index("/data/usd_index")
```

## Contributing

To extend the functionality:

1. Add new methods to `CLIPUSDSearch` class
2. Update the setup script for new features
3. Add tests for new functionality
4. Update this README with new features

## License

This implementation is part of the real2usd package and follows the same license terms. 