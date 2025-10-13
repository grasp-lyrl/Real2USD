import os
import cv2
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import clip
import json
import asyncio
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ipdb import set_trace as st
import seaborn as sns

sns.set_theme()
fsz = 16
plt.rc("font", size=fsz)
plt.rc("axes", titlesize=fsz)
plt.rc("axes", labelsize=fsz)
plt.rc("xtick", labelsize=fsz)
plt.rc("ytick", labelsize=fsz)
plt.rc("legend", fontsize=0.5*fsz)
plt.rc("figure", titlesize=fsz)
plt.rc("pdf", fonttype=42)
sns.set_style("ticks", rc={"axes.grid": True})

class CLIPUSDSearch:
    """
    CLIP-based USD search implementation using FAISS for similarity search.
    Replaces the NVIDIA USD Search API with a simpler Similarity Search based on CLIP embeddings.
    
    Clip model names include: ViT-B/32
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP-based USD search.
        
        Args:
            usd_images_dir: Directory containing images of USD objects with corresponding USD file paths
            model_name: CLIP model to use (default: ViT-B/32)
        """
        self.model_name = model_name
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Initialize FAISS index
        self.index = None
        self.image_paths = []
        self.usd_paths = []
        self.image_embeddings = []


        self.fig = plt.figure(figsize=(8, 4))
    
    def build_index(self, usd_images_dir: str):
        """Build FAISS index from images in the USD images directory."""
        usd_images_dir = Path(usd_images_dir)
        print(f"Building FAISS index from {usd_images_dir}")
        
        # Collect all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            for img_path in usd_images_dir.rglob(f"*{ext}"):
                if any(p.name.lower() == "textures" for p in img_path.parents):
                    continue
                image_files.append(img_path)
            for img_path in usd_images_dir.rglob(f"*{ext.upper()}"):
                if any(p.name.lower() == "textures" for p in img_path.parents):
                    continue
                image_files.append(img_path)
        
        if not image_files:
            print(f"No image files found in {usd_images_dir}")
            return
        
        print(f"Found {len(image_files)} image files")
        
        # Process images and extract embeddings with progress bar
        embeddings = []
        paths = []
        usd_paths = []
        
        print("Processing images and extracting embeddings...")
        for img_path in tqdm(image_files, desc="Building FAISS index", unit="image"):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Extract embedding
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features = F.normalize(image_features, p=2, dim=1)
                
                embeddings.append(image_features.cpu().numpy())
                paths.append(str(img_path))
                
                # Try to find corresponding USD file path
                usd_path = self._find_usd_path(img_path)
                usd_paths.append(usd_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not embeddings:
            print("No valid embeddings extracted")
            return
        
        # Stack embeddings
        embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.image_paths = paths
        self.usd_paths = usd_paths
        self.image_embeddings = embeddings
        
        print(f"Built FAISS index with {len(embeddings)} embeddings")
    
    def _find_usd_path(self, image_path: Path) -> str:
        """
        Find corresponding USD file path for an image.
        Looks for a .usd file in the same directory as the image.
        Returns the alphabetically first .usd file in the directory.
        So the one you want to label it as 0_xxx.usd .
        """
        usd_files = list(image_path.parent.glob("*.usd"))
        if usd_files:
            # Sort alphabetically to ensure consistent ordering
            usd_files.sort(key=lambda x: x.name)
            return str(usd_files[0])
        else:
            raise FileNotFoundError(f"No .usd file found in directory {image_path.parent} for image {image_path}")
    
    def process_image(self, image: np.ndarray, formatted: bool = True) -> str:
        """
        Process image for search (maintains compatibility with original interface).
        
        Args:
            image: OpenCV image (BGR format)
            formatted: Whether to return formatted string (for compatibility)
        
        Returns:
            Processed image string (for compatibility with original interface)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        height, width = image_rgb.shape[:2]
        if height > 336 or width > 336:
            image_rgb = cv2.resize(image_rgb, (336, 336))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess for CLIP
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        # Return embedding as numpy array for search
        return image_features.cpu().numpy()
    
    async def call_search_post_api(self, description: str = "", image_string: List = None, 
                                 search_path: str = "SimReadyAssets", exclude_file_name: str = "", 
                                 limit: int = 5, retrieval_mode: str = "cosine") -> Tuple[List[str], List[float], List[str], List[np.ndarray]]:
        """
        Search for similar USD objects using CLIP embeddings and FAISS or PCA distance.
        
        Args:
            description: Text description (not used in CLIP version)
            image_string: List of image embeddings
            search_path: Search path (not used in local version)
            exclude_file_name: Exclude file names (not implemented)
            limit: Maximum number of results to return
            retrieval_mode: 'cosine' (default, uses FAISS/cosine similarity) or 'pca' (uses Euclidean distance in PCA space)
        
        Returns:
            Tuple of (urls, scores, image_paths, result_images)
        """
        if self.index is None:
            print("FAISS index not built. Please ensure USD images directory exists.")
            return None, None, None, None
        
        if not image_string:
            print("No image provided for search")
            return None, None, None, None
        
        try:
            # Use the first image embedding
            query_embedding = image_string[0]
            
            urls = []
            result_scores = []
            result_image_paths = []
            result_images = []
            if retrieval_mode == "cosine":
                # Search FAISS index (cosine similarity)
                scores, indices = self.index.search(query_embedding.astype('float32'), limit)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.usd_paths):
                        usd_path = self.usd_paths[idx]
                        image_path = self.image_paths[idx]
                        urls.append(usd_path)
                        result_scores.append(float(score))
                        result_image_paths.append(image_path)
                        # Load result image
                        try:
                            result_image = cv2.imread(image_path)
                            if result_image is not None:
                                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            result_images.append(result_image)
                        except Exception as e:
                            print(f"Error loading result image {image_path}: {e}")
                            result_images.append(None)
            elif retrieval_mode == "pca":
                # Project all embeddings and query to 2D PCA space for both retrieval and visualization
                from sklearn.decomposition import PCA
                embeddings = np.array(self.image_embeddings)
                n_components = 2
                pca = PCA(n_components=n_components)
                X_proj = pca.fit_transform(embeddings)
                query_proj = pca.transform(query_embedding.reshape(1, -1))
                # Compute Euclidean distances in 2D PCA space
                dists = np.linalg.norm(X_proj - query_proj, axis=1)
                top_indices = np.argsort(dists)[:limit]
                for idx in top_indices:
                    usd_path = self.usd_paths[idx]
                    image_path = self.image_paths[idx]
                    urls.append(usd_path)
                    result_scores.append(float(-dists[idx]))  # negative distance for compatibility
                    result_image_paths.append(image_path)
                    # Load result image
                    try:
                        result_image = cv2.imread(image_path)
                        if result_image is not None:
                            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        result_images.append(result_image)
                    except Exception as e:
                        print(f"Error loading result image {image_path}: {e}")
                        result_images.append(None)
            else:
                raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")
            return urls, result_scores, result_image_paths, result_images
            
        except Exception as e:
            print(f"Error in CLIP search: {e}")
            return None, None, None, None
    
    def add_image_to_index(self, image_path: str, usd_path: str):
        """
        Add a new image to the FAISS index.
        
        Args:
            image_path: Path to the image file
            usd_path: Corresponding USD file path
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
            
            # Add to index
            if self.index is None:
                dimension = image_features.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
            
            self.index.add(image_features.cpu().numpy().astype('float32'))
            
            # Store metadata
            self.image_paths.append(image_path)
            self.usd_paths.append(usd_path)
            self.image_embeddings.append(image_features.cpu().numpy())
            
            print(f"Added {image_path} to index")
            
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
    
    def save_index(self, index_path: str):
        """Save the FAISS index and metadata."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{index_path}.faiss")
            
            # Save metadata
            metadata = {
                'image_paths': self.image_paths,
                'usd_paths': self.usd_paths,
                'image_embeddings': [emb.tolist() for emb in self.image_embeddings]
            }
            
            with open(f"{index_path}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Saved index to {index_path}")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, index_path: str):
        """Load the FAISS index and metadata."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load metadata
            with open(f"{index_path}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.image_paths = metadata['image_paths']
            self.usd_paths = metadata['usd_paths']
            self.image_embeddings = [np.array(emb) for emb in metadata['image_embeddings']]
            
            print(f"Loaded index from {index_path}")
            
        except Exception as e:
            print(f"Error loading the index. Please run setup_clip_search.py to build the index")

    def visualize_cosine_similarity(self, test_embedding, highlight_indices=None, label_mode='folder'):
        """
        Plot cosine similarity between test_embedding and all database embeddings, with labels.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        db_embeds = self.image_embeddings
        db_embeds = db_embeds / np.linalg.norm(db_embeds, axis=1, keepdims=True)
        test_emb = test_embedding / np.linalg.norm(test_embedding)
        similarities = db_embeds @ test_emb.T
        similarities = similarities.flatten()
        indices = np.arange(len(similarities))
        if label_mode == 'folder':
            labels = [Path(p).parent.name for p in self.image_paths]
        elif label_mode == 'usd':
            labels = [Path(p).name for p in self.usd_paths]
        else:
            labels = [''] * len(self.image_paths)
        self.fig.clf()
        # plt.figure(figsize=(8, 4))
        ax = self.fig.add_subplot()
        ax.scatter(indices, similarities, label='All DB Embeddings', alpha=0.6)
        if highlight_indices is not None:
            ax.scatter(highlight_indices, similarities[highlight_indices], color='red', s=80, label='Top Results')
        # Label the top 50 highest cosine similarity scores
        top_n = 50
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        for i in top_indices:
            ax.text(indices[i], similarities[i], labels[i], fontsize=7, alpha=0.9, rotation=45, color='blue')
        ax.set_xlabel('Database Index (d)')
        ax.set_ylabel('Cosine Similarity')
        # plt.title('Cosine Similarity to Test Embedding')
        # plt.legend()
        self.fig.tight_layout()
        # plt.show()
        self.fig.canvas.draw()
        plt.pause(0.001)

    def visualize_embeddings_pca(self, test_embedding=None, test_label=None, highlight_indices=None, n_components=2, color_by='folder'):
        """
        Visualize CLIP embeddings using PCA. Optionally overlay a test embedding and highlight top results.
        Args:
            test_embedding: np.ndarray shape (1, D) or (D,)
            test_label: str, label for the test embedding
            highlight_indices: list of int, indices of top results to highlight
            n_components: 2 or 3
            color_by: 'folder' or 'usd'
        """
        embeddings = self.image_embeddings
        image_paths = self.image_paths
        usd_paths = self.usd_paths
        pca = PCA(n_components=n_components)
        X_proj = pca.fit_transform(embeddings)
        if test_embedding is not None:
            test_proj = pca.transform(test_embedding.reshape(1, -1))
        # Coloring
        from pathlib import Path
        if color_by == 'folder':
            labels = [Path(p).parent.name for p in image_paths]
        elif color_by == 'usd':
            labels = [Path(p).name for p in usd_paths]
        else:
            labels = [''] * len(image_paths)
        unique_labels = list(set(labels))
        colors = sns.color_palette("tab10", len(unique_labels))
        label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]
        fig = plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(X_proj[:, 0], X_proj[:, 1], c=point_colors, alpha=0.7, label='DB Embeddings')
            # Highlight top results
            if highlight_indices is not None:
                plt.scatter(X_proj[highlight_indices, 0], X_proj[highlight_indices, 1],
                            facecolors='none', edgecolors='black', s=120, linewidths=2, label='Top Results')
            # Plot test embedding
            if test_embedding is not None:
                plt.scatter(test_proj[0, 0], test_proj[0, 1], marker='*', c='red', s=200, label='Test Image')
                if test_label:
                    plt.text(test_proj[0, 0], test_proj[0, 1], test_label, fontsize=10, color='red')
            # Add labels (downsample if too many)
            max_labels = 50
            step = max(1, len(labels)//max_labels)
            for i in range(0, len(labels), step):
                plt.text(X_proj[i, 0], X_proj[i, 1], labels[i], fontsize=7, alpha=0.7)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=point_colors, alpha=0.7)
            if highlight_indices is not None:
                ax.scatter(X_proj[highlight_indices, 0], X_proj[highlight_indices, 1], X_proj[highlight_indices, 2],
                           facecolors='none', edgecolors='black', s=120, linewidths=2, label='Top Results')
            if test_embedding is not None:
                ax.scatter(test_proj[0, 0], test_proj[0, 1], test_proj[0, 2], marker='*', c='red', s=200, label='Test Image')
                if test_label:
                    ax.text(test_proj[0, 0], test_proj[0, 1], test_proj[0, 2], test_label, fontsize=10, color='red')
            # Add labels (downsample if too many)
            max_labels = 50
            step = max(1, len(labels)//max_labels)
            for i in range(0, len(labels), step):
                ax.text(X_proj[i, 0], X_proj[i, 1], X_proj[i, 2], labels[i], fontsize=7, alpha=0.7)
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
        plt.title('CLIP Embeddings PCA Projection')
        plt.legend()
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Visualize PCA or cosine similarity projection with test image and top results')
    parser.add_argument('--mode', choices=['pca', 'cosine'], default='pca', help='Visualization mode')
    parser.add_argument('--test_image', type=str, default='/data/tests/seg-chair.png', help='Test image path')
    parser.add_argument('--index_path', type=str, default='/data/FAISS/FAISS', help='Index path')
    args = parser.parse_args()
    visualize = args.visualize
    mode = args.mode

    index_path = args.index_path
    clip_search = CLIPUSDSearch()
    clip_search.load_index(index_path)
    # Test with a sample image
    test_image = cv2.imread(args.test_image)
    if test_image is not None:
        image_embedding = clip_search.process_image(test_image)
        urls, scores, image_paths, result_images = asyncio.run(
            clip_search.call_search_post_api("", [image_embedding], limit=5, retrieval_mode=mode)
        )
        print("Search results:", urls, scores, image_paths)
        if visualize:
            # Find indices of top results in the database
            highlight_indices = [clip_search.image_paths.index(p) for p in image_paths if p in clip_search.image_paths]
            if mode == 'pca':
                clip_search.visualize_embeddings_pca(
                    test_embedding=image_embedding,
                    test_label="Test Image",
                    highlight_indices=highlight_indices,
                    n_components=2,
                    color_by='folder'
                )
            elif mode == 'cosine':
                clip_search.visualize_cosine_similarity(
                    test_embedding=image_embedding,
                    highlight_indices=highlight_indices,
                    label_mode='folder'
                )
