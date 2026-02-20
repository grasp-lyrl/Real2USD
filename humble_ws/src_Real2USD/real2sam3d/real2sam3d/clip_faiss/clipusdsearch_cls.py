# Copied from real2usd/scripts_r2u/clipusdsearch_cls.py for self-contained real2sam3d retrieval.
# Matplotlib/sns/fig are lazy so headless (e.g. Docker) works for load/process/search.
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

# Use CPU if no GPU or if GPU has unsupported CUDA capability (e.g. RTX 5090 sm_120)
def _infer_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] > 9:
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"

import clip
import json
import asyncio
from tqdm import tqdm

# matplotlib/sns/plt imported only in visualize_* so headless retrieval works
# (no TkAgg or display required for load_index / process_image / call_search_post_api)


class CLIPUSDSearch:
    """
    CLIP-based USD search implementation using FAISS for similarity search.
    Replaces the NVIDIA USD Search API with a simpler Similarity Search based on CLIP embeddings.
    Clip model names include: ViT-B/32
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self.device = _infer_device()
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.index = None
        self.image_paths = []
        self.usd_paths = []
        self.image_embeddings = []
        self._fig = None  # lazy for headless; set in visualize_* if needed

    def build_index(self, usd_images_dir: str):
        """Build FAISS index from images in the USD images directory."""
        usd_images_dir = Path(usd_images_dir)
        print(f"Building FAISS index from {usd_images_dir}")

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

        embeddings = []
        paths = []
        usd_paths = []
        print("Processing images and extracting embeddings...")
        for img_path in tqdm(image_files, desc="Building FAISS index", unit="image"):
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features = F.normalize(image_features, p=2, dim=1)
                embeddings.append(image_features.cpu().numpy())
                paths.append(str(img_path))
                usd_path = self._find_usd_path(img_path)
                usd_paths.append(usd_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if not embeddings:
            print("No valid embeddings extracted")
            return
        embeddings = np.vstack(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        self.image_paths = paths
        self.usd_paths = usd_paths
        self.image_embeddings = embeddings
        print(f"Built FAISS index with {len(embeddings)} embeddings")

    def _find_usd_path(self, image_path: Path) -> str:
        usd_files = list(image_path.parent.glob("*.usd"))
        if usd_files:
            usd_files.sort(key=lambda x: x.name)
            return str(usd_files[0])
        raise FileNotFoundError(f"No .usd file found in directory {image_path.parent} for image {image_path}")

    def process_image(self, image: np.ndarray, formatted: bool = True) -> str:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        if height > 336 or width > 336:
            image_rgb = cv2.resize(image_rgb, (336, 336))
        pil_image = Image.fromarray(image_rgb)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, p=2, dim=1)
        return image_features.cpu().numpy()

    async def call_search_post_api(self, description: str = "", image_string: List = None,
                                 search_path: str = "SimReadyAssets", exclude_file_name: str = "",
                                 limit: int = 5, retrieval_mode: str = "cosine") -> Tuple[List[str], List[float], List[int], List[str], List[np.ndarray]]:
        """Returns (urls, scores, top_indices, result_image_paths, result_images). top_indices are FAISS indices for the returned urls."""
        if self.index is None:
            print("FAISS index not built. Please ensure USD images directory exists.")
            return None, None, None, None, None
        if not image_string:
            print("No image provided for search")
            return None, None, None, None, None
        try:
            query_embedding = image_string[0]
            urls = []
            result_scores = []
            top_indices = []
            result_image_paths = []
            result_images = []
            if retrieval_mode == "cosine":
                scores, indices = self.index.search(query_embedding.astype('float32'), limit)
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx >= 0 and idx < len(self.usd_paths):
                        usd_path = self.usd_paths[idx]
                        image_path = self.image_paths[idx]
                        urls.append(usd_path)
                        result_scores.append(float(score))
                        top_indices.append(int(idx))
                        result_image_paths.append(image_path)
                        try:
                            result_image = cv2.imread(image_path)
                            if result_image is not None:
                                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            result_images.append(result_image)
                        except Exception as e:
                            print(f"Error loading result image {image_path}: {e}")
                            result_images.append(None)
            elif retrieval_mode == "pca":
                from sklearn.decomposition import PCA
                embeddings = np.array(self.image_embeddings)
                n_components = 2
                pca = PCA(n_components=n_components)
                X_proj = pca.fit_transform(embeddings)
                query_proj = pca.transform(query_embedding.reshape(1, -1))
                dists = np.linalg.norm(X_proj - query_proj, axis=1)
                pca_top = np.argsort(dists)[:limit]
                for idx in pca_top:
                    usd_path = self.usd_paths[idx]
                    image_path = self.image_paths[idx]
                    urls.append(usd_path)
                    result_scores.append(float(-dists[idx]))
                    top_indices.append(int(idx))
                    result_image_paths.append(image_path)
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
            return urls, result_scores, top_indices, result_image_paths, result_images
        except Exception as e:
            print(f"Error in CLIP search: {e}")
            return None, None, None, None, None

    def add_image_to_index(self, image_path: str, usd_path: str):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = F.normalize(image_features, p=2, dim=1)
            if self.index is None:
                dimension = image_features.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
            self.index.add(image_features.cpu().numpy().astype('float32'))
            self.image_paths.append(image_path)
            self.usd_paths.append(usd_path)
            self.image_embeddings.append(image_features.cpu().numpy())
            print(f"Added {image_path} to index")
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")

    def save_index(self, index_path: str):
        try:
            faiss.write_index(self.index, f"{index_path}.faiss")
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
        try:
            self.index = faiss.read_index(f"{index_path}.faiss")
            with open(f"{index_path}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            self.image_paths = metadata['image_paths']
            self.usd_paths = metadata['usd_paths']
            self.image_embeddings = [np.array(emb) for emb in metadata['image_embeddings']]
            print(f"Loaded index from {index_path}")
        except Exception as e:
            print(f"Error loading the index. Please run setup_clip_search.py to build the index")

    def visualize_cosine_similarity(self, test_embedding, highlight_indices=None, label_mode='folder'):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
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
        if self._fig is None:
            self._fig = plt.figure(figsize=(8, 4))
        self._fig.clf()
        ax = self._fig.add_subplot()
        ax.scatter(indices, similarities, label='All DB Embeddings', alpha=0.6)
        if highlight_indices is not None:
            ax.scatter(highlight_indices, similarities[highlight_indices], color='red', s=80, label='Top Results')
        top_n = 50
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        for i in top_indices:
            ax.text(indices[i], similarities[i], labels[i], fontsize=7, alpha=0.9, rotation=45, color='blue')
        ax.set_xlabel('Database Index (d)')
        ax.set_ylabel('Cosine Similarity')
        self._fig.tight_layout()
        self._fig.canvas.draw()
        plt.pause(0.001)

    def visualize_embeddings_pca(self, test_embedding=None, test_label=None, highlight_indices=None, n_components=2, color_by='folder'):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        embeddings = self.image_embeddings
        image_paths = self.image_paths
        usd_paths = self.usd_paths
        pca = PCA(n_components=n_components)
        X_proj = pca.fit_transform(embeddings)
        if test_embedding is not None:
            test_proj = pca.transform(test_embedding.reshape(1, -1))
        if color_by == 'folder':
            labels = [Path(p).parent.name for p in image_paths]
        elif color_by == 'usd':
            labels = [Path(p).name for p in usd_paths]
        else:
            labels = [''] * len(image_paths)
        unique_labels = list(set(labels))
        colors = sns.color_palette("tab10", len(unique_labels))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]
        fig = plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(X_proj[:, 0], X_proj[:, 1], c=point_colors, alpha=0.7, label='DB Embeddings')
            if highlight_indices is not None:
                plt.scatter(X_proj[highlight_indices, 0], X_proj[highlight_indices, 1],
                            facecolors='none', edgecolors='black', s=120, linewidths=2, label='Top Results')
            if test_embedding is not None:
                plt.scatter(test_proj[0, 0], test_proj[0, 1], marker='*', c='red', s=200, label='Test Image')
                if test_label:
                    plt.text(test_proj[0, 0], test_proj[0, 1], test_label, fontsize=10, color='red')
            max_labels = 50
            step = max(1, len(labels) // max_labels)
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
            max_labels = 50
            step = max(1, len(labels) // max_labels)
            for i in range(0, len(labels), step):
                ax.text(X_proj[i, 0], X_proj[i, 1], X_proj[i, 2], labels[i], fontsize=7, alpha=0.7)
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
        plt.title('CLIP Embeddings PCA Projection')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--mode', choices=['pca', 'cosine'], default='pca')
    parser.add_argument('--test_image', type=str, default='/data/tests/seg-chair.png')
    parser.add_argument('--index_path', type=str, default='/data/FAISS/FAISS')
    args = parser.parse_args()
    index_path = args.index_path
    clip_search = CLIPUSDSearch()
    clip_search.load_index(index_path)
    test_image = cv2.imread(args.test_image)
    if test_image is not None:
        image_embedding = clip_search.process_image(test_image)
        urls, scores, top_indices, result_image_paths, result_images = asyncio.run(
            clip_search.call_search_post_api("", [image_embedding], limit=5, retrieval_mode=args.mode)
        )
        print("Search results:", urls, scores, top_indices, result_image_paths)
        if args.visualize:
            highlight_indices = top_indices if top_indices else []
            if args.mode == 'pca':
                clip_search.visualize_embeddings_pca(
                    test_embedding=image_embedding, test_label="Test Image",
                    highlight_indices=highlight_indices, n_components=2, color_by='folder'
                )
            elif args.mode == 'cosine':
                clip_search.visualize_cosine_similarity(
                    test_embedding=image_embedding, highlight_indices=highlight_indices, label_mode='folder'
                )
