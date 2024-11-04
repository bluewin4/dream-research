import numpy as np
from pathlib import Path
import json
from openai import OpenAI
from numpy.linalg import norm
from typing import Dict, List, Optional, Set
import asyncio
from sklearn.decomposition import PCA
import pickle
import hashlib
import logging
from dotenv import load_dotenv
import os

class PersonalityEmbeddingLibrary:
    def __init__(self, cache_dir: str = "data/embeddings/cache"):
        # Load environment variables
        load_dotenv()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pca_model_path = self.cache_dir / "pca_model.pkl"
        self.vocab_path = self.cache_dir / "vocabulary.json"
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize cache tracking
        self.vocabulary: Set[str] = set()
        self.pca_model = None
        self._load_metadata()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_metadata(self):
        """Load PCA model and vocabulary"""
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r') as f:
                self.vocabulary = set(json.load(f))
                
        if self.pca_model_path.exists():
            with open(self.pca_model_path, 'rb') as f:
                self.pca_model = pickle.load(f)

    def _save_metadata(self):
        """Save PCA model and vocabulary"""
        with open(self.vocab_path, 'w') as f:
            json.dump(list(self.vocabulary), f)
            
        if self.pca_model is not None:
            with open(self.pca_model_path, 'wb') as f:
                pickle.dump(self.pca_model, f)

    def _get_embedding_path(self, text: str) -> Path:
        """Generate unique file path for embedding based on text hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"embed_{text_hash}.npy"

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available"""
        embed_path = self._get_embedding_path(text)
        
        # Check cache first
        if embed_path.exists():
            self.logger.debug(f"Cache hit for: {text[:50]}...")
            return np.load(embed_path)
            
        try:
            self.logger.info(f"Fetching embedding for: {text[:50]}...")
            
            # New OpenAI API format
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            normalized = embedding / norm(embedding)
            
            # Save to cache
            np.save(embed_path, normalized)
            self.vocabulary.add(text)
            self._save_metadata()
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error getting embedding for '{text[:50]}...': {str(e)}")
            return np.zeros(1536)  # Default embedding dimension

    async def compute_personality_vector(self, personality: Dict) -> np.ndarray:
        """Compute 3D vector for personality using cached embeddings"""
        # Extract components
        i_s = personality.get('I_S', '')
        i_g = personality.get('I_G', [])
        i_w = personality.get('I_W', '')
        
        # Convert goals list to string if needed
        if isinstance(i_g, list):
            i_g = ' '.join(i_g)
            
        # Get embeddings
        i_s_embed = await self.get_embedding(str(i_s))
        i_g_embed = await self.get_embedding(str(i_g))
        i_w_embed = await self.get_embedding(str(i_w))
        
        # Stack embeddings
        all_embeddings = np.vstack([i_s_embed, i_g_embed, i_w_embed])
        
        # Initialize or update PCA model
        if self.pca_model is None:
            self.pca_model = PCA(n_components=3)
            self.pca_model.fit(all_embeddings)
            self._save_metadata()
            
        # Transform to 3D
        vec = self.pca_model.transform(all_embeddings).mean(axis=0)
        
        return vec

    def compute_angle(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute angle between two personality vectors"""
        dot_product = np.dot(vec1, vec2)
        norms = norm(vec1) * norm(vec2)
        return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
        
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two personality vectors"""
        return 1 - (self.compute_angle(vec1, vec2) / np.pi)

    def get_vocabulary_stats(self) -> Dict:
        """Get statistics about cached embeddings"""
        return {
            "total_embeddings": len(self.vocabulary),
            "cache_size_mb": sum(f.stat().st_size for f in self.cache_dir.glob("embed_*.npy")) / (1024 * 1024),
            "unique_terms": len(self.vocabulary)
        }

    def clear_cache(self):
        """Clear all cached embeddings"""
        for f in self.cache_dir.glob("embed_*.npy"):
            f.unlink()
        self.vocabulary.clear()
        self._save_metadata()