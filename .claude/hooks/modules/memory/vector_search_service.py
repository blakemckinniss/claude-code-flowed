"""
Vector Search Service - Provides embedding generation and similarity search
Supports multiple providers with caching and error handling
"""

import json
import os
import hashlib
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

class VectorSearchConfig:
    """Configuration for vector search service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("vectorSearch", {}).get("enabled", False)
        self.provider = config.get("vectorSearch", {}).get("provider", "openai")
        self.model = config.get("vectorSearch", {}).get("model", "text-embedding-3-small")
        self.dimensions = config.get("vectorSearch", {}).get("dimensions", 1536)
        self.similarity_threshold = config.get("vectorSearch", {}).get("similarityThreshold", 0.75)
        self.cache_size = config.get("vectorSearch", {}).get("cacheSize", 1000)
        self.batch_size = config.get("vectorSearch", {}).get("batchSize", 50)
        self.timeout = config.get("vectorSearch", {}).get("timeout", 30)
        
        # Context injection settings
        self.context_injection = config.get("contextInjection", {})
        self.max_context_memories = self.context_injection.get("maxMemories", 5)
        self.context_relevance_threshold = self.context_injection.get("relevanceThreshold", 0.8)


class EmbeddingProvider:
    """Base class for embedding providers"""
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        raise NotImplementedError
    
    async def generate_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for batch of texts"""
        # Default implementation - can be optimized per provider
        results = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            if embedding is None:
                return None
            results.append(embedding)
        return results


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found, OpenAI provider will be disabled")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        if not self.api_key:
            return None
        
        try:
            # Use subprocess to call openai API (since we don't have openai package)
            # This is a simplified implementation - in production, use proper HTTP client
            import json
            import tempfile
            
            # Create temporary file with text
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(text)
                temp_file = f.name
            
            try:
                # Use curl to call OpenAI API
                cmd = [
                    "curl", "-s", "https://api.openai.com/v1/embeddings",
                    "-H", f"Authorization: Bearer {self.api_key}",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps({
                        "input": text,
                        "model": self.config.model
                    })
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.config.timeout
                )
                
                if process.returncode == 0:
                    response = json.loads(stdout.decode())
                    if "data" in response and len(response["data"]) > 0:
                        return response["data"][0]["embedding"]
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
        
        return None


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using simple TF-IDF-like approach"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.vocabulary = {}
        self.idf_scores = {}
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate simple TF-IDF-like embedding"""
        try:
            # Simple tokenization
            words = text.lower().split()
            
            # Build vocabulary if empty
            if not self.vocabulary:
                self._build_vocabulary([text])
            
            # Create embedding vector
            embedding = [0.0] * min(self.config.dimensions, len(self.vocabulary))
            
            # Calculate TF-IDF scores
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            vocab_list = list(self.vocabulary.keys())[:len(embedding)]
            for i, vocab_word in enumerate(vocab_list):
                if vocab_word in word_counts:
                    tf = word_counts[vocab_word] / len(words)
                    idf = self.idf_scores.get(vocab_word, 1.0)
                    embedding[i] = tf * idf
            
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            return None
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_freq = {}
        doc_count = {}
        
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
                doc_count[word] = doc_count.get(word, 0) + 1
        
        # Keep most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: i for i, (word, _) in enumerate(sorted_words[:self.config.dimensions])}
        
        # Calculate IDF scores
        total_docs = len(texts)
        for word in self.vocabulary:
            self.idf_scores[word] = np.log(total_docs / (doc_count.get(word, 1) + 1))


class VectorSearchService:
    """Vector search service with embedding generation and similarity search"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = VectorSearchConfig(config)
        self.provider = self._create_provider()
        self.embedding_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = timedelta(hours=24)  # Cache embeddings for 24 hours
        
    def _create_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on configuration"""
        if self.config.provider == "openai":
            return OpenAIEmbeddingProvider(self.config)
        elif self.config.provider == "local":
            return LocalEmbeddingProvider(self.config)
        else:
            logger.warning(f"Unknown provider {self.config.provider}, using local provider")
            return LocalEmbeddingProvider(self.config)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached embedding is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[cache_key]
        return datetime.now() - timestamp < self.cache_ttl
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with caching"""
        if not self.config.enabled:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache and self._is_cache_valid(cache_key):
            return self.embedding_cache[cache_key]
        
        # Generate new embedding
        embedding = await self.provider.generate_embedding(text)
        
        # Cache result
        if embedding is not None:
            # Implement LRU cache manually
            if len(self.embedding_cache) >= self.config.cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
                del self.embedding_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.embedding_cache[cache_key] = embedding
            self.cache_timestamps[cache_key] = datetime.now()
        
        return embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for efficient computation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def find_similar_memories(self, query_text: str, memories: List[Dict[str, Any]], 
                                   top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar memories using vector search"""
        if not self.config.enabled or not memories:
            return []
        
        try:
            # Get query embedding
            query_embedding = await self.get_embedding(query_text)
            if query_embedding is None:
                return []
            
            # Calculate similarities
            similarities = []
            for memory in memories:
                # Extract searchable text from memory
                memory_text = self._extract_searchable_text(memory)
                
                # Get memory embedding
                memory_embedding = await self.get_embedding(memory_text)
                if memory_embedding is None:
                    continue
                
                # Calculate similarity
                similarity = self.cosine_similarity(query_embedding, memory_embedding)
                
                # Only include if above threshold
                if similarity >= self.config.similarity_threshold:
                    similarities.append((memory, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    def _extract_searchable_text(self, memory: Dict[str, Any]) -> str:
        """Extract searchable text from memory object"""
        parts = []
        
        # Add key
        if "key" in memory:
            parts.append(memory["key"])
        
        # Add value (convert to string if needed)
        if "value" in memory:
            value = memory["value"]
            if isinstance(value, dict):
                # Extract text from dict values
                for v in value.values():
                    if isinstance(v, str):
                        parts.append(v)
            elif isinstance(value, str):
                parts.append(value)
            else:
                parts.append(str(value))
        
        # Add category
        if "category" in memory:
            parts.append(memory["category"])
        
        # Add metadata text
        if "metadata" in memory and isinstance(memory["metadata"], dict):
            for v in memory["metadata"].values():
                if isinstance(v, str):
                    parts.append(v)
        
        return " ".join(parts)
    
    async def get_contextual_memories(self, context: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get contextually relevant memories for Claude Code visibility"""
        if not self.config.enabled:
            return []
        
        try:
            # Find similar memories
            similar_memories = await self.find_similar_memories(
                context, 
                memories, 
                top_k=self.config.max_context_memories
            )
            
            # Filter by context relevance threshold
            relevant_memories = [
                memory for memory, similarity in similar_memories
                if similarity >= self.config.context_relevance_threshold
            ]
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Error getting contextual memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector search service statistics"""
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "cached_embeddings": len(self.embedding_cache),
            "cache_size_limit": self.config.cache_size,
            "similarity_threshold": self.config.similarity_threshold,
            "dimensions": self.config.dimensions
        }