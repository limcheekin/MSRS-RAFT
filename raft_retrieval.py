"""
RAFT Retrieval System Module
Implements embedding, indexing, retrieval, and re-ranking
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import pickle
import numpy as np
from dataclasses import dataclass

import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

logger = logging.getLogger("RAFT.Retrieval")


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChunkInfo:
    """Information about a text chunk"""
    chunk_id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int
    metadata: Optional[Dict[str, Any]] = None


class TextChunker:
    """Handles text chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized chunker: size={chunk_size}, overlap={overlap}")
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None
    ) -> List[ChunkInfo]:
        """
        Chunk text with overlap
        
        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Optional metadata to attach
            
        Returns:
            List of ChunkInfo objects
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            # Single chunk
            chunk = ChunkInfo(
                chunk_id=f"{doc_id}_0",
                doc_id=doc_id,
                text=text,
                start_idx=0,
                end_idx=len(words),
                metadata=metadata
            )
            return [chunk]
        
        # Multiple chunks with overlap
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunk = ChunkInfo(
                chunk_id=f"{doc_id}_{chunk_idx}",
                doc_id=doc_id,
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Move to next chunk
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - self.overlap
            chunk_idx += 1
        
        return chunks
    
    def chunk_documents(
        self,
        documents: Dict[str, str],
        batch_size: int = 100
    ) -> List[ChunkInfo]:
        """
        Chunk multiple documents
        
        Args:
            documents: Dict mapping doc_id to text
            batch_size: Batch size for processing
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        doc_items = list(documents.items())
        
        for i in tqdm(range(0, len(doc_items), batch_size), desc="Chunking documents"):
            batch = doc_items[i:i + batch_size]
            
            for doc_id, text in batch:
                chunks = self.chunk_text(text, doc_id)
                all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class EmbeddingModel:
    """Wrapper for embedding model"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Loaded embedding model: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class FAISSIndex:
    """FAISS-based vector index"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index (flat, ivf, hnsw)
            metric: Similarity metric (cosine, l2)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        if metric == "cosine":
            # Use inner product after normalization
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.chunk_mapping = []  # Maps index position to chunk_id
        logger.info(f"Initialized FAISS index: type={index_type}, metric={metric}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ):
        """
        Add embeddings to index
        
        Args:
            embeddings: Numpy array of embeddings
            chunk_ids: List of chunk IDs corresponding to embeddings
        """
        try:
            # Ensure float32
            embeddings = embeddings.astype(np.float32)
            
            # Add to index
            self.index.add(embeddings)
            self.chunk_mapping.extend(chunk_ids)
            
            logger.info(f"Added {len(embeddings)} embeddings to index")
            logger.info(f"Total index size: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            raise
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for nearest neighbors
        
        Args:
            query_embeddings: Query embeddings
            k: Number of results to return
            
        Returns:
            Tuple of (scores, indices)
        """
        try:
            query_embeddings = query_embeddings.astype(np.float32)
            scores, indices = self.index.search(query_embeddings, k)
            return scores, indices
        except Exception as e:
            logger.error(f"Failed to search index: {str(e)}")
            raise
    
    def get_chunk_ids(self, indices: np.ndarray) -> List[List[str]]:
        """
        Convert indices to chunk IDs
        
        Args:
            indices: Array of indices from search
            
        Returns:
            List of lists of chunk IDs
        """
        chunk_ids = []
        for idx_list in indices:
            chunk_ids.append([
                self.chunk_mapping[idx] if idx < len(self.chunk_mapping) else None
                for idx in idx_list
            ])
        return chunk_ids
    
    def save(self, path: str):
        """Save index to disk"""
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save mapping
            with open(f"{path}.mapping.pkl", 'wb') as f:
                pickle.dump({
                    'chunk_mapping': self.chunk_mapping,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type,
                    'metric': self.metric
                }, f)
            
            logger.info(f"Saved index to {path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'FAISSIndex':
        """Load index from disk"""
        try:
            # Load mapping
            with open(f"{path}.mapping.pkl", 'rb') as f:
                data = pickle.load(f)
            
            # Create instance
            instance = cls(
                embedding_dim=data['embedding_dim'],
                index_type=data['index_type'],
                metric=data['metric']
            )
            
            # Load FAISS index
            instance.index = faiss.read_index(f"{path}.faiss")
            instance.chunk_mapping = data['chunk_mapping']
            
            logger.info(f"Loaded index from {path}: {instance.index.ntotal} vectors")
            return instance
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise


class Reranker:
    """Cross-encoder reranker"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None
    ):
        """
        Initialize reranker
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
        """
        self.model_name = model_name
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"Loaded reranker: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {str(e)}")
            raise
    
    def rerank(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank texts for a query
        
        Args:
            query: Query text
            texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        try:
            # Create pairs
            pairs = [[query, text] for text in texts]
            
            # Score pairs
            scores = self.model.predict(pairs)
            
            # Sort by score
            ranked = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            if top_k:
                ranked = ranked[:top_k]
            
            return ranked
        except Exception as e:
            logger.error(f"Failed to rerank: {str(e)}")
            raise


class RetrievalSystem:
    """Complete retrieval system with embedding, indexing, and reranking"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        index_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize retrieval system
        
        Args:
            embedding_model: Embedding model name
            reranker_model: Reranker model name (None to disable)
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            index_path: Path to load existing index
            device: Device to use
        """
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingModel(embedding_model, device=device)
        self.reranker = Reranker(reranker_model, device=device) if reranker_model else None
        
        self.index = None
        self.chunk_store = {}  # Maps chunk_id to ChunkInfo
        
        if index_path and Path(index_path + ".faiss").exists():
            self.load_index(index_path)
        
        logger.info("Initialized retrieval system")
    
    def build_index(
        self,
        documents: Dict[str, str],
        batch_size: int = 32,
        save_path: Optional[str] = None
    ):
        """
        Build index from documents
        
        Args:
            documents: Dict mapping doc_id to text
            batch_size: Batch size for embedding
            save_path: Path to save index
        """
        logger.info(f"Building index from {len(documents)} documents")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        # Store chunks
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = chunk
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Encode texts
        logger.info("Encoding chunks...")
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        
        # Create index
        embedding_dim = self.embedder.get_embedding_dim()
        self.index = FAISSIndex(embedding_dim, metric="cosine")
        
        # Add embeddings
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.index.add_embeddings(embeddings, chunk_ids)
        
        # Save if requested
        if save_path:
            self.save_index(save_path)
        
        logger.info("Index building complete")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of results from initial retrieval
            rerank_top_k: Number of results after reranking
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedder.encode([query], show_progress=False)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k=top_k)
        chunk_ids_list = self.index.get_chunk_ids(indices)
        
        # Get chunks
        chunk_ids = chunk_ids_list[0]
        scores = scores[0]
        
        results = []
        for i, (chunk_id, score) in enumerate(zip(chunk_ids, scores)):
            if chunk_id is None or chunk_id not in self.chunk_store:
                continue
            
            chunk = self.chunk_store[chunk_id]
            result = RetrievalResult(
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(score),
                rank=i,
                metadata={'chunk_id': chunk_id}
            )
            results.append(result)
        
        # Rerank if enabled
        if self.reranker and results:
            results = self._rerank_results(query, results, rerank_top_k)
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank retrieval results"""
        texts = [r.text for r in results]
        ranked = self.reranker.rerank(query, texts, top_k=top_k)
        
        reranked_results = []
        for new_rank, (old_idx, new_score) in enumerate(ranked):
            result = results[old_idx]
            result.score = float(new_score)
            result.rank = new_rank
            reranked_results.append(result)
        
        return reranked_results
    
    def save_index(self, path: str):
        """Save index and chunk store"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        self.index.save(path)
        
        # Save chunk store
        with open(f"{path}.chunks.pkl", 'wb') as f:
            pickle.dump(self.chunk_store, f)
        
        logger.info(f"Saved retrieval system to {path}")
    
    def load_index(self, path: str):
        """Load index and chunk store"""
        # Load FAISS index
        self.index = FAISSIndex.load(path)
        
        # Load chunk store
        with open(f"{path}.chunks.pkl", 'rb') as f:
            self.chunk_store = pickle.load(f)
        
        logger.info(f"Loaded retrieval system from {path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
    documents = {
        "doc1": "This is a sample document about machine learning.",
        "doc2": "Natural language processing is a subfield of AI.",
        "doc3": "Deep learning uses neural networks with multiple layers."
    }
    
    # Initialize system
    retrieval_system = RetrievalSystem(
        embedding_model="BAAI/bge-small-en-v1.5",
        reranker_model=None,  # Disable for demo
        chunk_size=100,
        chunk_overlap=20
    )
    
    # Build index
    retrieval_system.build_index(documents)
    
    # Retrieve
    query = "What is machine learning?"
    results = retrieval_system.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for result in results:
        print(f"  Rank {result.rank}: {result.doc_id} (score: {result.score:.4f})")
        print(f"  Text: {result.text[:100]}...")