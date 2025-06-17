import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Generator
import heapq
from dataclasses import dataclass
import gc
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChunkNode:
    """Represents a node in the RAPTOR tree structure."""
    text: str
    embedding: np.ndarray
    children: List['ChunkNode']
    level: int
    metadata: Dict
    parent: Optional['ChunkNode'] = None

class RaptorRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAPTOR RAG system.
        
        Args:
            model_name (str): Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Load configuration from environment variables with defaults
        self.batch_size = int(os.getenv("RAPTOR_BATCH_SIZE", "16"))
        self.chunk_size = int(os.getenv("RAPTOR_CHUNK_SIZE", "1024"))
        self.similarity_threshold = float(os.getenv("RAPTOR_SIMILARITY_THRESHOLD", "0.8"))
        self.max_levels = int(os.getenv("RAPTOR_MAX_LEVELS", "2"))
        
        print(f"RAPTOR Configuration:")
        print(f"- Batch Size: {self.batch_size}")
        print(f"- Chunk Size: {self.chunk_size}")
        print(f"- Similarity Threshold: {self.similarity_threshold}")
        print(f"- Max Levels: {self.max_levels}")
    
    def _create_initial_chunks(self, text: str) -> Generator[str, None, None]:
        """
        Create initial chunks from text using sliding window approach.
        Uses a generator to avoid storing all chunks in memory.
        
        Args:
            text (str): Input text to chunk
            
        Yields:
            str: Text chunks
        """
        overlap = self.chunk_size // 4  # 25% overlap between chunks
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            yield chunk
            start = end - overlap
    
    def _compute_embeddings_batch(self, chunks: List[str]) -> List[np.ndarray]:
        """
        Compute embeddings for chunks in batches.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            List[np.ndarray]: List of embeddings
        """
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            # Force garbage collection after each batch
            gc.collect()
        return embeddings
    
    def _compute_similarity_matrix_batch(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity matrix between embeddings in batches to save memory.
        
        Args:
            embeddings (List[np.ndarray]): List of embedding vectors
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        # Process in batches to save memory
        for i in range(0, n, self.batch_size):
            end_i = min(i + self.batch_size, n)
            for j in range(0, n, self.batch_size):
                end_j = min(j + self.batch_size, n)
                
                # Compute similarities for current batch
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if ii < jj:  # Only compute upper triangle
                            similarity = np.dot(embeddings[ii], embeddings[jj]) / (
                                np.linalg.norm(embeddings[ii]) * np.linalg.norm(embeddings[jj])
                            )
                            similarity_matrix[ii, jj] = similarity_matrix[jj, ii] = similarity
                
                # Force garbage collection after each batch
                gc.collect()
        
        return similarity_matrix
    
    def _merge_chunks(self, chunks: List[str], embeddings: List[np.ndarray], 
                     similarity_matrix: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Merge similar chunks based on similarity threshold.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (List[np.ndarray]): List of chunk embeddings
            similarity_matrix (np.ndarray): Similarity matrix between chunks
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of merged chunks and their embeddings
        """
        n = len(chunks)
        merged = [False] * n
        merged_chunks = []
        
        # Process in smaller batches to save memory
        for i in range(0, n, self.batch_size):
            end_i = min(i + self.batch_size, n)
            
            for ii in range(i, end_i):
                if merged[ii]:
                    continue
                    
                current_chunk = chunks[ii]
                current_embedding = embeddings[ii]
                similar_indices = []
                
                # Find similar chunks
                for j in range(ii + 1, n):
                    if not merged[j] and similarity_matrix[ii, j] > self.similarity_threshold:
                        similar_indices.append(j)
                
                # Merge similar chunks
                if similar_indices:
                    for idx in similar_indices:
                        current_chunk += " " + chunks[idx]
                        current_embedding = (current_embedding + embeddings[idx]) / 2
                        merged[idx] = True
                    merged[ii] = True
                    merged_chunks.append((current_chunk, current_embedding))
                else:
                    merged_chunks.append((chunks[ii], embeddings[ii]))
                    merged[ii] = True
            
            # Force garbage collection after each batch
            gc.collect()
                
        return merged_chunks
    
    def _build_tree(self, chunks: List[str], embeddings: List[np.ndarray]) -> ChunkNode:
        """
        Build hierarchical tree structure from chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (List[np.ndarray]): List of chunk embeddings
            
        Returns:
            ChunkNode: Root node of the tree
        """
        if len(chunks) == 1:
            return ChunkNode(
                text=chunks[0],
                embedding=embeddings[0],
                children=[],
                level=0,
                metadata={"is_leaf": True}
            )
            
        # Compute similarity matrix in batches
        similarity_matrix = self._compute_similarity_matrix_batch(embeddings)
        
        # Create priority queue for merging
        pq = []
        for i in range(0, len(chunks), self.batch_size):
            end_i = min(i + self.batch_size, len(chunks))
            for j in range(i, len(chunks), self.batch_size):
                end_j = min(j + self.batch_size, len(chunks))
                
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if ii < jj:
                            similarity = similarity_matrix[ii, jj]
                            heapq.heappush(pq, (-similarity, ii, jj))
                
                # Force garbage collection after each batch
                gc.collect()
        
        # Initialize nodes
        nodes = [ChunkNode(
            text=chunk,
            embedding=emb,
            children=[],
            level=0,
            metadata={"is_leaf": True}
        ) for chunk, emb in zip(chunks, embeddings)]
        
        # Merge nodes level by level
        for level in range(self.max_levels - 1):
            if len(nodes) <= 1:
                break
                
            new_nodes = []
            used = set()
            
            while pq and len(new_nodes) < len(nodes) // 2:
                _, i, j = heapq.heappop(pq)
                if i in used or j in used:
                    continue
                    
                # Merge nodes
                merged_text = nodes[i].text + " " + nodes[j].text
                merged_embedding = (nodes[i].embedding + nodes[j].embedding) / 2
                
                parent = ChunkNode(
                    text=merged_text,
                    embedding=merged_embedding,
                    children=[nodes[i], nodes[j]],
                    level=level + 1,
                    metadata={"is_leaf": False}
                )
                
                nodes[i].parent = parent
                nodes[j].parent = parent
                
                new_nodes.append(parent)
                used.add(i)
                used.add(j)
            
            # Add remaining nodes
            for i, node in enumerate(nodes):
                if i not in used:
                    new_nodes.append(node)
            
            nodes = new_nodes
            
            # Force garbage collection after each level
            gc.collect()
            
        return nodes[0] if nodes else None
    
    def process_document(self, text: str) -> ChunkNode:
        """
        Process a document using RAPTOR algorithm.
        
        Args:
            text (str): Input document text
            
        Returns:
            ChunkNode: Root node of the RAPTOR tree
        """
        # Create initial chunks using generator
        initial_chunks = list(self._create_initial_chunks(text))
        
        # Generate embeddings in batches
        print("Generating embeddings...")
        initial_embeddings = self._compute_embeddings_batch(initial_chunks)
        
        # Compute similarity matrix in batches
        print("Computing similarity matrix...")
        # similarity_matrix = self._compute_similarity_matrix_batch(initial_embeddings)
        
        # # Merge similar chunks
        # print("Merging chunks...")
        # merged_chunks, merged_embeddings = zip(*self._merge_chunks(
        #     initial_chunks, initial_embeddings
        # ))
        
        # # Clear memory
        # del initial_chunks, initial_embeddings
        # gc.collect()
        
        # Build hierarchical tree
        print("Building tree...")
        root = self._build_tree(initial_chunks, initial_embeddings)
        
        # Clear memory
        del initial_chunks, initial_embeddings
        gc.collect()
        
        return root
    
    def search(self, query: str, tree: ChunkNode, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the RAPTOR tree for relevant chunks.
        
        Args:
            query (str): Search query
            tree (ChunkNode): Root node of the RAPTOR tree
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk text, similarity score) pairs
        """
        query_embedding = self.model.encode(query)
        results = []
        
        def search_node(node: ChunkNode):
            # Compute similarity with current node
            similarity = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            
            # Add to results if it's a leaf node
            if node.metadata.get("is_leaf", False):
                results.append((node.text, similarity))
            
            # Recursively search children
            for child in node.children:
                search_node(child)
        
        # Perform search
        search_node(tree)
        
        # Sort by similarity and return top-k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k] 