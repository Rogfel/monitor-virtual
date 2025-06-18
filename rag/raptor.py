import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import heapq
from dataclasses import dataclass
from rag.supabase_rag import get_embedding

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
    
    def _create_initial_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Create initial chunks from text using sliding window approach.
        
        Args:
            text (str): Input text to chunk
            chunk_size (int): Target size for each chunk in characters
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        overlap = chunk_size // 4  # 25% overlap between chunks
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def _compute_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity matrix between all embeddings.
        
        Args:
            embeddings (List[np.ndarray]): List of embedding vectors
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                
        return similarity_matrix
    
    def _merge_chunks(self, chunks: List[str], embeddings: List[np.ndarray], 
                     similarity_matrix: np.ndarray, threshold: float = 0.7) -> List[Tuple[str, np.ndarray]]:
        """
        Merge similar chunks based on similarity threshold.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (List[np.ndarray]): List of chunk embeddings
            similarity_matrix (np.ndarray): Similarity matrix between chunks
            threshold (float): Similarity threshold for merging
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of merged chunks and their embeddings
        """
        n = len(chunks)
        merged = [False] * n
        merged_chunks = []
        
        for i in range(n):
            if merged[i]:
                continue
                
            current_chunk = chunks[i]
            current_embedding = embeddings[i]
            similar_indices = []
            
            # Find similar chunks
            for j in range(i + 1, n):
                if not merged[j] and similarity_matrix[i, j] > threshold:
                    similar_indices.append(j)
            
            # Merge similar chunks
            if similar_indices:
                for idx in similar_indices:
                    current_chunk += " " + chunks[idx]
                    current_embedding = (current_embedding + embeddings[idx]) / 2
                    merged[idx] = True
                merged[i] = True
                merged_chunks.append((current_chunk, current_embedding))
            else:
                merged_chunks.append((chunks[i], embeddings[i]))
                merged[i] = True
                
        return merged_chunks
    
    def _build_tree(self, chunks: List[str], embeddings: List[np.ndarray], 
                   max_levels: int = 3) -> ChunkNode:
        """
        Build hierarchical tree structure from chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (List[np.ndarray]): List of chunk embeddings
            max_levels (int): Maximum number of tree levels
            
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
            
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Create priority queue for merging
        pq = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                similarity = similarity_matrix[i, j]
                heapq.heappush(pq, (-similarity, i, j))
        
        # Initialize nodes
        nodes = [ChunkNode(
            text=chunk,
            embedding=emb,
            children=[],
            level=0,
            metadata={"is_leaf": True}
        ) for chunk, emb in zip(chunks, embeddings)]
        
        # Merge nodes level by level
        for level in range(max_levels - 1):
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
            
        return nodes[0] if nodes else None
    
    def process_document(self, text: str, chunk_size: int = 512, 
                        similarity_threshold: float = 0.7,
                        max_levels: int = 3) -> ChunkNode:
        """
        Process a document using RAPTOR algorithm.
        
        Args:
            text (str): Input document text
            chunk_size (int): Target size for initial chunks
            similarity_threshold (float): Threshold for merging similar chunks
            max_levels (int): Maximum number of tree levels
            
        Returns:
            ChunkNode: Root node of the RAPTOR tree
        """
        # Create initial chunks
        initial_chunks = self._create_initial_chunks(text, chunk_size)
        
        # Generate embeddings for initial chunks
        initial_embeddings = [self.model.encode(chunk) for chunk in initial_chunks]
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(initial_embeddings)
        
        # Merge similar chunks
        merged_chunks, merged_embeddings = zip(*self._merge_chunks(
            initial_chunks, initial_embeddings, similarity_matrix, similarity_threshold
        ))
        
        # Build hierarchical tree
        root = self._build_tree(merged_chunks, merged_embeddings, max_levels)
        
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