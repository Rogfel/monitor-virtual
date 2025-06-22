import os
import numpy as np
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Global model instance to avoid reloading for each embedding
_sentence_model = None

def get_embedding(text, model_name="all-MiniLM-L6-v2", batch=False):
    """
    Creates an embedding for the given text using SentenceTransformer.

    Args:
    text (str or list): Input text or list of texts for batch processing.
    model_name (str): Embedding model name.
    batch (bool): Whether the input is a batch of texts.

    Returns:
    np.ndarray: The embedding vector or array of vectors.
    """
    global _sentence_model
    
    # Initialize model only once
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(model_name)
    
    # Process single text or batch
    if batch:
        return _sentence_model.encode(text, batch_size=32)
    return _sentence_model.encode(text)

class SupabaseRAG:
    def __init__(self, supabase_url=None, supabase_key=None, table_name="documents"):
        """
        Initialize the SupabaseRAG client.
        
        Args:
            supabase_url (str): Supabase URL. If None, reads from SUPABASE_URL env var.
            supabase_key (str): Supabase API key. If None, reads from SUPABASE_KEY env var.
            table_name (str): Name of the table to store documents. Default is "documents".
        """
        # Get Supabase credentials from environment variables if not provided
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and API key must be provided either as arguments or environment variables.")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.table_name = table_name
        
        # Initialize SentenceTransformer model
        self.model_name = "all-MiniLM-L6-v2"
        
        # Configuration for timeouts and retries
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    def __total_records(self):
        response = self.supabase.table(self.table_name).select("*", count="exact").execute()
        return response.count
    
    def __execute_search_with_retry(self, query_embedding, match_threshold, match_count, offset_value):
        """
        Execute search with retry logic and timeout handling.
        
        Args:
            query_embedding: The query embedding vector
            match_threshold: Similarity threshold
            match_count: Number of results to return
            offset_value: Offset for pagination
            
        Returns:
            dict: Response from Supabase
        """
        for attempt in range(self.max_retries):
            try:
                response = self.supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding.tolist(),
                        'match_threshold': match_threshold,
                        'match_count': match_count,
                        'offset_value': offset_value
                    }
                ).execute()
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # If it's a timeout error and we have more retries, wait and try again
                if ("timeout" in error_msg or "57014" in str(e)) and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # If it's not a timeout or we're out of retries, raise the exception
                    raise e
        
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed")
    
    def insert_document(self, title, document_text, embedding, nome_documento=None,
                        pagina=None, pdf_path=None, titulo_secao=None):
        """
        Insert a document into the Supabase table with its embedding.
        
        Args:
            title (str): Title of the document.
            document_text (str): Text content of the document.
            embedding (np.ndarray): Embedding of the document.
            nome_documento (str): Name of the source document.
            pagina (int): Page number where the chunk was found.
            pdf_path (str): Path to the PDF file.
            titulo_secao (str): Title of the section.
            
        Returns:
            dict: Response from Supabase.
        """
        # Ensure embedding is a numpy array with correct shape
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Ensure embedding has the correct shape (384 dimensions)
        if embedding.shape != (384,):
            raise ValueError(f"Embedding must have shape (384,), got {embedding.shape}")
        
        # Convert embedding to list and ensure it's a flat list of floats
        embedding_list = embedding.astype(np.float32).tolist()
        
        # Insert document into Supabase
        response = self.supabase.table(self.table_name).insert({
            "titulo": title,
            "documento": document_text,
            "nome_documento": nome_documento,
            "pagina": pagina,
            "pdf_path": pdf_path,
            "titulo_secao": titulo_secao,
            "embedding": embedding_list
        }).execute()
        
        return response
    
    def insert_documents_batch(self, titles, documents_text,
                               embeddings, nome_documento=None,
                               numero_chunks=None, paginas=None,
                               pdf_path=None, section_titles=None):
        """
        Insert multiple documents into the Supabase table with their embeddings.
        
        Args:
            titles (list): List of document titles.
            documents_text (list): List of document texts.
            embeddings (list): List of embeddings.
            nome_documento (str): Name of the source document.
            numero_chunks (list): List of chunk numbers.
            paginas (list): List of page numbers.
            pdf_path (str): Path to the PDF file.
            section_titles (list): List of section titles.
            
        Returns:
            dict: Response from Supabase.
        """
        if len(titles) != len(documents_text):
            raise ValueError("The number of titles must match the number of documents")
        
        if numero_chunks and len(numero_chunks) != len(documents_text):
            raise ValueError("The number of chunk numbers must match the number of documents")
            
        if paginas and len(paginas) != len(documents_text):
            raise ValueError("The number of page numbers must match the number of documents")
        
        
        # Prepare data for batch insertion
        data = [
            {
                "titulo": titles,
                "documento": doc_text,
                "nome_documento": nome_documento,
                "numero_chunk": numero_chunks[i] if numero_chunks else None,
                "pagina": paginas[i] if paginas else None,
                "embedding": embedding,
                "pdf_path": pdf_path,
                "titulo_secao": section_titles[i] if section_titles else None
            }
            for i, (doc_text, embedding) in enumerate(zip(documents_text, embeddings))
        ]
        
        # Insert documents into Supabase
        response = self.supabase.table(self.table_name).insert(data).execute()
        
        return response
    
    def search_documents(self, query_text, top_k=None, match_threshold=None, ensure_function=False):
        """
        Search for the top k most similar documents to the query.
        
        Args:
            query_text (str): Query text to search for.
            top_k (int): Number of top results to return. If None, uses TOP_VALUE from env.
            match_threshold (float): Minimum similarity score for a match. If None, uses MATCH_THRESHOLD from env.
            ensure_function (bool): Whether to ensure the match_documents function exists.
            
        Returns:
            list: List of documents sorted by relevance.
        """
        # Set default values from environment variables
        if top_k is None:
            top_k = int(os.getenv("TOP_VALUE", "10"))
        if match_threshold is None:
            match_threshold = float(os.getenv("MATCH_THRESHOLD", "0.5"))
        
        # Ensure the match_documents function exists if requested
        if ensure_function:
            try:
                self.create_match_documents_function()
                print("Created or updated match_documents function.")
            except Exception as e:
                print(f"Warning: Could not create match_documents function: {e}")
                print("Attempting to proceed with search anyway...")
        
        # Generate embedding for the query
        query_embedding = get_embedding(query_text, self.model_name)
        
        # Get batch size from environment, with fallback
        batch_size = int(os.getenv("BATCH_SIZE", "50"))
        
        # # Limit batch size to top_k to avoid unnecessary queries
        # batch_size = min(batch_size, top_k)
        
        results = []
        offset = 0
        max_iterations = (top_k // batch_size) + 2  # Add buffer for safety
        iteration = 0
        
        print(f"Searching for top {top_k} results with batch size {batch_size}")
        
        while len(results) < top_k and iteration < max_iterations:
            print(f"Iteration {iteration} of {max_iterations}, offset {offset} of {top_k}")
            try:
                # Calculate how many results we still need
                remaining = top_k - len(results)
                current_batch_size = min(batch_size, remaining)
                
                # Perform vector similarity search using Supabase pgvector
                response = self.__execute_search_with_retry(query_embedding, match_threshold, current_batch_size, offset)
                
                batch_results = response.data
                if not batch_results:
                    print(f"No more results found at offset {offset}")
                    break
                
                results.extend(batch_results)
                offset += len(batch_results)
                iteration += 1
                
                print(f"Found {len(batch_results)} results in batch {iteration}, total: {len(results)}")
                
                # If we got fewer results than requested, we've reached the end
                if len(batch_results) < current_batch_size:
                    break
                    
            except Exception as e:
                print(f"Error during search at iteration {iteration}: {e}")
                # If it's a timeout error, try with smaller batch size
                if "timeout" in str(e).lower() and batch_size > 10:
                    batch_size = max(10, batch_size // 2)
                    print(f"Reducing batch size to {batch_size} due to timeout")
                    continue
                else:
                    print("Search failed, returning partial results")
                    break
    
        # Limit results to top_k and sort by similarity
        results = results[:top_k]
        
        print(f"Search completed. Found {len(results)} results")
        return results
    
    def create_match_documents_function(self):
        """
        Creates the match_documents function in Supabase if it doesn't exist.
        This is a helper method to set up the necessary database function.
        
        Note: This requires appropriate permissions in your Supabase project.
        """
        # SQL to create the match_documents function with the correct table name
        sql = f"""
        -- Create the documents table if it doesn't exist
        create table if not exists public.{self.table_name} (
            id bigint generated by default as identity primary key,
            created_at timestamp with time zone default timezone('utc'::text, now()) not null,
            titulo text,
            documento text,
            nome_documento text,
            pagina integer,
            pdf_path text,
            titulo_secao text,
            embedding vector(384)
        );

        -- Drop existing indexes to recreate them with better configuration
        drop index if exists documents_embedding_idx;
        drop index if exists {self.table_name}_embedding_idx;

        -- Create a more efficient index for vector searches
        -- Using HNSW index which is generally faster than IVFFlat for similarity search
        create index if not exists {self.table_name}_embedding_idx 
        on public.{self.table_name} 
        using hnsw (embedding vector_cosine_ops) 
        with (m = 16, ef_construction = 64);

        -- Create additional indexes for better performance on other columns
        create index if not exists {self.table_name}_nome_documento_idx 
        on public.{self.table_name} (nome_documento);
        
        create index if not exists {self.table_name}_pagina_idx 
        on public.{self.table_name} (pagina);

        -- Drop the existing function if it exists
        drop function if exists public.match_documents;

        -- Create the optimized match_documents function
        create or replace function public.match_documents (
            query_embedding vector(384),
            match_threshold float,
            match_count int,
            offset_value int
        )
        returns table (
            id bigint,
            titulo text,
            documento text,
            nome_documento text,
            pdf_path text,
            titulo_secao text,
            pagina integer,
            similarity float
        )
        language plpgsql
        as $$
        begin
            return query
            select
                d.id,
                d.titulo,
                d.documento,
                d.nome_documento,
                d.pdf_path,
                d.titulo_secao,
                d.pagina,
                1 - (d.embedding <=> query_embedding) as similarity
            from public.{self.table_name} d
            where 1 - (d.embedding <=> query_embedding) > match_threshold
            order by d.embedding <=> query_embedding  -- More efficient ordering
            limit match_count
            offset offset_value;
            set statement_timeout TO '5min';
        end;
        $$;

        -- Add permissions for the function
        grant execute on function public.match_documents(vector(384), float, int, int) to anon, authenticated, service_role;
        
        -- Set statement timeout for the function to prevent long-running queries
        alter function public.match_documents(vector(384), float, int, int) set statement_timeout = '30s';
        """
        
        try:
            # Execute the SQL using the REST API
            self.supabase.rpc('exec_sql', {'sql': sql}).execute()
            return "match_documents function created successfully"
        except Exception as e:
            print(f"Error creating function: {e}")
            # Try alternative method if the first one fails
            try:
                self.supabase.table(self.table_name).select("*").limit(1).execute()
                return "Table exists, proceeding with existing function"
            except Exception as e2:
                raise Exception(f"Failed to create or verify function: {str(e2)}")

    def optimize_database(self):
        """
        Optimize database settings for better performance.
        This method should be called after creating the table and function.
        """
        sql = f"""
        -- Analyze the table to update statistics
        analyze public.{self.table_name};
        
        -- Set work_mem for better performance on vector operations
        set work_mem = '256MB';
        
        -- Set effective_cache_size for better query planning
        set effective_cache_size = '1GB';
        
        -- Set random_page_cost for SSD storage
        set random_page_cost = 1.1;
        
        -- Set seq_page_cost for SSD storage
        set seq_page_cost = 1.0;
        
        -- Set shared_preload_libraries if not already set
        -- Note: This requires a database restart to take effect
        -- set shared_preload_libraries = 'vector';
        """
        
        try:
            self.supabase.rpc('exec_sql', {'sql': sql}).execute()
            print("Database optimization completed")
        except Exception as e:
            print(f"Warning: Could not optimize database settings: {e}")
    
    def get_table_stats(self):
        """
        Get statistics about the documents table.
        
        Returns:
            dict: Table statistics
        """
        try:
            # Get total count
            total_count = self.__total_records()
            
            # Get sample of embeddings to check dimensions
            sample = self.supabase.table(self.table_name).select("embedding").limit(1).execute()
            
            stats = {
                "total_documents": total_count,
                "table_name": self.table_name,
                "has_embeddings": len(sample.data) > 0 if sample.data else False
            }
            
            if sample.data and sample.data[0].get('embedding'):
                stats["embedding_dimensions"] = len(sample.data[0]['embedding'])
            
            return stats
            
        except Exception as e:
            print(f"Error getting table stats: {e}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Set your Supabase credentials
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Initialize the SupabaseRAG client
    rag = SupabaseRAG(SUPABASE_URL, SUPABASE_KEY)
    
    # Get table statistics
    stats = rag.get_table_stats()
    print("Table stats:", stats)
    
    # Create the match_documents function in Supabase (run once)
    # rag.create_match_documents_function()
    
    # Optimize database settings (run once after setup)
    # rag.optimize_database()
    
    # Example: Insert a document
    sample_embedding = get_embedding("This is an example document about artificial intelligence and machine learning.")
    response = rag.insert_document(
        title="Example Document",
        document_text="This is an example document about artificial intelligence and machine learning.",
        embedding=sample_embedding,
        nome_documento="Source Document",
        pagina=1
    )
    print("Document inserted:", response)
    
    # Example: Search for documents with improved error handling
    try:
        results = rag.search_documents("What is artificial intelligence?", top_k=5)
        print(f"Search results: {len(results)} documents found")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('titulo', 'No title')} (similarity: {result.get('similarity', 0):.3f})")
    except Exception as e:
        print(f"Search failed: {e}")