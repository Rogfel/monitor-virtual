import os
import numpy as np
from supabase import create_client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.table_name = table_name
        
        # Initialize SentenceTransformer model
        self.model_name = "all-MiniLM-L6-v2"
    
    def insert_document(self, title, document_text):
        """
        Insert a document into the Supabase table with its embedding.
        
        Args:
            title (str): Title of the document.
            document_text (str): Text content of the document.
            
        Returns:
            dict: Response from Supabase.
        """
        # Generate embedding for the document
        embedding = get_embedding(document_text, self.model_name)
        
        # Insert document into Supabase
        response = self.supabase.table(self.table_name).insert({
            "titulo": title,
            "documento": document_text,
            "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
        }).execute()
        
        return response
    
    def insert_documents_batch(self, titles, documents_text):
        """
        Insert multiple documents into the Supabase table with their embeddings.
        
        Args:
            titles (list): List of document titles.
            documents_text (list): List of document texts.
            
        Returns:
            dict: Response from Supabase.
        """
        if len(titles) != len(documents_text):
            raise ValueError("The number of titles must match the number of documents")
        
        # Generate embeddings for all documents at once
        embeddings = get_embedding(documents_text, self.model_name, batch=True)
        
        # Prepare data for batch insertion
        data = [
            {
                "titulo": title,
                "documento": doc_text,
                "embedding": emb.tolist()  # Convert numpy array to list for JSON serialization
            }
            for title, doc_text, emb in zip(titles, documents_text, embeddings)
        ]
        
        # Insert documents into Supabase
        response = self.supabase.table(self.table_name).insert(data).execute()
        
        return response
    
    def search_documents(self, query_text, top_k=5, ensure_function=True):
        """
        Search for the top k most similar documents to the query.
        
        Args:
            query_text (str): Query text to search for.
            top_k (int): Number of top results to return.
            ensure_function (bool): Whether to ensure the match_documents function exists.
            
        Returns:
            list: List of documents sorted by relevance.
        """
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
        
        # Convert embedding to string format for Supabase vector search
        embedding_str = str(query_embedding.tolist())
        
        try:
            # Perform vector similarity search using Supabase pgvector
            response = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': embedding_str,
                    'match_threshold': 0.5,
                    'match_count': top_k
                }
            ).execute()
            
            return response.data
        except Exception as e:
            print(f"Error during search: {e}")
            print("This may be due to the match_documents function not existing or using the wrong table name.")
            print(f"Make sure the function is created and uses the table '{self.table_name}'.")
            return []
    
    def create_match_documents_function(self):
        """
        Creates the match_documents function in Supabase if it doesn't exist.
        This is a helper method to set up the necessary database function.
        
        Note: This requires appropriate permissions in your Supabase project.
        """
        # SQL to create the match_documents function with the correct table name
        sql = f"""
        create or replace function public.match_documents (
            query_embedding vector(384),  -- Dimension for all-MiniLM-L6-v2 model
            match_threshold float,
            match_count int
            )
            returns table (
            id bigint,
            titulo text,
            documento text,
            similarity float
            )
            language plpgsql
            as $$
            begin
            -- Input validation
            if query_embedding is null then
                raise exception 'query_embedding cannot be null';
            end if;
            
            if match_threshold < 0 or match_threshold > 1 then
                raise exception 'match_threshold must be between 0 and 1';
            end if;
            
            if match_count <= 0 then
                raise exception 'match_count must be greater than 0';
            end if;

            return query
            select
                d.id,
                d.titulo,
                d.documento,
                1 - (d.embedding <=> query_embedding) as similarity
            from public.{self.table_name} d
            where 1 - (d.embedding <=> query_embedding) > match_threshold
            order by similarity desc
            limit match_count;
            end;
            $$;

        -- Add permissions for the function
        grant execute on function public.match_documents(vector(384), float, int) to anon, authenticated, service_role;
        """
        
        # Add table name to the SQL for better error messages
        sql = sql.replace("{self.table_name}", self.table_name)
        
        # Execute the SQL
        self.supabase.query(sql).execute()
        return "match_documents function created successfully"


# Example usage
if __name__ == "__main__":
    # Set your Supabase credentials
    SUPABASE_URL = "YOUR_SUPABASE_URL"
    SUPABASE_KEY = "YOUR_SUPABASE_API_KEY"
    
    # Initialize the SupabaseRAG client
    rag = SupabaseRAG(SUPABASE_URL, SUPABASE_KEY)
    
    # Create the match_documents function in Supabase (run once)
    # rag.create_match_documents_function()
    
    # Example: Insert a document
    response = rag.insert_document(
        title="Example Document",
        document_text="This is an example document about artificial intelligence and machine learning."
    )
    print("Document inserted:", response)
    
    # Example: Search for documents
    results = rag.search_documents("What is artificial intelligence?", top_k=3)
    print("Search results:", results)