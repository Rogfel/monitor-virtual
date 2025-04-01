import os
from supabase_rag import SupabaseRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get Supabase credentials from environment variables or set them directly
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Please set your SUPABASE_URL and SUPABASE_KEY environment variables")
        print("You can add them to your .env file or set them directly in this script")
        
        # Uncomment and set your credentials here if not using environment variables
        # supabase_url = "YOUR_SUPABASE_URL"
        # supabase_key = "YOUR_SUPABASE_API_KEY"
        return
    
    # Initialize the SupabaseRAG client
    rag = SupabaseRAG(supabase_url, supabase_key)
    
    # Create the match_documents function in Supabase (only need to run this once)
    try:
        print("Creating match_documents function in Supabase...")
        result = rag.create_match_documents_function()
        print(result)
    except Exception as e:
        print(f"Error creating function: {e}")
        print("If you've already created this function, you can ignore this error.")
    
    # Example documents to insert
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "text": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        },
        {
            "title": "Natural Language Processing",
            "text": "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages."
        },
        {
            "title": "Computer Vision",
            "text": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world."
        }
    ]
    
    # Insert example documents
    print("\nInserting documents...")
    for doc in documents:
        try:
            response = rag.insert_document(doc["title"], doc["text"])
            print(f"Inserted: {doc['title']}")
        except Exception as e:
            print(f"Error inserting document '{doc['title']}': {e}")
    
    # Example searches
    search_queries = [
        "What is machine learning?",
        "How does NLP work?",
        "Tell me about computer vision"
    ]
    
    # Perform searches
    print("\nPerforming searches...")
    for query in search_queries:
        try:
            print(f"\nSearch query: '{query}'")
            results = rag.search_documents(query, top_k=2)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. Title: {result['titulo']}")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   Document: {result['documento'][:100]}...")
            else:
                print("No results found.")
        except Exception as e:
            print(f"Error searching for '{query}': {e}")

