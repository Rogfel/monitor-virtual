import os
import glob
from supabase_rag import SupabaseRAG
from process_pdf_to_supabase import process_pdf_to_supabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_search():
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
    
    # Example searches
    search_queries = [
        "Que é a física do arco?",
        "Quais são os distintos modos de tranferência?",
        "Dime os principais elementos do arco"
    ]
    
    # Perform searches
    print("\nPerforming searches...")
    for query in search_queries:
        # try:
        print(f"\nSearch query: '{query}'")
        results = rag.search_documents(query, top_k=10)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. Title: {result['titulo']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Document: {result['documento'][:100]}...")
        else:
            print("No results found.")
        # except Exception as e:
        #     print(f"Error searching for '{query}': {e}")


def loading_from_path():
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
    
    
    # Insert example documents
    print("\nInserting documents...")
    for pdf_path in glob.glob(os.path.join("data/", "*.pdf")):
        try:
            num_chunks = process_pdf_to_supabase(
                pdf_path, 
                pdf_path[:-4].split('/')[-1], 
                supabase_url, 
                supabase_key
            )
            print(f"Inserted: {pdf_path} with {num_chunks} chunks")
        except Exception as e:
            print(f"Error inserting document '{pdf_path}': {e}")


if __name__ == "__main__":
    # loading_from_path()
    test_search()