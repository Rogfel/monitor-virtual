import os
import argparse
from rag.supabase_rag import SupabaseRAG
from rag.process_pdf_to_supabase import process_pdf_to_supabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def search(queries=None, top_k=os.getenv("MATCH_COUNT"),
           match_threshold=os.getenv("MATCH_THRESHOLD")):
    """
    Perform searches using the SupabaseRAG client.
    
    Args:
        queries (list or str): Search queries. If None, default queries will be used.
        top_k (int): Number of top results to return for each query.
        match_threshold (float): Minimum similarity score for a match.
    Returns:
        dict: Dictionary containing search results.
    """
    # Get Supabase credentials from environment variables or set them directly
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Please set your SUPABASE_URL and SUPABASE_KEY environment variables")
        print("You can add them to your .env file or set them directly in this script")
        
        # Uncomment and set your credentials here if not using environment variables
        # supabase_url = "YOUR_SUPABASE_URL"
        # supabase_key = "YOUR_SUPABASE_API_KEY"
        return {"error": "Missing Supabase credentials"}
    
    # Initialize the SupabaseRAG client
    rag = SupabaseRAG(supabase_url, supabase_key)
    
    # Handle string query by converting to list
    if isinstance(queries, str):
        queries = [queries]
    
    # Use default queries if none provided
    if queries is None:
        queries = [
            "Que é a física do arco?",
            "Quais são os distintos modos de tranferência?",
            "Dime os principais elementos do arco"
        ]
    
    # Prepare response dictionary
    response = {"results": []}
    
    # Perform searches
    print("\nPerforming searches...")
    for query in queries:
        try:
            print(f"\nSearch query: '{query}'")
            results = rag.search_documents(query)
            
            query_results = {
                "query": query,
                "matches": []
            }
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. Title: {result['titulo']}")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   Document: {result['documento'][:100]}...")
                    
                    # Add result to response
                    query_results["matches"].append({
                        "title": result['titulo'],
                        "similarity": result['similarity'],
                        "document": result['documento']
                    })
            else:
                print("No results found.")
                
            response["results"].append(query_results)
        except Exception as e:
            error_msg = f"Error searching for '{query}': {str(e)}"
            print(error_msg)
            response["results"].append({
                "query": query,
                "error": error_msg,
                "matches": []
            })
    
    return response


def loading_from_path(data_path="data/", create_function=True):
    """
    Load documents from a specified path into Supabase.
    
    Args:
        data_path (str): Path to the directory containing PDF files.
        create_function (bool): Whether to create the match_documents function in Supabase.
    """
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
    if create_function:
        try:
            print("Creating match_documents function in Supabase...")
            result = rag.create_match_documents_function()
            print(result)
        except Exception as e:
            print(f"Error creating function: {e}")
    
    # Insert documents
    print("\nInserting documents in recursive mode...")
    pdf_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in {data_path} or its subdirectories")
        return
        
    for pdf_path in pdf_files:
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


def main():
    """Main function to parse command line arguments and execute the appropriate function."""
    parser = argparse.ArgumentParser(description="Supabase RAG operations")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Parser for loading_from_path command
    load_parser = subparsers.add_parser("from-path", help="Load documents from a path into Supabase")
    load_parser.add_argument("--path", default="data/", help="Path to the directory containing PDF files")
    load_parser.add_argument("--no-create-function", action="store_true",
                            help="Skip creating the match_documents function in Supabase")
    
    # Parser for test_search command
    search_parser = subparsers.add_parser("search", help="Test search functionality")
    search_parser.add_argument("--queries", nargs="+", help="Search queries to test")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of top results to return")
    
    args = parser.parse_args()
    
    if args.command == "from-path":
        loading_from_path(data_path=args.path, create_function=not args.no_create_function)
    elif args.command == "search":
        search(queries=args.queries, top_k=args.top_k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()