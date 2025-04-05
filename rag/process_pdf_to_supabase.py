import os
import fitz  # PyMuPDF
import numpy as np
from rag.supabase_rag import SupabaseRAG, get_embedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text
    
    # Iterate through each page in the PDF
    for page in mypdf:
        # Extract text from the current page and add spacing
        all_text += page.get_text("text") + " "

    # Return the extracted text, stripped of leading/trailing whitespace
    return all_text.strip()

def split_text_into_semantic_chunks(text, threshold=90):
    """
    Splits text into semantic chunks based on similarity.

    Args:
    text (str): Text to split.
    threshold (int): Percentile threshold for splitting.

    Returns:
    list: List of text chunks.
    """
    # Splitting text into sentences (basic split)
    sentences = text.split(". ")
    print(f"Number of sentences: {len(sentences)}")
    
    # If there are too few sentences, return the whole text as one chunk
    if len(sentences) <= 5:
        return [text]
    
    # Generate embeddings for all sentences at once using batch processing
    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = get_embedding(sentences, batch=True)
    print(f"Generated {len(embeddings)} sentence embeddings.")
    
    # Compute similarity between consecutive sentences using vectorized operations
    print("Computing similarities between consecutive sentences...")
    similarities = []
    
    # Pre-compute norms for all vectors at once
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Calculate all similarities at once using vectorized operations
    embeddings_array = np.array(embeddings)
    # Dot product between each vector and the next one
    dot_products = np.sum(embeddings_array[:-1] * embeddings_array[1:], axis=1)
    # Element-wise product of consecutive norms
    norm_products = norms[:-1] * norms[1:]
    # Calculate all similarities at once
    similarities = dot_products / norm_products
    
    # Compute breakpoints using the percentile method
    threshold_value = np.percentile(similarities, threshold)
    breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold_value]
    
    # Create chunks
    chunks = []
    start = 0
    
    # Iterate through each breakpoint to create chunks
    for bp in breakpoints:
        # Append the chunk of sentences from start to the current breakpoint
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1
    
    # Append the remaining sentences as the last chunk
    if start < len(sentences):
        chunks.append(". ".join(sentences[start:]))
    
    return chunks

def process_pdf_to_supabase(pdf_path, title_prefix="", supabase_url=None, supabase_key=None):
    """
    Process a PDF file and insert its chunks into Supabase.

    Args:
    pdf_path (str): Path to the PDF file.
    title_prefix (str): Prefix for the document titles.
    supabase_url (str): Supabase URL.
    supabase_key (str): Supabase API key.

    Returns:
    int: Number of chunks inserted.
    """
    # Extract text from the PDF file
    print(f"Extracting text from {pdf_path}...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Get the filename without extension for use in titles
    filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    if not title_prefix:
        title_prefix = filename_without_ext
    
    # Split text into semantic chunks
    print("Splitting text into semantic chunks...")
    text_chunks = split_text_into_semantic_chunks(extracted_text)
    print(f"Created {len(text_chunks)} semantic chunks.")
    # Initialize Supabase RAG client
    rag = SupabaseRAG(supabase_url, supabase_key)
    
    # Ensure the match_documents function exists with the correct table name
    try:
        rag.create_match_documents_function()
        print("Created or updated match_documents function.")
    except Exception as e:
        print(f"Warning: Could not create match_documents function: {e}")
        print("Proceeding with document insertion anyway...")
    
    # Insert chunks into Supabase using batch processing
    print("Inserting chunks into Supabase...")
    
    # Pre-generate all titles
    chunk_titles = [f"{title_prefix} - Chunk {i+1}" for i in range(len(text_chunks))]
    
    # Process chunks in batches to avoid overwhelming the database
    batch_size = 10
    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i+batch_size]
        batch_titles = chunk_titles[i:i+batch_size]
        
        try:
            # Use batch insertion for better performance
            rag.insert_documents_batch(batch_titles, batch_chunks)
            print(f"Inserted batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} " +
                  f"(chunks {i+1}-{min(i+batch_size, len(text_chunks))}/{len(text_chunks)})")
        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1}: {e}")
            
            # Fall back to individual insertion if batch fails
            for j, (chunk_title, chunk) in enumerate(zip(batch_titles, batch_chunks)):
                try:
                    rag.insert_document(chunk_title, chunk)
                    print(f"Inserted: {chunk_title} ({i+j+1}/{len(text_chunks)})")
                except Exception as e:
                    print(f"Error inserting chunk {i+j+1}: {e}")
    
    return len(text_chunks)

if __name__ == "__main__":
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Please set your SUPABASE_URL and SUPABASE_KEY environment variables")
        exit(1)
    
    # Process a PDF file
    pdf_path = input("Enter the path to the PDF file: ")
    title_prefix = input("Enter a title prefix for the documents (optional): ")
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        exit(1)
    
    try:
        # Create the SupabaseRAG client to check if the match_documents function needs to be created
        rag = SupabaseRAG(supabase_url, supabase_key)
        rag.create_match_documents_function()
        print("Created or updated match_documents function.")
    except Exception as e:
        print(f"Warning: Could not create match_documents function: {e}")
        print("Proceeding with document processing anyway...")
    
    # Process the PDF
    num_chunks = process_pdf_to_supabase(
        pdf_path,
        title_prefix,
        supabase_url,
        supabase_key
    )
    
    print(f"Successfully processed PDF and inserted {num_chunks} chunks into Supabase.")
    print("You can now search these documents using the SupabaseRAG.search_documents() method.")