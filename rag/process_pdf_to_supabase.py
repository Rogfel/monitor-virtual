import os
import re
import sys
from rag.supabase_rag import SupabaseRAG, get_embedding
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    list: (extracted text by page)
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_pages_text = []

    # Iterate through each page in the PDF
    for page in mypdf:
        # Extract text from the current page
        page_text = page.get_text("text")
        all_text = ''
        # Split into sentences to remove spaces
        sentences = page_text.split(". ")
        for sentence in sentences:
            if sentence.strip():  # Only add non-empty sentences
                all_text += sentence.strip() + ". "
        all_pages_text.append(all_text)
    # Return the extracted text
    return all_pages_text

def chunk_with_structure(page_text: str, page_number: int) -> list[dict]:
        """Chunking pages text with structure detection"""
        chunks_list = []
        
        lines = page_text.split('\n')
        current_section = {
                    "title": "",
                    "content": "",
                    "page_numbers": page_number
                }
        for line_index in range(len(lines)):
            line = lines[line_index].strip()
            if not line:
                continue
            
            # Heuristic to detect headers
            if (len(line) < 100 and 
                line[0].isupper() and 
                not line.endswith('.')):

                if current_section["content"] != "":
                    chunks_list.append(current_section)
                    current_section = {
                        "title": "",
                        "content": "",
                        "page_numbers": page_number
                    }
                                        
                # New section
                current_section["title"] = line
                current_section["content"] = lines[line_index+1]
                line_index += 1
            else:
                current_section["content"] += line + " "  
                
        if current_section["content"] != "" and is_content_valid(current_section["content"]):
            chunks_list.append(current_section)
        
        return chunks_list

def process_pdf_to_supabase(pdf_path, title_prefix="", supabase_url=None, supabase_key=None, min_alphanumeric_chars=None):
    """
    Process a PDF file and insert its chunks into Supabase.

    Args:
    pdf_path (str): Path to the PDF file.
    title_prefix (str): Prefix for the document titles.
    supabase_url (str): Supabase URL.
    supabase_key (str): Supabase API key.
    min_alphanumeric_chars (int): Minimum number of alphanumeric characters required. If None, uses MIN_ALPHANUMERIC_CHARS env var or defaults to 100.

    Returns:
    int: Number of chunks inserted.
    """
    # Set minimum alphanumeric characters from environment or parameter
    if min_alphanumeric_chars is None:
        min_alphanumeric_chars = int(os.getenv("MIN_ALPHANUMERIC_CHARS", "100"))
    
    print(f"Using minimum {min_alphanumeric_chars} alphanumeric characters for content validation.")
    
    # Extract text from the PDF file
    print(f"Extracting text from {pdf_path}...")
    extracted_text_list = extract_text_from_pdf(pdf_path)
    
    # Get the filename without extension for use in titles
    filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    chunks_list = []
    total_chunks_created = 0
    total_chunks_filtered = 0
    
    for page_number, page_text in enumerate(extracted_text_list):
        page_chunks = chunk_with_structure(page_text, page_number)
        total_chunks_created += len(page_chunks)
        
        # Filter chunks based on content validation
        valid_chunks = []
        for chunk in page_chunks:
            if is_content_valid(chunk["content"], min_alphanumeric_chars):
                valid_chunks.append(chunk)
            else:
                total_chunks_filtered += 1
                alphanumeric_count = count_alphanumeric_chars(chunk["content"])
                print(f"Filtered chunk with {alphanumeric_count} alphanumeric chars (min: {min_alphanumeric_chars}): {chunk['title'][:50]}...")
        
        chunks_list.extend(valid_chunks)
    
    print(f"Created {total_chunks_created} total chunks, filtered {total_chunks_filtered} chunks, kept {len(chunks_list)} valid chunks.")

    # Check if we have any valid chunks to process
    if not chunks_list:
        print("No valid chunks found. Please check your content validation criteria.")
        return 0

    # Add embeddings to chunks_list
    embeddings = get_embedding([chunk["content"] for chunk in chunks_list],
                               batch=True)
    print(f"Created {len(embeddings)} embeddings.")
    
    # Initialize Supabase RAG client
    rag = SupabaseRAG(supabase_url, supabase_key)
    
    # Insert chunks into Supabase using batch processing
    print("Inserting chunks into Supabase...")
    
    # Process chunks in batches to avoid overwhelming the database
    batch_size = 10
    for i in range(0, len(chunks_list), batch_size):
        batch_titles = [title_prefix] * batch_size
        batch_chunks = chunks_list[i:i+batch_size]
        batch_content = [chunk["content"] for chunk in batch_chunks]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_section_titles = [chunk["title"] for chunk in batch_chunks]
        batch_pages = [chunk["page_numbers"] for chunk in batch_chunks]
        batch_pdf_path = [pdf_path] * batch_size
        batch_nome_documento = [filename_without_ext] * batch_size
        
        try:
            # Use batch insertion for better performance
            rag.insert_documents_batch(
                batch_titles,
                batch_content,
                batch_embeddings,
                nome_documento=batch_nome_documento,
                paginas=batch_pages,
                pdf_path=batch_pdf_path,
                section_titles=batch_section_titles
            )
            print(f"Inserted batch {i//batch_size + 1}/{(len(chunks_list)-1)//batch_size + 1} " +
                  f"(chunks {i+1}-{min(i+batch_size, len(chunks_list))}/{len(chunks_list)})")
        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1}: {e}")
            
            # Fall back to individual insertion if batch fails
            for j, (title, chunk, embedding,
                    page_numbers, pdf_path,
                    nome_documento, section_title) in enumerate(zip(batch_titles, batch_content,
                                                     batch_embeddings, batch_pages,
                                                     batch_pdf_path, batch_nome_documento,
                                                     batch_section_titles)):
                try:
                    rag.insert_document(
                        title,
                        chunk,
                        embedding,
                        nome_documento=nome_documento,
                        pagina=page_numbers,
                        pdf_path=pdf_path,
                        titulo_secao=section_title
                    )
                    print(f"Inserted: {title_prefix} ({i+j+1}/{len(chunks_list)})")
                except Exception as e:
                    print(f"Error inserting chunk {i+j+1}: {e}")
    
    return len(chunks_list)

def count_alphanumeric_chars(text):
    """
    Count the number of alphanumeric characters in a text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        int: Number of alphanumeric characters
    """
    if not text:
        return 0
    # Remove all non-alphanumeric characters and count
    alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', text)
    return len(alphanumeric_only)

def is_content_valid(content, min_alphanumeric_chars=100):
    """
    Validate if content has enough alphanumeric characters.
    
    Args:
        content (str): The content to validate
        min_alphanumeric_chars (int): Minimum number of alphanumeric characters required
        
    Returns:
        bool: True if content is valid, False otherwise
    """
    if not content or not isinstance(content, str):
        return False
    
    alphanumeric_count = count_alphanumeric_chars(content)
    return alphanumeric_count >= min_alphanumeric_chars

def test_content_validation():
    """
    Test function to demonstrate content validation.
    """
    test_cases = [
        "This is a short text with only 50 alphanumeric characters.",
        "This is a longer text with more than 100 alphanumeric characters including numbers 12345 and more text to reach the minimum requirement for validation.",
        "Short text",
        "This text has exactly 100 alphanumeric characters including numbers 1234567890 and letters to test the boundary condition properly.",
        "",
        "   ",  # Only spaces
        "!@#$%^&*()",  # Only symbols
        "This is a valid text with 123 numbers and more than 100 alphanumeric characters total including spaces and punctuation marks."
    ]
    
    print("Testing content validation:")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        alphanumeric_count = count_alphanumeric_chars(text)
        is_valid = is_content_valid(text)
        print(f"Test {i}:")
        print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Alphanumeric chars: {alphanumeric_count}")
        print(f"  Valid: {is_valid}")
        print()

if __name__ == "__main__":
    # Test content validation if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test-validation":
        test_content_validation()
        exit(0)
    
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