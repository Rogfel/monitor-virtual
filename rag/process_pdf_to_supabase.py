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

        chunks_list.append(current_section)
        
        return chunks_list

def split_text_into_semantic_chunks(text, page_numbers):
    """
    Splits text into semantic chunks based on similarity.

    Args:
    text (str): Text to split.
    page_numbers (list): List of page numbers corresponding to each sentence.

    Returns:
    tuple: (list of text chunks, list of chunk metadata)
    """
    # Splitting text into sentences (basic split)
    sentences = chunk_with_structure(text)
    print(f"Number of sentences: {len(sentences)}")
    
   
    
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
    
    # Create chunks and metadata
    chunks = []
    chunk_metadata = []
    start = 0
    
    # Iterate through each breakpoint to create chunks
    for bp in breakpoints:
        # Append the chunk of sentences from start to the current breakpoint
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        # Store the page numbers for this chunk
        chunk_metadata.append({
            "page_numbers": page_numbers[start:bp + 1]
        })
        start = bp + 1
    
    # Append the remaining sentences as the last chunk
    if start < len(sentences):
        chunks.append(". ".join(sentences[start:]))
        chunk_metadata.append({
            "page_numbers": page_numbers[start:]
        })
    
    return chunks, chunk_metadata

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
    extracted_text_list = extract_text_from_pdf(pdf_path)
    
    # Get the filename without extension for use in titles
    filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    chunks_list = []
    for page_number, page_text in enumerate(extracted_text_list):
        page_chunks = chunk_with_structure(page_text, page_number)
        chunks_list.extend(page_chunks)
    print(f"Created {len(chunks_list)} sections chunks.")


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