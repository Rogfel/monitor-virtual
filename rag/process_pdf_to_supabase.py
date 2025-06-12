import os
import fitz  # PyMuPDF
from rag.raptor_supabase import RaptorSupabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    tuple: (extracted text, list of page numbers for each sentence)
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text
    page_numbers = []  # Initialize a list to store page numbers for each sentence
    
    # Iterate through each page in the PDF
    for page_num, page in enumerate(mypdf, 1):
        # Extract text from the current page
        page_text = page.get_text("text")
        # Split into sentences and add page number for each
        sentences = page_text.split(". ")
        for sentence in sentences:
            if sentence.strip():  # Only add non-empty sentences
                all_text += sentence.strip() + ". "
                page_numbers.append(page_num)

    # Return the extracted text and page numbers
    return all_text.strip(), page_numbers

def process_pdf_to_supabase(pdf_path, title_prefix="", supabase_url=None, supabase_key=None):
    """
    Process a PDF file and insert its chunks into Supabase using RAPTOR.

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
    extracted_text, page_numbers = extract_text_from_pdf(pdf_path)
    
    # Get the filename without extension for use in titles
    filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    if not title_prefix:
        title_prefix = filename_without_ext
    
    # Initialize RaptorSupabase client
    raptor = RaptorSupabase(supabase_url, supabase_key)
    
    # Ensure the match_raptor_nodes function exists
    try:
        raptor.create_match_function()
        print("Created or updated match_raptor_nodes function.")
    except Exception as e:
        print(f"Warning: Could not create match_raptor_nodes function: {e}")
        print("Proceeding with document processing anyway...")
    
    # Process and store document using RAPTOR
    print("Processing document with RAPTOR...")
    try:
        document_id = raptor.store_document(
            extracted_text,
            metadata={
                "title": title_prefix,
                "filename": filename,
                "page_numbers": page_numbers
            }
        )
        print(f"Successfully processed and stored document with ID: {document_id}")
        return document_id
    except Exception as e:
        print(f"Error processing document: {e}")
        return None

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
    
    # Process the PDF
    document_id = process_pdf_to_supabase(
        pdf_path,
        title_prefix,
        supabase_url,
        supabase_key
    )
    
    if document_id:
        print(f"Successfully processed PDF and stored in Supabase with document ID: {document_id}")
        print("You can now search these documents using the RaptorSupabase.search() method.")
    else:
        print("Failed to process PDF.")