import os
import uuid
import argparse
import logging
from datetime import datetime
import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer
import config

# PDF text extraction
import pdfplumber

# Reduzir verbosidade do asyncio
logging.getLogger('asyncio').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File type mappings (same as in endpoint.py)
ALLOWED_EXTENSIONS = {
    "pdf", "txt", "md", "html", "xml",  # documents
    "mp4", "mov", "avi", "wmv", "mpe", "mpeg", "mpg",  # videos
    "jpg", "jpeg", "png",  # images
    "mp3", "wav", "m4a",  # audio
    "csv", "xlsx"  # tabular
}

# Extensions to skip (compressed files that cannot be processed)
SKIP_EXTENSIONS = {
    "rar", "zip", "7z", "tar", "gz", "bz2"  # compressed files
}

def pdf_to_text(pdf_path):
    """Convert PDF file to text using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF, or None if extraction fails
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None


def pdf_to_temp_text_file(pdf_path):
    """Convert PDF to text and save to a temporary file.
    
    Pixeltable's Document type expects a file path, so we need to save
    the extracted text to a temporary file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Path to the temporary text file, or None if extraction fails
    """
    import tempfile
    import os
    
    try:
        text_content = pdf_to_text(pdf_path)
        if text_content is None:
            return None
        
        # Create a temporary file with .txt extension
        # Get the base name of the PDF without extension
        pdf_basename = os.path.basename(pdf_path)
        if pdf_basename.lower().endswith('.pdf'):
            pdf_basename = pdf_basename[:-4]
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_pdf_text')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temp file with original PDF name (sanitized) + .txt
        safe_name = "".join(c for c in pdf_basename if c.isalnum() or c in (' ', '-', '_')).strip()
        temp_file_path = os.path.join(temp_dir, f"{safe_name}.txt")
        
        # Write the text content to the file
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Created temporary text file: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        logger.error(f"Error creating temp text file from {pdf_path}: {str(e)}")
        return None

def get_file_type_and_column(file_path):
    """Determine file type and corresponding table column based on file extension."""
    file_ext = file_path.rsplit(".", 1)[1].lower() if "." in file_path else ""
    
    # Skip compressed files
    if file_ext in SKIP_EXTENSIONS:
        logger.debug(f"Skipping compressed file: {file_path}")
        return None, None
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return None, None
    
    if file_ext in {"pdf", "txt", "md", "html", "xml"}:
        return "document", "document"
    elif file_ext in {"mp4", "mov", "avi", "wmv", "mpe", "mpeg", "mpg"}:
        return "video", "video"
    elif file_ext in {"jpg", "jpeg", "png"}:
        return "image", "image"
    elif file_ext in {"mp3", "wav", "m4a"}:
        return "audio", "audio"
    elif file_ext in {"csv", "xlsx"}:
        return "tabular", "tabular"
    
    return None, None

def get_pxt_table(table_key):
    """Safely gets a Pixeltable table using the TABLE_MAP."""
    table_name = config.TABLE_MAP.get(table_key)
    if not table_name:
        raise ValueError(f"Invalid table key: {table_key}")
    try:
        return pxt.get_table(table_name)
    except Exception as e:
        logger.error(f"Error accessing Pixeltable table '{table_name}': {str(e)}")
        raise

def load_sources(file_path, table_key=None, user_id="local_user", recreate_chunks=True):
    """
    Load files from a directory into the appropriate Pixeltable tables.
    
    Args:
        file_path: Path to the directory containing files
        table_key: Optional specific table type. If None, will auto-detect for each file
        user_id: User ID to associate with the files
        recreate_chunks: Whether to recreate chunks view after loading documents.
                       Set to False to avoid corrupting agents.tools table.
    """
    logger.info(f"Starting load_sources for file_path: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Path does not exist: {file_path}")
        return
    
    if not os.path.isdir(file_path):
        logger.error(f"Path is not a directory: {file_path}")
        return

    # If table_key is specified, use the old behavior
    if table_key:
        logger.info(f"Using specified table_key: {table_key}")
        _load_sources_by_type(table_key, file_path, user_id, recreate_chunks)
    else:
        logger.info("Auto-detecting file types and loading into appropriate tables")
        _load_sources_auto_detect(file_path, user_id, recreate_chunks)

def _load_sources_by_type(table_key, file_path, user_id, recreate_chunks=True):
    """Load files of a specific type (original behavior).
    
    Args:
        table_key: Type of files to load (document, image, video, audio, tabular)
        file_path: Path to directory containing files
        user_id: User ID to associate with the files
        recreate_chunks: Whether to recreate chunks view after loading documents.
    """
    files_sources = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if (table_key == 'document' and file.lower().endswith((".pdf", ".txt", ".md", ".html", ".xml"))) or \
            (table_key == 'image' and file.lower().endswith((".jpg", ".jpeg", ".png"))) or \
            (table_key == 'video' and file.lower().endswith((".mp4", ".mov", ".avi", ".wmv", ".mpe", ".mpeg", ".mpg"))) or \
            (table_key == 'audio' and file.lower().endswith((".mp3", ".wav", ".m4a"))) or \
            (table_key == 'tabular' and file.lower().endswith((".csv", ".xlsx"))):
                    files_sources.append(os.path.join(root, file))

    logger.info(f"Found {len(files_sources)} {table_key} files to process")
    
    if not files_sources:
        logger.warning(f'No {table_key} files found in {file_path} or its subdirectories')
        return

    try:
        table = get_pxt_table(table_key)
        logger.info(f"Successfully connected to table: {config.TABLE_MAP[table_key]}")
    except Exception as e:
        logger.error(f'Error accessing table {table_key}: {str(e)}')
        return

    _insert_files_to_table(files_sources, table, table_key, user_id, recreate_chunks)

def _load_sources_auto_detect(file_path, user_id, recreate_chunks=True):
    """Auto-detect file types and load into appropriate tables.
    
    Args:
        file_path: Path to the directory containing files
        user_id: User ID to associate with the files
        recreate_chunks: Whether to recreate the chunks view after loading documents.
                        Set to False to avoid corrupting agents.tools table.
    """
    # Group files by type
    files_by_type = {
        "document": [],
        "image": [],
        "video": [],
        "audio": [],
        "tabular": []
    }
    
    for root, dirs, files in os.walk(file_path):
        for file in files:
            file_path_full = os.path.join(root, file)
            file_type, _ = get_file_type_and_column(file_path_full)
            
            if file_type:
                files_by_type[file_type].append(file_path_full)
            else:
                logger.warning(f"Skipping unsupported file: {file_path_full}")

    # Process each type
    total_processed = 0
    documents_processed = 0
    
    for file_type, files in files_by_type.items():
        if files:
            logger.info(f"Processing {len(files)} {file_type} files")
            try:
                table = get_pxt_table(file_type)
                logger.info(f"Successfully connected to table: {config.TABLE_MAP[file_type]}")
                _insert_files_to_table(files, table, file_type, user_id, recreate_chunks=False)
                total_processed += len(files)
                if file_type == "document":
                    documents_processed += len(files)
            except Exception as e:
                logger.error(f'Error processing {file_type} files: {str(e)}')
    
    logger.info(f"Completed auto-detect load_sources. Total files processed: {total_processed}")
    
    # Only recreate chunks view if explicitly requested and documents were processed
    if recreate_chunks and documents_processed > 0:
        logger.info(f"Documents processed ({documents_processed}), recreating chunks view...")
        _recreate_chunks_view()

def _recreate_chunks_view():
    """Safely recreate the chunks view with all documents, avoiding corruption of agents.tools."""
    try:
        logger.info("Checking chunks view status...")
        documents = pxt.get_table("agents.collection")
        
        # Try to get existing chunks view first
        chunks = None
        view_existed = False
        try:
            chunks = pxt.get_table("agents.chunks")
            if chunks is not None:
                view_existed = True
                # Check if it already has an embedding index
                try:
                    # Try to access the index - if it exists, we don't need to recreate
                    existing_idx = chunks.get_embedding_index("text_embedding")
                    if existing_idx is not None:
                        chunk_count = chunks.count()
                        logger.info(f"Chunks view already exists with embedding index ({chunk_count} chunks). Skipping recreation to avoid corrupting agents.tools.")
                        return True
                except Exception:
                    # Index doesn't exist, we'll add it
                    pass
                logger.info("Found existing chunks view without embedding index, will add index only")
        except Exception:
            # View doesn't exist, create it
            logger.info("Creating new chunks view...")
        
        # Only create view if it didn't exist
        if not view_existed:
            chunks = pxt.create_view(
                "agents.chunks",
                documents,
                iterator=document_splitter(
                    document=documents.document,
                    separators="page",
                    metadata="title, heading, page"
                ),
                if_exists="ignore",
            )
        
        # Add or update embedding index with error handling
        # Use "ignore" to avoid replacing existing index which can corrupt dependent tables like agents.tools
        try:
            chunks.add_embedding_index(
                "text",
                idx_name="text_embedding",
                string_embed=sentence_transformer.using(
                    model_id=config.EMBEDDING_MODEL_ID,
                ),
                if_exists="ignore",
            )
            
            chunk_count = chunks.count()
            logger.info(f"✅ Successfully updated chunks view with {chunk_count} chunks")
            return True
            
        except Exception as embedding_error:
            logger.error(f"Error adding embedding index: {embedding_error}")
            logger.info("Chunks view exists but without embedding index")
            return False
        
    except Exception as e:
        logger.error(f"Error recreating chunks view: {str(e)}")
        return False

def _insert_files_to_table(files_sources, table, table_key, user_id, recreate_chunks=True):
    """Insert files into the specified table.
    
    Args:
        files_sources: List of file paths to insert
        table: Pixeltable table to insert into
        table_key: Type of file (document, image, video, audio, tabular)
        user_id: User ID to associate with the files
        recreate_chunks: Whether to recreate chunks view after loading documents.
                       Set to False to avoid corrupting agents.tools table.
    """
    successful_inserts = 0
    
    for file_source in files_sources:
        try:
            # Validate file exists
            if not os.path.exists(file_source):
                logger.warning(f"Warning: File {file_source} does not exist, skipping...")
                continue
            
            file_uuid = str(uuid.uuid4())
            current_timestamp = datetime.now()
            
            # Check if file is a PDF - convert to text to avoid Pixeltable paragraph splitting error
            file_ext = file_source.rsplit(".", 1)[1].lower() if "." in file_source else ""
            document_value = file_source
            
            if file_ext == "pdf":
                # Convert PDF to text file to avoid "Paragraph splitting is not currently supported for PDF documents" error
                # Pixeltable's Document type expects a file path, so we save the extracted text to a temp file
                logger.info(f"Converting PDF to text: {file_source}")
                temp_text_file = pdf_to_temp_text_file(file_source)
                if temp_text_file is None:
                    logger.error(f"Failed to convert PDF to text, skipping: {file_source}")
                    continue
                document_value = temp_text_file
            
            table.insert(
                [{table_key: document_value, "uuid": file_uuid, "timestamp": current_timestamp, "user_id": user_id}]
            )
            logger.info(f"Successfully loaded {file_source} into {table_key}")
            successful_inserts += 1
            
        except Exception as e:
            logger.error(f"Error loading {file_source}: {str(e)}")
            # Continue with next file instead of stopping
            continue
    
    logger.info(f"Completed loading {table_key} files. Successfully inserted {successful_inserts}/{len(files_sources)} files.")
    
    # Only recreate chunks view if explicitly requested
    if recreate_chunks and table_key == "document" and successful_inserts > 0:
        logger.info("Documents inserted, recreating chunks view...")
        _recreate_chunks_view()

def load_all_from_data(data_path="data", user_id="local_user", recreate_chunks=True):
    """
    Load all supported documents and videos from the data/ folder.
    
    Args:
        data_path: Path to the data folder (default: "data")
        user_id: User ID to associate with the files
        recreate_chunks: Whether to recreate chunks view after loading documents.
                        Set to False to avoid corrupting agents.tools table.
    
    Returns:
        Dictionary with counts of loaded files by type
    """
    logger.info(f"Loading all sources from: {data_path}")
    
    # Verify data folder exists
    if not os.path.exists(data_path):
        logger.error(f"Data folder does not exist: {data_path}")
        return {"error": f"Data folder does not exist: {data_path}"}
    
    # Use auto-detect to load all file types
    load_sources(data_path, table_key=None, user_id=user_id, recreate_chunks=recreate_chunks)
    
    # Return summary
    return {
        "status": "completed",
        "source_folder": data_path,
        "message": "All files loaded successfully. Check logs for details."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carregamento de arquivos para o Monitor Virtual")
    parser.add_argument("--file_path", type=str, help="Caminho para a pasta com os arquivos (padrão: data/)", default="data/")
    parser.add_argument("--table_key", choices=["document", "video", "image", "audio", "tabular"], 
                       help="Chave da tabela para carregamento dos arquivos (opcional - se não especificado, detecta automaticamente)")
    parser.add_argument("--user_id", type=str, default="local_user", 
                       help="ID do usuário para associar aos arquivos (padrão: local_user)")
    parser.add_argument("--data", action="store_true", 
                       help="Carregar todos os arquivos da pasta data/ (atalho para --file_path data/)")
    parser.add_argument("--recreate-chunks", dest="recreate_chunks", action="store_true", default=True,
                       help="Recriar a view de chunks após carregar documentos (padrão: True, use --no-recreate-chunks para desativar)")
    parser.add_argument("--no-recreate-chunks", dest="recreate_chunks", action="store_false",
                       help="Não recriar a view de chunks após carregar documentos (evita corromper agents.tools)")
    args = parser.parse_args()
    
    # If --data flag is set, use data/ folder
    if args.data:
        logger.info(f"Using --data flag: loading from data/ folder (recreate_chunks={args.recreate_chunks})")
        load_all_from_data("data/", args.user_id, args.recreate_chunks)
    else:
        load_sources(args.file_path, args.table_key, args.user_id, args.recreate_chunks)