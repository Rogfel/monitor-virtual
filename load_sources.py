import os
import uuid
import argparse
import logging
from datetime import datetime
import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File type mappings (same as in endpoint.py)
ALLOWED_EXTENSIONS = {
    "pdf", "jpg", "jpeg", "png", "mp4", "mov", "avi", "txt", "md", 
    "html", "xml", "mp3", "wav", "m4a", "csv", "xlsx"
}

def get_file_type_and_column(file_path):
    """Determine file type and corresponding table column based on file extension."""
    file_ext = file_path.rsplit(".", 1)[1].lower() if "." in file_path else ""
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return None, None
    
    if file_ext in {"pdf", "txt", "md", "html", "xml"}:
        return "document", "document"
    elif file_ext in {"mp4", "mov", "avi"}:
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

def load_sources(file_path, table_key=None, user_id="local_user"):
    """
    Load files from a directory into the appropriate Pixeltable tables.
    
    Args:
        file_path: Path to the directory containing files
        table_key: Optional specific table type. If None, will auto-detect for each file
        user_id: User ID to associate with the files
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
        _load_sources_by_type(table_key, file_path, user_id)
    else:
        logger.info("Auto-detecting file types and loading into appropriate tables")
        _load_sources_auto_detect(file_path, user_id)

def _load_sources_by_type(table_key, file_path, user_id):
    """Load files of a specific type (original behavior)."""
    files_sources = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if (table_key == 'document' and file.lower().endswith((".pdf", ".txt", ".md", ".html", ".xml"))) or \
            (table_key == 'image' and file.lower().endswith((".jpg", ".jpeg", ".png"))) or \
            (table_key == 'video' and file.lower().endswith((".mp4", ".mov", ".avi"))) or \
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

    _insert_files_to_table(files_sources, table, table_key, user_id)

def _load_sources_auto_detect(file_path, user_id):
    """Auto-detect file types and load into appropriate tables."""
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
                _insert_files_to_table(files, table, file_type, user_id)
                total_processed += len(files)
                if file_type == "document":
                    documents_processed += len(files)
            except Exception as e:
                logger.error(f'Error processing {file_type} files: {str(e)}')
    
    logger.info(f"Completed auto-detect load_sources. Total files processed: {total_processed}")
    
    # If documents were processed, recreate chunks view
    if documents_processed > 0:
        logger.info(f"Documents processed ({documents_processed}), recreating chunks view...")
        _recreate_chunks_view()

def _recreate_chunks_view():
    """Recreate the chunks view with all documents."""
    try:
        logger.info("Recreating chunks view...")
        documents = pxt.get_table("agents.collection")
        
        # Drop existing chunks view if it exists
        try:
            pxt.drop_table("agents.chunks")
            logger.info("Dropped existing chunks view")
        except Exception as drop_error:
            logger.warning(f"Could not drop existing chunks view: {drop_error}")
        
        # Create new chunks view with error handling
        try:
            chunks = pxt.create_view(
                "agents.chunks",
                documents,
                iterator=DocumentSplitter.create(
                    document=documents.document,
                    separators="paragraph",
                    metadata="title, heading, page"
                ),
                if_exists="replace",
            )
            
            # Add embedding index with error handling
            try:
                chunks.add_embedding_index(
                    "text",
                    string_embed=sentence_transformer.using(
                        model_id=config.EMBEDDING_MODEL_ID,
                    ),
                    if_exists="replace",
                )
                
                chunk_count = chunks.count()
                logger.info(f"✅ Successfully recreated chunks view with {chunk_count} chunks")
                
                return True
                
            except Exception as embedding_error:
                logger.error(f"Error adding embedding index: {embedding_error}")
                logger.info("Chunks view created but without embedding index")
                return False
                
        except Exception as view_error:
            logger.error(f"Error creating chunks view: {view_error}")
            return False
        
    except Exception as e:
        logger.error(f"Error recreating chunks view: {str(e)}")
        return False

def _insert_files_to_table(files_sources, table, table_key, user_id):
    """Insert files into the specified table."""
    successful_inserts = 0
    
    for file_source in files_sources:
        try:
            # Validate file exists
            if not os.path.exists(file_source):
                logger.warning(f"Warning: File {file_source} does not exist, skipping...")
                continue
                
            file_uuid = str(uuid.uuid4())
            current_timestamp = datetime.now()
            
            table.insert(
                [{table_key: file_source, "uuid": file_uuid, "timestamp": current_timestamp, "user_id": user_id}]
            )
            logger.info(f"Successfully loaded {file_source} into {table_key}")
            successful_inserts += 1
            
        except Exception as e:
            logger.error(f"Error loading {file_source}: {str(e)}")
            # Continue with next file instead of stopping
            continue
    
    logger.info(f"Completed loading {table_key} files. Successfully inserted {successful_inserts}/{len(files_sources)} files.")
    
    # If documents were inserted, recreate chunks view
    if table_key == "document" and successful_inserts > 0:
        logger.info("Documents inserted, recreating chunks view...")
        _recreate_chunks_view()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carregamento de arquivos para o Monitor Virtual")
    parser.add_argument("--file_path", type=str, help="Caminho para a pasta com os arquivos", required=True)
    parser.add_argument("--table_key", choices=["document", "video", "image", "audio", "tabular"], 
                       help="Chave da tabela para carregamento dos arquivos (opcional - se não especificado, detecta automaticamente)")
    parser.add_argument("--user_id", type=str, default="local_user", 
                       help="ID do usuário para associar aos arquivos (padrão: local_user)")
    args = parser.parse_args()
    
    load_sources(args.file_path, args.table_key, args.user_id)