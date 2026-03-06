#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir problemas no banco de dados Pixeltable.
Este script identifica problemas comuns e tenta corrigi-los automaticamente.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_health():
    """Check the overall health of the database."""
    try:
        import pixeltable as pxt
        
        # Test basic connectivity
        logger.info("🔍 Checking database connectivity...")
        
        # Try to access main tables
        tables_to_check = [
            "agents.tools",
            "agents.collection", 
            "agents.images",
            "agents.videos",
            "agents.audios",
            "agents.tabular"
        ]
        
        healthy_tables = []
        problematic_tables = []
        
        for table_name in tables_to_check:
            try:
                table = pxt.get_table(table_name)
                if table is not None:
                    healthy_tables.append(table_name)
                    logger.info(f"✅ {table_name} - OK")
                else:
                    problematic_tables.append(table_name)
                    logger.warning(f"❌ {table_name} - Table is None")
            except Exception as e:
                problematic_tables.append(table_name)
                logger.error(f"❌ {table_name} - Error: {str(e)}")
        
        # Check views
        views_to_check = [
            "agents.chunks",
            "agents.video_frames",
            "agents.video_audio_chunks",
            "agents.video_transcript_sentences",
            "agents.audio_chunks",
            "agents.audio_transcript_sentences"
        ]
        
        healthy_views = []
        problematic_views = []
        
        for view_name in views_to_check:
            try:
                view = pxt.get_table(view_name)
                if view is not None:
                    healthy_views.append(view_name)
                    logger.info(f"✅ {view_name} - OK")
                else:
                    problematic_views.append(view_name)
                    logger.warning(f"❌ {view_name} - View is None")
            except Exception as e:
                problematic_views.append(view_name)
                logger.error(f"❌ {view_name} - Error: {str(e)}")
        
        return {
            'healthy_tables': healthy_tables,
            'problematic_tables': problematic_tables,
            'healthy_views': healthy_views,
            'problematic_views': problematic_views
        }
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {str(e)}")
        return None

def check_chunks_status():
    """Check the status of chunks created in the database."""
    try:
        import pixeltable as pxt
        
        logger.info("🔍 Checking chunks status...")
        
        chunks_info = {}
        
        # Check different types of chunks
        chunk_types = {
            "agents.chunks": "Document chunks",
            "agents.video_frames": "Video frame chunks", 
            "agents.video_audio_chunks": "Video audio chunks",
            "agents.video_transcript_sentences": "Video transcript chunks",
            "agents.audio_chunks": "Audio chunks",
            "agents.audio_transcript_sentences": "Audio transcript chunks"
        }
        
        total_chunks = 0
        chunks_by_type = {}
        
        for chunk_view, description in chunk_types.items():
            try:
                view = pxt.get_table(chunk_view)
                if view is not None:
                    # Count rows in the view using .count() method
                    row_count = view.count()
                    chunks_by_type[description] = row_count
                    total_chunks += row_count
                    
                    if row_count > 0:
                        logger.info(f"✅ {description}: {row_count:,} chunks")
                    else:
                        logger.warning(f"⚠️  {description}: No chunks found")
                        
                    # Get sample data for analysis
                    if row_count > 0:
                        sample_data = view.head(1)
                        logger.info(f"   📝 Sample data available for {description}")
                        
                else:
                    chunks_by_type[description] = 0
                    logger.error(f"❌ {description}: View not accessible")
                    
            except Exception as e:
                chunks_by_type[description] = 0
                logger.error(f"❌ {description}: Error - {str(e)}")
        
        # Check collection table for overall document count
        try:
            collection = pxt.get_table("agents.collection")
            if collection is not None:
                collection_count = collection.count()
                logger.info(f"📚 Total documents in collection: {collection_count:,}")
            else:
                collection_count = 0
                logger.warning("⚠️  Collection table not accessible")
        except Exception as e:
            collection_count = 0
            logger.error(f"❌ Error accessing collection: {str(e)}")
        
        # Check individual media tables
        media_tables = {
            "agents.images": "Images",
            "agents.videos": "Videos",
            "agents.audios": "Audios",
            "agents.tabular": "Tabular data"
        }
        
        media_counts = {}
        for table_name, media_type in media_tables.items():
            try:
                table = pxt.get_table(table_name)
                if table is not None:
                    count = table.count()
                    media_counts[media_type] = count
                    logger.info(f"📁 {media_type}: {count:,} files")
                else:
                    media_counts[media_type] = 0
                    logger.warning(f"⚠️  {media_type} table not accessible")
            except Exception as e:
                media_counts[media_type] = 0
                logger.error(f"❌ Error accessing {media_type}: {str(e)}")
        
        chunks_info = {
            'total_chunks': total_chunks,
            'chunks_by_type': chunks_by_type,
            'collection_count': collection_count,
            'media_counts': media_counts
        }
        
        logger.info(f"📊 Chunks Summary:")
        logger.info(f"   Total chunks: {total_chunks:,}")
        logger.info(f"   Total documents: {collection_count:,}")
        
        return chunks_info
        
    except Exception as e:
        logger.error(f"❌ Chunks status check failed: {str(e)}")
        return None

def repair_database():
    """Attempt to repair the database."""
    try:
        logger.info("🔧 Attempting database repair...")
        
        # Check if setup script exists
        setup_script = Path("setup_pixeltable.py")
        if not setup_script.exists():
            logger.error("❌ setup_pixeltable.py not found")
            return False
        
        # Run the setup script
        logger.info("🔄 Running database setup script...")
        result = subprocess.run([sys.executable, "setup_pixeltable.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Database repair completed successfully")
            return True
        else:
            logger.error(f"❌ Database repair failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during database repair: {str(e)}")
        return False

def clean_database():
    """Clean and recreate the database from scratch."""
    try:
        logger.info("🧹 Cleaning database completely...")
        
        import pixeltable as pxt
        
        # Drop the entire agents directory
        try:
            pxt.drop_dir("agents", force=True)
            logger.info("✅ Dropped agents directory")
        except Exception as e:
            logger.warning(f"Could not drop agents directory: {e}")
        
        # Run setup to recreate everything
        return repair_database()
        
    except Exception as e:
        logger.error(f"❌ Error during database cleanup: {str(e)}")
        return False

def main():
    """Main diagnostic function."""
    logger.info("🚀 Starting database diagnosis...")
    
    # Check database health
    health_status = check_database_health()
    
    if health_status is None:
        logger.error("❌ Cannot connect to database")
        return False
    
    # Check chunks status
    chunks_status = check_chunks_status()
    
    # Analyze results
    total_tables = len(health_status['healthy_tables']) + len(health_status['problematic_tables'])
    total_views = len(health_status['healthy_views']) + len(health_status['problematic_views'])
    
    logger.info(f"📊 Database Health Summary:")
    logger.info(f"   Tables: {len(health_status['healthy_tables'])}/{total_tables} healthy")
    logger.info(f"   Views: {len(health_status['healthy_views'])}/{total_views} healthy")
    
    if chunks_status:
        logger.info(f"   Total chunks: {chunks_status['total_chunks']:,}")
        logger.info(f"   Total documents: {chunks_status['collection_count']:,}")
    
    if health_status['problematic_tables'] or health_status['problematic_views']:
        logger.warning("⚠️  Database has problems that need attention")
        
        # Ask user what to do
        print("\n" + "="*50)
        print("DATABASE PROBLEMS DETECTED")
        print("="*50)
        print("1. Attempt repair (recommended)")
        print("2. Clean and recreate database (WARNING: will lose all data)")
        print("3. Check chunks status only")
        print("4. Exit without changes")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                logger.info("🔄 Attempting repair...")
                if repair_database():
                    logger.info("✅ Repair completed successfully")
                    # Re-check health
                    new_health = check_database_health()
                    if new_health and not new_health['problematic_tables'] and not new_health['problematic_views']:
                        logger.info("✅ Database is now healthy!")
                        # Re-check chunks status
                        new_chunks = check_chunks_status()
                        return True
                    else:
                        logger.warning("⚠️  Some problems may still exist")
                        return False
                else:
                    logger.error("❌ Repair failed")
                    return False
                    
            elif choice == "2":
                confirm = input("⚠️  WARNING: This will delete ALL data. Type 'YES' to confirm: ")
                if confirm == "YES":
                    logger.info("🧹 Cleaning database...")
                    if clean_database():
                        logger.info("✅ Database cleaned and recreated successfully")
                        return True
                    else:
                        logger.error("❌ Database cleanup failed")
                        return False
                else:
                    logger.info("❌ Cleanup cancelled")
                    return False
                    
            elif choice == "3":
                logger.info("🔍 Checking chunks status only...")
                if chunks_status:
                    logger.info("✅ Chunks status check completed")
                    return True
                else:
                    logger.error("❌ Chunks status check failed")
                    return False
                    
            elif choice == "4":
                logger.info("❌ Exiting without changes")
                return False
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
    else:
        logger.info("✅ Database is healthy!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
