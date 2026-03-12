#!/usr/bin/env python3
"""
Script para corrigir especificamente a tabela agents.tools sem perder dados de outras tabelas.
Este script preserva todas as outras tabelas e views, recriando apenas a tabela tools.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_existing_data():
    """Backup existing data from other tables before fixing tools table."""
    try:
        import pixeltable as pxt
        
        logger.info("📦 Creating backup of existing data...")
        
        # Tables to backup (excluding tools)
        tables_to_backup = [
            "agents.collection",
            "agents.images", 
            "agents.videos",
            "agents.audios",
            "agents.tabular",
            "agents.memory_bank",
            "agents.chat_history",
            "agents.user_personas",
            "agents.image_generation_tasks"
        ]
        
        backup_data = {}
        
        for table_name in tables_to_backup:
            try:
                table = pxt.get_table(table_name)
                if table is not None:
                    # Get all data from the table
                    data = table.select().collect()
                    backup_data[table_name] = data
                    logger.info(f"✅ Backed up {table_name}: {len(data)} records")
                else:
                    logger.warning(f"⚠️  Table {table_name} is None, skipping backup")
            except Exception as e:
                logger.warning(f"⚠️  Could not backup {table_name}: {str(e)}")
        
        return backup_data
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {str(e)}")
        return None

def restore_backup_data(backup_data):
    """Restore data from backup to the tables."""
    try:
        import pixeltable as pxt
        
        logger.info("🔄 Restoring backup data...")
        
        for table_name, data in backup_data.items():
            if data and len(data) > 0:
                try:
                    table = pxt.get_table(table_name)
                    if table is not None:
                        # Insert the backed up data
                        table.insert(data)
                        logger.info(f"✅ Restored {len(data)} records to {table_name}")
                    else:
                        logger.warning(f"⚠️  Table {table_name} is None, cannot restore")
                except Exception as e:
                    logger.warning(f"⚠️  Could not restore {table_name}: {str(e)}")
        
        logger.info("✅ Backup restoration completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backup restoration failed: {str(e)}")
        return False

def recreate_tools_table():
    """Recreate only the tools table with all its computed columns."""
    try:
        import pixeltable as pxt
        from pixeltable.functions.anthropic import invoke_tools, messages
        from pixeltable.functions.huggingface import sentence_transformer, clip
        from pixeltable.functions import openai
        from pixeltable.functions.mistralai import chat_completions as mistral
        from pixeltable.functions import string as pxt_str
        import config
        import functions
        
        logger.info("🔧 Recreating tools table...")
        
        # First, try to drop the problematic tools table
        try:
            pxt.drop_table("agents.tools")
            logger.info("✅ Dropped existing tools table")
        except Exception as e:
            logger.warning(f"Could not drop existing tools table: {e}")
        
        # Register tools - use the query functions from setup_pixeltable.py
        # We need to import the functions that were already defined
        from setup_pixeltable import search_video_transcripts, search_audio_transcripts
        
        tools = pxt.tools(
            # Query Functions registered as Tools - Agentic RAG
            search_video_transcripts,
            search_audio_transcripts
        )
        
        # Create the tools table
        tool_agent = pxt.create_table(
            "agents.tools",
            {
                # Input fields from the user query
                "prompt": pxt.String,
                "timestamp": pxt.Timestamp,
                "user_id": pxt.String,
                "initial_system_prompt": pxt.String,
                "final_system_prompt": pxt.String,
                # LLM parameters
                "max_tokens": pxt.Int,
                "stop_sequences": pxt.Json,
                "temperature": pxt.Float,
                "top_p": pxt.Float,
                "top_k": pxt.Int,  # Add top_k parameter to schema
            },
            if_exists="replace",
        )
        
        logger.info("✅ Created tools table")
        
        # Add computed columns step by step
        logger.info("🔧 Adding computed columns...")
        
        # Step 1: Initial LLM Reasoning
        tool_agent.add_computed_column(
            initial_response=messages(
                model=config.CLAUDE_MODEL_ID,
                system=tool_agent.initial_system_prompt,
                messages=[{"role": "user", "content": tool_agent.prompt}],
                tools=tools,
                tool_choice=tools.choice(required=True),
                max_tokens=tool_agent.max_tokens,
                stop_sequences=tool_agent.stop_sequences,
                temperature=tool_agent.temperature,
                top_p=tool_agent.top_p,
                top_k=tool_agent.top_k,  # Pass top_k parameter
            ),
            if_exists="replace",
        )
        logger.info("✅ Added initial_response column")
        
        # Step 2: Tool Execution
        tool_agent.add_computed_column(
            tool_output=invoke_tools(tools, tool_agent.initial_response),
            if_exists="replace"
        )
        logger.info("✅ Added tool_output column")
        
        # Step 3: Context Retrieval
        # Get references to other tables for context
        documents = pxt.get_table("agents.collection")
        images = pxt.get_table("agents.images")
        videos = pxt.get_table("agents.videos")
        chunks = pxt.get_table("agents.chunks")
        video_frames = pxt.get_table("agents.video_frames")
        memory_bank = pxt.get_table("agents.memory_bank")
        chat_history = pxt.get_table("agents.chat_history")
        
        # Document context - use the search function from setup_pixeltable
        from setup_pixeltable import search_documents
        tool_agent.add_computed_column(
            doc_context=search_documents(tool_agent.prompt, tool_agent.user_id),
            if_exists="replace",
        )
        logger.info("✅ Added doc_context column")
        
        # Image context - use the search function from setup_pixeltable
        from setup_pixeltable import search_images
        tool_agent.add_computed_column(
            image_context=search_images(tool_agent.prompt, tool_agent.user_id),
            if_exists="replace"
        )
        logger.info("✅ Added image_context column")
        
        # Video frame context - use the search function from setup_pixeltable
        from setup_pixeltable import search_video_frames
        tool_agent.add_computed_column(
            video_frame_context=search_video_frames(tool_agent.prompt, tool_agent.user_id),
            if_exists="replace"
        )
        logger.info("✅ Added video_frame_context column")
        
        # Memory context - use the search function from setup_pixeltable
        from setup_pixeltable import search_memory
        tool_agent.add_computed_column(
            memory_context=search_memory(tool_agent.prompt, tool_agent.user_id),
            if_exists="replace"
        )
        logger.info("✅ Added memory_context column")
        
        # Chat memory context - use the search function from setup_pixeltable
        from setup_pixeltable import search_chat_history
        tool_agent.add_computed_column(
            chat_memory_context=search_chat_history(tool_agent.prompt, tool_agent.user_id),
            if_exists="replace"
        )
        logger.info("✅ Added chat_memory_context column")
        
        # History context - use the search function from setup_pixeltable
        from setup_pixeltable import get_recent_chat_history
        tool_agent.add_computed_column(
            history_context=get_recent_chat_history(tool_agent.user_id),
            if_exists="replace"
        )
        logger.info("✅ Added history_context column")
        
        # Step 4: Multimodal context summary
        tool_agent.add_computed_column(
            multimodal_context_summary=functions.assemble_multimodal_context(
                tool_agent.prompt,
                tool_agent.tool_output,
                tool_agent.doc_context,
                tool_agent.memory_context,
                tool_agent.chat_memory_context,
            ),
            if_exists="replace",
        )
        logger.info("✅ Added multimodal_context_summary column")
        
        # Step 5: Final prompt messages
        tool_agent.add_computed_column(
            final_prompt_messages=functions.assemble_final_messages(
                tool_agent.history_context,
                tool_agent.multimodal_context_summary,
                image_context=tool_agent.image_context,
                video_frame_context=tool_agent.video_frame_context,
            ),
            if_exists="replace",
        )
        logger.info("✅ Added final_prompt_messages column")
        
        # Step 6: Final response
        tool_agent.add_computed_column(
            final_response=messages(
                model=config.CLAUDE_MODEL_ID,
                system=tool_agent.final_system_prompt,
                messages=tool_agent.final_prompt_messages,
                max_tokens=tool_agent.max_tokens,
                stop_sequences=tool_agent.stop_sequences,
                temperature=tool_agent.temperature,
                top_p=tool_agent.top_p,
            ),
            if_exists="replace",
        )
        logger.info("✅ Added final_response column")
        
        # Step 7: Extract answer
        tool_agent.add_computed_column(
            answer=tool_agent.final_response.content[0].text,
            if_exists="replace",
        )
        logger.info("✅ Added answer column")
        
        # Step 8: Follow-up input
        tool_agent.add_computed_column(
            follow_up_input_message=functions.assemble_follow_up_prompt(
                original_prompt=tool_agent.prompt,
                answer_text=tool_agent.answer
            ),
            if_exists="replace",
        )
        logger.info("✅ Added follow_up_input_message column")
        
        # Step 9: Follow-up response
        tool_agent.add_computed_column(
            follow_up_raw_response=mistral(
                model=config.MISTRAL_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": tool_agent.follow_up_input_message,
                    }
                ],
                max_tokens=150,
                temperature=0.6,
            ),
            if_exists="replace",
        )
        logger.info("✅ Added follow_up_raw_response column")
        
        # Step 10: Extract follow-up text
        tool_agent.add_computed_column(
            follow_up_text=tool_agent.follow_up_raw_response.choices[0].message.content,
            if_exists="replace",
        )
        logger.info("✅ Added follow_up_text column")
        
        logger.info("✅ Tools table recreation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recreating tools table: {str(e)}")
        return False

def test_tools_table():
    """Test if the tools table is working correctly."""
    try:
        import pixeltable as pxt
        
        logger.info("🧪 Testing tools table...")
        
        # Try to access the table
        tool_agent = pxt.get_table("agents.tools")
        if tool_agent is None:
            logger.error("❌ Tools table is None")
            return False
        
        # Try to insert a test record
        test_data = {
            "prompt": "Test query",
            "timestamp": datetime.now(),
            "user_id": "test_user",
            "initial_system_prompt": "You are a helpful assistant.",
            "final_system_prompt": "You are a helpful assistant.",
            "max_tokens": 1000,
            "stop_sequences": [],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        tool_agent.insert([test_data])
        logger.info("✅ Successfully inserted test record")
        
        # Try to query the table
        result = tool_agent.select().limit(1).collect()
        if result:
            logger.info("✅ Successfully queried tools table")
            return True
        else:
            logger.warning("⚠️  Tools table query returned no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Tools table test failed: {str(e)}")
        return False

def main():
    """Main function to fix the tools table."""
    logger.info("🚀 Starting tools table repair...")
    
    # Step 1: Backup existing data
    backup_data = backup_existing_data()
    if backup_data is None:
        logger.error("❌ Backup failed, aborting repair")
        return False
    
    # Step 2: Recreate tools table
    if not recreate_tools_table():
        logger.error("❌ Tools table recreation failed")
        return False
    
    # Step 3: Test the new table
    if not test_tools_table():
        logger.error("❌ Tools table test failed")
        return False
    
    # Step 4: Restore backup data
    if not restore_backup_data(backup_data):
        logger.warning("⚠️  Backup restoration failed, but tools table is fixed")
    
    logger.info("✅ Tools table repair completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Tools table is now working correctly!")
    else:
        logger.error("💥 Tools table repair failed!")
    
    sys.exit(0 if success else 1)
