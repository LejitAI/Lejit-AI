import chromadb
from chromadb.config import Settings
import json
from pathlib import Path
import logging

# Correct logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Create a logger instance instead of using the function directly

def debug_chromadb_collection():
    try:
        # 1. Initialize client with explicit settings
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # 2. Get collection and print basic info
        collection = client.get_collection("document_collection")
        count = collection.count()
        logger.info(f"Collection count: {count}")
        
        # 3. Get all items from collection
        all_results = collection.get()
        logger.info(f"Number of items in collection: {len(all_results['ids'])}")
        
        # 4. Print first few documents if they exist
        if len(all_results['documents']) > 0:
            logger.info("First document sample:")
            logger.info(all_results['documents'][0][:500] + "...")  # First 500 chars
        
        # 5. Try different search approaches
        test_queries = [
            "WHAT RESOURCES TO REFER TO DURING THE PROCESS?",
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            
            # Try with different parameters
            results = collection.query(
                query_texts=[query],
                n_results=5,
                include=['documents', 'distances', 'metadatas']
            )
            
            logger.info(f"Results found: {len(results['documents'][0])}")
            if results['documents'][0]:
                logger.info(f"First result similarity score: {results['distances'][0][0]}")
                logger.info(f"First result preview: {results['documents'][0][0][:1000]}...")
        
        # 6. Check processed files
        with open('C:\\Users\\daina\\Desktop\\lejit\\chroma_db\\processed_files.json', 'r') as f:
            processed_files = json.load(f)
        
        file_info = processed_files.get("C:\\Users\\daina\\Desktop\\lejit\\data_induja\\Style guides for legal writing\\Legal-Research-and-Writing-Guide-1.pdf-1.pdf")
        if file_info:
            logger.info(f"\nProcessed file info:")
            logger.info(f"Number of chunks: {file_info['num_chunks']}")
            logger.info(f"Processing status: {file_info['status']}")
            logger.info(f"Processing date: {file_info['processed_date']}")
        
        return {
            "collection_count": count,
            "items_in_collection": len(all_results['ids']),
            "has_documents": len(all_results['documents']) > 0,
            "processed_chunks": file_info['num_chunks'] if file_info else None
        }

    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        raise

def suggest_fixes(debug_results):
    suggestions = []
    
    if debug_results["collection_count"] == 0:
        suggestions.append("Collection is empty. Need to reload documents.")
    
    if debug_results["collection_count"] != debug_results["items_in_collection"]:
        suggestions.append("Mismatch between collection count and actual items. Consider rebuilding the collection.")
    
    if not debug_results["has_documents"]:
        suggestions.append("No documents in collection. Check document processing pipeline.")
    
    if debug_results["processed_chunks"] and debug_results["collection_count"] != debug_results["processed_chunks"]:
        suggestions.append("Number of chunks in collection doesn't match processed files. Reload documents.")
    
    return suggestions

# Usage example:
if __name__ == "__main__":
    try:
        debug_results = debug_chromadb_collection()
        fixes = suggest_fixes(debug_results)
        
        logger.info("\nSuggested fixes:")
        for fix in fixes:
            logger.info(f"- {fix}")
    except Exception as e:
        logger.error(f"Script failed: {e}")