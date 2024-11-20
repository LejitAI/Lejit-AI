import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

@dataclass
class ProcessedFile:
    file_path: str
    hash: str
    processed_date: str
    num_chunks: int
    status: str = "success"
    error_message: str = ""

class PDFFeeder:
    def __init__(self):
        # Define paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_dir = os.path.join(self.base_dir, "Data_Devika")
        self.chroma_dir = os.path.join(self.base_dir, "chroma_db")
        self.processed_files_path = os.path.join(self.chroma_dir, "processed_files.json")
        
        # Initialize tracking
        self.processed_files = {}
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "skipped": 0
        }
        
        # Ensure directories exist
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Initialize Langchain's Chroma wrapper
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="document_collection",
            embedding_function=self.embedding_function
        )
        
        # Load processed files record
        self._load_processed_files()

    def _load_processed_files(self):
        """Load record of previously processed files"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                data = json.load(f)
                self.processed_files = {
                    k: ProcessedFile(**v) for k, v in data.items()
                }
            print(f"✅ Loaded {len(self.processed_files)} processed file records")
        else:
            print("📝 No existing processed files record found")

    def _save_processed_files(self):
        """Save record of processed files"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(
                {k: v.__dict__ for k, v in self.processed_files.items()},
                f,
                indent=2
            )
        print("💾 Saved processed files record")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash to detect changes"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _should_process_file(self, file_path: str, file_hash: str) -> bool:
        """Check if file needs processing"""
        if file_path not in self.processed_files:
            return True
        return self.processed_files[file_path].hash != file_hash

    def process_pdf(self, file_path: str) -> Optional[List[dict]]:
        """Process a single PDF file"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=20,
                length_function=len,
                add_start_index=True,
            )
            
            chunks = text_splitter.split_documents(documents)
            chunks = filter_complex_metadata(chunks)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata['source'] = file_path
                chunk.metadata['processed_date'] = datetime.now().isoformat()
            
            return chunks
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
            return None

    def find_pdf_files(self) -> List[str]:
        """Find all PDF files in the Data_Devika directory"""
        pdf_files = []
        for root, _, files in os.walk(self.pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def run(self):
        """Main execution function"""
        print("\n" + "="*50)
        print("🚀 Starting PDF Processing")
        print("="*50)
        print(f"📁 PDF Directory: {self.pdf_dir}")
        print(f"💾 ChromaDB Directory: {self.chroma_dir}")
        
        # Find PDF files
        pdf_files = self.find_pdf_files()
        if not pdf_files:
            print("❌ No PDF files found in Data_Devika directory!")
            return
        
        print(f"\n📚 Found {len(pdf_files)} PDF files")
        
        # Process files
        all_chunks = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                file_hash = self._calculate_file_hash(pdf_file)
                
                if not self._should_process_file(pdf_file, file_hash):
                    print(f"⏭️  Skipping {os.path.basename(pdf_file)} (unchanged)")
                    self.stats["skipped"] += 1
                    continue
                
                print(f"\n📄 Processing: {os.path.basename(pdf_file)}")
                chunks = self.process_pdf(pdf_file)
                
                if chunks:
                    # Add chunks to ChromaDB
                    print(f"💾 Adding {len(chunks)} chunks to ChromaDB...")
                    self.vector_store.add_documents(chunks)
                    
                    all_chunks.extend(chunks)
                    self.processed_files[pdf_file] = ProcessedFile(
                        file_path=pdf_file,
                        hash=file_hash,
                        processed_date=datetime.now().isoformat(),
                        num_chunks=len(chunks)
                    )
                    self.stats["files_processed"] += 1
                    self.stats["chunks_created"] += len(chunks)
                    print(f"✅ Created and stored {len(chunks)} chunks")
                else:
                    self.processed_files[pdf_file] = ProcessedFile(
                        file_path=pdf_file,
                        hash=file_hash,
                        processed_date=datetime.now().isoformat(),
                        num_chunks=0,
                        status="error",
                        error_message="Processing failed"
                    )
                    self.stats["errors"] += 1
            
            except Exception as e:
                print(f"❌ Error with {os.path.basename(pdf_file)}: {str(e)}")
                self.stats["errors"] += 1
        
        # Save progress
        self._save_processed_files()
        
        # Display summary
        print("\n" + "="*50)
        print("📊 Processing Summary")
        print("="*50)
        print(f"✅ Files Processed: {self.stats['files_processed']}")
        print(f"📄 Chunks Created: {self.stats['chunks_created']}")
        print(f"⏭️  Files Skipped: {self.stats['skipped']}")
        print(f"❌ Errors: {self.stats['errors']}")
        print("="*50)

if __name__ == "__main__":
    feeder = PDFFeeder()
    feeder.run()