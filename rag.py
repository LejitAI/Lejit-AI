# rag.py

from chunk_vector_store import LegalChunkVectorStore
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

class LegalRag:
    def __init__(self) -> None:
        self.cvs_obj = LegalChunkVectorStore()
        self.vector_store = None
        self.retriever = None
        self.pdf_chain = None
        self.combined_chain = None
        self.persist_directory = "./chroma_db"
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Enhanced legal-focused prompt template
        self.prompt_template = """
        <s> [INST] You are an AI legal assistant with expertise in analyzing legal documents, contracts, and providing legal information. Your role is to:
        
        1. Provide clear, accurate legal information while noting that you cannot provide legal advice
        2. Help understand legal documents and identify key provisions
        3. Explain legal concepts in plain language
        4. Flag potential legal issues that may require attorney review
        5. Reference relevant legal frameworks when applicable
        
        Important Guidelines:
        - Always include a disclaimer that this is for informational purposes only and not legal advice
        - Maintain attorney-client privilege by keeping all discussions confidential
        - Be precise in legal terminology while explaining concepts clearly
        - Highlight any limitations in your analysis
        - Recommend consulting with a qualified attorney for specific legal advice
        
        {mode_specific_context}
        
        User Query: {question}
        
        Please provide a {response_type} response that adheres to these guidelines.
        [/INST]
        """
        
        # Enhanced mode-specific contexts
        self.mode_contexts = {
            "text_only": """
            Respond based on general legal knowledge and principles.
            Focus on providing educational information about legal concepts and frameworks.
            """,
            
            "pdf_only": """
            Context from legal documents:
            {context}
            
            Analyze the provided legal document context while:
            - Identifying key legal provisions and terms
            - Explaining complex legal language in plain terms
            - Highlighting important dates, obligations, and requirements
            - Noting any potential areas that may need attorney review
            """,
            
            "combined": """
            Context from legal documents:
            {context}
            
            Consider both the document context and general legal principles while:
            - Connecting document-specific details with broader legal frameworks
            - Providing relevant legal context and background
            - Explaining how general legal principles apply to the specific document
            - Identifying areas where additional legal research may be needed
            """
        }
        
        # Enhanced response types
        self.response_types = {
            "text_only": "clear, educational legal information",
            "pdf_only": "detailed legal document analysis",
            "combined": "comprehensive legal analysis with context"
        }
        
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.model = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-70b-8192"
        )
        
        # Initialize chains
        self._initialize_chains()

    # Rest of the methods remain the same as in the original code
    def _initialize_chains(self):
        """Initialize all chat chains"""
        # Text-only chain
        text_prompt = PromptTemplate.from_template(
            self.prompt_template.format(
                mode_specific_context=self.mode_contexts["text_only"],
                response_type=self.response_types["text_only"],
                question="{question}"
            )
        )
        self.text_chain = (
            {"question": RunnablePassthrough()}
            | text_prompt 
            | self.model
            | StrOutputParser()
        )

    def _setup_document_chains(self):
        """Setup PDF and combined chains after document processing"""
        if not self.retriever:
            return
        
        # PDF-only chain
        pdf_prompt = PromptTemplate.from_template(
            self.prompt_template.format(
                mode_specific_context=self.mode_contexts["pdf_only"],
                response_type=self.response_types["pdf_only"],
                question="{question}"
            )
        )
        self.pdf_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | pdf_prompt 
            | self.model
            | StrOutputParser()
        )
        
        # Combined chain
        combined_prompt = PromptTemplate.from_template(
            self.prompt_template.format(
                mode_specific_context=self.mode_contexts["combined"],
                response_type=self.response_types["combined"],
                question="{question}"
            )
        )
        self.combined_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | combined_prompt 
            | self.model
            | StrOutputParser()
        )

    def set_retriever(self):
        """Set up the document retriever"""
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5,
                }
            )
            self._setup_document_chains()

    def ask_text_only(self, query: str) -> str:
        """Handle text-only queries"""
        try:
            return self.text_chain.invoke(query)
        except Exception as e:
            return f"Error processing text-only query: {str(e)}"

    def ask_pdf(self, query: str) -> str:
        """Handle PDF-only queries"""
        if not self.pdf_chain:
            return "Please upload a legal document first."
        try:
            return self.pdf_chain.invoke(query)
        except Exception as e:
            return f"Error processing PDF query: {str(e)}"

    def ask_combined(self, query: str) -> str:
        """Handle combined queries"""
        if not self.combined_chain:
            return self.ask_text_only(query)
        try:
            return self.combined_chain.invoke(query)
        except Exception as e:
            return f"Error processing combined query: {str(e)}"

    def feed(self, file_path: str, document_type: str = "Other"):
        """Process and store document content"""
        try:
            # Create or get collection
            try:
                self.chroma_client.delete_collection("document_collection")
            except ValueError:
                pass
            
            collection = self.chroma_client.create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Process document with document type
            chunks = self.cvs_obj.split_into_chunks(file_path, document_type)
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name="document_collection",
                embedding_function=self.cvs_obj.embedding
            )
            
            self.vector_store.add_documents(chunks)
            self.set_retriever()
            
        except Exception as e:
            raise Exception(f"Error feeding document: {str(e)}")

    def clear(self):
        """Clear all stored documents and reset chains"""
        try:
            if self.chroma_client is not None:
                try:
                    self.chroma_client.delete_collection("document_collection")
                except ValueError:
                    pass
            
            self.vector_store = None
            self.retriever = None
            self.pdf_chain = None
            self.combined_chain = None
            
        except Exception as e:
            print(f"Error clearing resources: {str(e)}")