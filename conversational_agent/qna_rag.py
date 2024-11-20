from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import tempfile
from langchain.schema import HumanMessage, AIMessage
from chunk_vector_store import LegalChunkVectorStore
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="API for legal document analysis and general legal queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: str

class DocumentRequest(BaseModel):
    document_type: str = "Other"
    session_id: str

class QueryResponse(BaseModel):
    response: str
    source_documents: Optional[List[Dict[str, Any]]] = None

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]

# Global storage for RAG instances
rag_instances: Dict[str, 'LegalRAG'] = {}

class LegalRAG:
    def __init__(self) -> None:
        self.cvs_obj = LegalChunkVectorStore()
        self.vector_store = None
        self.retriever = None
        self.pdf_chain = None
        self.combined_chain = None
        self.persist_directory = tempfile.mkdtemp()  # Create temporary directory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            
            self.model = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192",
                temperature=0.1
            )
            
            self._initialize_chains()
            logger.info("Successfully initialized LegalRAG instance")
        except Exception as e:
            logger.error(f"Failed to initialize LegalRAG: {e}")
            raise

    # Include all the methods from the original LegalRAG class
    # [Previous methods remain the same - _format_chat_history, _initialize_chains, etc.]

    def _format_chat_history(self, chat_history) -> str:
        formatted_history = []
        for message in chat_history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        return "\n".join(formatted_history[-3:])


    def _initialize_chains(self) -> None:
        try:
            # Modified text prompt for Indian legal context
            text_prompt = PromptTemplate.from_template(
                """You are an experienced Indian legal assistant providing accurate information based on Indian law while maintaining that this is not legal advice. You must reference Indian legal statutes, acts, and precedents where applicable.

                Document Context:
                {context}

                Chat History:
                {chat_history}

                Human Question: {question}

                Please provide a clear, professional response that:
                1. Primarily uses information from the retrieved document context when available
                2. References relevant Indian legal concepts, statutes, and case laws
                3. Clearly distinguishes between document-specific information and general Indian legal knowledge
                4. Explains legal terms in simple language while retaining Indian legal terminology
                5. Mentions relevant High Court or Supreme Court judgments if applicable
                6. Maintains appropriate legal disclaimers as per Indian Bar Council guidelines
                7. Suggests when consultation with an Indian advocate may be needed
                8. Specifies if any information pertains to specific Indian states, as laws may vary by jurisdiction

                Assistant Response:"""
            )
            
            def format_context(context_docs):
                formatted_contexts = []
                for i, doc in enumerate(context_docs, 1):
                    metadata = doc.metadata or {}
                    source_info = f"[Source: Page {metadata.get('page', 'N/A')}]"
                    formatted_contexts.append(f"Context {i} {source_info}:\n{doc.page_content}")
                
                return "\n\n".join(formatted_contexts)
            
            # Rest of the initialization code remains same
            self.text_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])) if self.retriever else "No document context available.",
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | text_prompt 
                | self.model
                | StrOutputParser()
            )
            
            logger.info("Successfully initialized Indian legal assistant text chain with RAG capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize text chain: {e}")
            raise

    def _setup_document_chains(self) -> None:
        if not self.retriever:
            logger.warning("Retriever not initialized. Skipping document chain setup.")
            return

        try:
            def format_context(context_docs):
                formatted_contexts = []
                for i, doc in enumerate(context_docs, 1):
                    metadata = doc.metadata or {}
                    source_info = f"[Source: Page {metadata.get('page', 'N/A')}]"
                    formatted_contexts.append(f"Context {i} {source_info}:\n{doc.page_content}")
                
                return "\n\n".join(formatted_contexts)

            # Modified PDF prompt for Indian legal context
            pdf_prompt = PromptTemplate.from_template(
            """You are an expert Indian legal document analyst specializing in court orders, contracts, compliance documents, government notifications, and statutory documents. Analyze the provided document context based on Indian legal framework.

            Document Context:
            {context}

            Chat History:
            {chat_history}

            Human Question: {question}

            Provide a detailed analysis following these guidelines:

            1. Document Classification and Initial Analysis:
               - Identify document type (Court Order/Contract/Compliance/Government/Case-related/Statutory)
               - Specify jurisdiction (Supreme Court/High Court/State/Central)
               - Identify parties involved and their roles
               - Note critical dates, deadlines, and timelines

            2. Detailed Content Analysis:
               For Court Orders/Judgments:
               - Extract key holdings and precedents
               - Identify cited cases and statutes
               - Highlight ratio decidendi
               - Note important observations and directions

               For Contracts/Agreements:
               - Identify key clauses and obligations
               - Highlight rights and liabilities
               - Note important terms, conditions, and deadlines
               - Flag potential legal issues

               For Compliance Documents:
               - List statutory requirements
               - Identify compliance deadlines
               - Note filing obligations
               - Highlight regulatory requirements

               For Government Documents:
               - Explain implementation requirements
               - Note procedural compliances
               - Identify affected parties
               - List required actions

               For Case-Related Documents:
               - Analyze procedural requirements
               - Identify next steps
               - Note response deadlines
               - List required documentation

               For Statutory Documents:
               - Explain applicability
               - List compliance requirements
               - Note relevant authorities
               - Highlight penalty provisions

            3. Response Format:
               - Quote relevant sections using "..."
               - Cite specific document parts with source information
               - Use clear headings for different aspects
               - Provide practical implications
               - Highlight immediate action items
               - List relevant compliance requirements
               - Include statutory references
               - Note jurisdiction-specific requirements

            4. Additional Guidelines:
               - Stay strictly within document context
               - Use appropriate Indian legal terminology
               - Explain technical terms in simple language
               - Maintain legal disclaimers as per Bar Council
               - Indicate if any information is outside document scope
               - Suggest professional consultation when needed
               - Note if multiple interpretations are possible
               - Highlight any ambiguities requiring clarification

            Assistant Response (structure your response based on the document type and query focus):"""
            )
            
            # Enhanced combined prompt for integrated analysis
            combined_prompt = PromptTemplate.from_template(
                """You are an expert Indian legal assistant providing comprehensive document analysis with additional legal context. 

                Document Context:
                {context}

                Chat History:
                {chat_history}

                Human Question: {question}

                Provide a detailed response that:

                1. Document-Based Analysis:
                - Analyze document content thoroughly
                - Quote relevant sections using "..."
                - Identify key provisions and requirements
                - Note critical dates and deadlines
                - Highlight important clauses and conditions

                2. Legal Framework Integration:
                - Connect document content with relevant:
                    * Indian statutes and regulations
                    * Supreme Court/High Court judgments
                    * Government notifications/circulars
                    * Regulatory requirements
                - Explain legal implications
                - Note jurisdiction-specific aspects
                - Highlight compliance requirements

                3. Practical Guidance:
                - Provide actionable insights
                - List required next steps
                - Explain procedural requirements
                - Note documentation needs
                - Highlight compliance deadlines
                - Suggest risk mitigation measures

                4. Response Requirements:
                - Distinguish between document information and general legal knowledge
                - Use appropriate Indian legal terminology
                - Explain complex terms simply
                - Maintain professional disclaimers
                - Indicate when professional help is needed
                - Note any ambiguities or uncertainties
                - Specify jurisdiction applicability

                5. Based on Document Type:
                Court Orders: Focus on holdings, precedents, directions
                Contracts: Emphasize rights, obligations, terms
                Compliance: Highlight requirements, deadlines, filings
                Government: Explain implementation, procedures
                Case Documents: Note procedures, timelines, requirements
                Statutory: Detail applicability, compliance needs

                Format your response based on document type and query focus, maintaining clarity and practical utility.

                Assistant Response:"""
            )
            
            # Chain setup
            self.pdf_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | pdf_prompt 
                | self.model
                | StrOutputParser()
            )
            
            self.combined_chain = (
                {
                    "context": lambda x: format_context(self.retriever.get_relevant_documents(x["question"])),
                    "chat_history": lambda x: self._format_chat_history(self.memory.chat_memory.messages),
                    "question": RunnablePassthrough()
                }
                | combined_prompt 
                | self.model
                | StrOutputParser()
            )
            
            logger.info("Successfully initialized Indian legal document chains with enhanced analysis capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize document chains: {e}")
            raise
        
    def _update_memory(self, query: str, response: str) -> None:
        try:
            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=response))
            logger.debug("Memory updated with new interaction")
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, Any]]:
        try:
            messages = self.memory.chat_memory.messages
            return [{"role": "human" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content} for msg in messages]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []

    def ask_text_only(self, query: str) -> str:
        try:
            # Check if documents are available in the vector store
            if self.retriever:
                # Get relevant documents
                context_docs = self.retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(context_docs)} contexts for text query: {[doc.page_content[:500] + '...' for doc in context_docs]}")
            
            # Process query with the text chain (which now includes document context if available)
            response = self.text_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing text-only query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_pdf(self, query: str) -> str:
        if not self.pdf_chain:
            return "Please upload a legal document first."
        try:
            context_docs = self.retriever.get_relevant_documents(query)
            if not context_docs:
                return "No relevant information found in the document. Please rephrase your question or ask about a different topic covered in the document."
            
            logger.debug(f"Retrieved contexts: {[doc.page_content[:500] + '...' for doc in context_docs]}")
            
            response = self.pdf_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response
        except Exception as e:
            error_msg = f"Error processing PDF query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_combined(self, query: str) -> str:
        """
        Handle combined queries with memory management

        Args:
            query (str): User's question
            
        Returns:
            str: Assistant's response
        """
        if not self.combined_chain:
            return self.ask_text_only(query)
        try:
            # Get relevant context to verify retrieval
            context_docs = self.retriever.get_relevant_documents(query)  # Removed fetch_k
            logger.debug(f"Retrieved contexts for combined query: {[doc.page_content[:500] + '...' for doc in context_docs]}")
            
            response = self.combined_chain.invoke({"question": query})
            self._update_memory(query, response)
            return response
        except Exception as e:
            error_msg = f"Error processing combined query: {str(e)}"
            logger.error(error_msg)
            return error_msg



    def feed(self, file_path: str, document_type: str = "Other") -> None:
        """
        Process and store document content with type-specific handling
        
        Args:
            file_path (str): Path to the document file
            document_type (str): Type of legal document
        """
        try:
            # Recreate collection
            try:
                self.chroma_client.delete_collection("document_collection")
                logger.info("Deleted existing document collection")
            except ValueError:
                pass
            
            collection = self.chroma_client.create_collection(
                name="document_collection",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new document collection")
            
            # Process document with document type
            chunks = self.cvs_obj.split_into_chunks(file_path, document_type)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Initialize vector store with proper embedding function
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name="document_collection",
                embedding_function=self.cvs_obj.embedding
            )
            
            # Add documents and setup retriever
            self.vector_store.add_documents(chunks)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Initialize document-aware chains
            self._setup_document_chains()
            logger.info(f"Successfully processed {document_type} document")
            
        except Exception as e:
            error_msg = f"Error feeding document: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def clear(self) -> None:
        """Clear all stored documents and reset system state"""
        try:
            if self.chroma_client is not None:
                try:
                    self.chroma_client.delete_collection("document_collection")
                    logger.info("Deleted document collection")
                except ValueError:
                    pass
            
            self.vector_store = None
            self.retriever = None
            self.pdf_chain = None
            self.combined_chain = None
            self.memory.clear()
            
            logger.info("Successfully cleared all resources")
            
        except Exception as e:
            error_msg = f"Error clearing resources: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            if self.persist_directory and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            logger.info("Successfully cleaned up resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# API endpoints
@app.post("/api/query/general", response_model=QueryResponse)
async def general_query(request: QueryRequest):
    """
    Handle general legal queries without document context
    """
    try:
        if request.session_id not in rag_instances:
            rag_instances[request.session_id] = LegalRAG()
        
        rag = rag_instances[request.session_id]
        # response = rag.ask_text_only(request.query)
        response = rag.text_chain.invoke({"question": request.query})
        rag._update_memory(request.query, response)
        
        return QueryResponse(
            response=response,
            source_documents=None
        )
    except Exception as e:
        logger.error(f"Error processing general query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Query("Other"),
    session_id: str = Query(...),
):
    """
    Upload and process a legal document
    """
    try:
        if session_id not in rag_instances:
            rag_instances[session_id] = LegalRAG()
        
        rag = rag_instances[session_id]
        
        # Save uploaded file temporarily
        temp_file_path = tempfile.mktemp(suffix='.pdf')
        with open(temp_file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        rag.feed(temp_file_path, document_type)
        
        # Clean up
        os.remove(temp_file_path)
        
        return {"message": "Document successfully processed"}
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/document", response_model=QueryResponse)
async def document_query(request: QueryRequest):
    """
    Query against uploaded document
    """
    try:
        if request.session_id not in rag_instances:
            raise HTTPException(status_code=400, detail="No document uploaded for this session")
        
        rag = rag_instances[request.session_id]
        response = rag.ask_pdf(request.query)
        
        # Get relevant documents if available
        source_docs = None
        if rag.retriever:
            docs = rag.retriever.get_relevant_documents(request.query)
            source_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        
        return QueryResponse(
            response=response,
            source_documents=source_docs
        )
    except Exception as e:
        logger.error(f"Error processing document query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    Get chat history for a session
    """
    try:
        if session_id not in rag_instances:
            return ChatHistoryResponse(history=[])
        
        rag = rag_instances[session_id]
        history = rag.get_chat_history()
        return ChatHistoryResponse(history=history)
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session and associated resources
    """
    try:
        if session_id in rag_instances:
            rag = rag_instances[session_id]
            rag.clear()
            rag.cleanup()
            del rag_instances[session_id]
        return {"message": "Session cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("qna_rag:app", host="0.0.0.0", port=6000, reload=True)