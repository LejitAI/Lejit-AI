# rag.py

from chunk_vector_store import LegalChunkVectorStore
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRag:
    def __init__(self, memory_key: str = "chat_history") -> None:
        """
        Initialize the Legal RAG system with memory management and error handling.
        
        Args:
            memory_key (str): Key used for storing chat history in memory
        """
        self.cvs_obj = LegalChunkVectorStore()
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.pdf_chain = None
        self.combined_chain = None
        self.persist_directory = "./chroma_db"
        self.memory_key = memory_key
        
        # Initialize memory with StreamlitChatMessageHistory for persistence
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            chat_memory=StreamlitChatMessageHistory()
        )
        
        # Initialize ChromaDB client with error handling
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # System messages for different contexts
        self.system_message = """You are an expert legal AI assistant trained in legal analysis, research, and documentation. Your purpose is to help understand and analyze legal documents while maintaining strict professional standards. You will:

            1. Provide factual legal information while explicitly stating this is not legal advice
            2. Analyze documents and identify key legal elements including:
            - Binding provisions and obligations
            - Definitions and interpretations
            - Rights and responsibilities
            - Timelines and deadlines
            - Jurisdiction and governing law
            - Risk factors and potential issues
            3. Break down complex legal language into clear, understandable terms
            4. Reference specific statutes, regulations, or case law when relevant
            5. Identify potential compliance issues or areas requiring attorney review
            6. Maintain proper legal terminology while explaining concepts
            7. Consider jurisdictional differences when applicable
            8. Review document structure and formatting for legal sufficiency

            Critical Guidelines:
            - Begin responses with: "Important: This analysis is for informational purposes only and does not constitute legal advice. Please consult with a qualified attorney for specific legal guidance."
            - Maintain strict confidentiality and attorney-client privilege
            - Cite specific sections/clauses when referencing documents
            - Highlight ambiguities or potential conflicts in language
            - Flag high-risk provisions or terms requiring special attention
            - Indicate when additional legal research may be necessary
            - Note jurisdictional limitations or requirements
            - Reference industry standards and best practices when applicable
            - Maintain professional tone and legal accuracy throughout

            {mode_specific_context}
            """
        
        # Enhanced mode-specific contexts for different types of analysis
        self.mode_contexts = {
            "text_only": """
            Context Type: General Legal Information Analysis

            Primary Focus:
            - Explain fundamental legal concepts and principles
            - Provide context for common legal terminology
            - Discuss general legal frameworks and requirements
            - Reference relevant statutes and regulations
            - Explain standard legal procedures and processes
            - Outline typical legal rights and obligations

            Analysis Guidelines:
            1. Start with broad legal principles before specific details
            2. Reference authoritative legal sources when applicable
            3. Explain legal terminology in plain language
            4. Discuss common interpretations and applications
            5. Highlight jurisdiction-specific considerations
            6. Note relevant exceptions or special cases
            7. Consider industry-standard practices
            8. Reference historical legal precedents if relevant

            Response Format:
            1. Legal Framework: Outline applicable laws/regulations
            2. Key Principles: Explain fundamental concepts
            3. Practical Application: Discuss real-world context
            4. Limitations: Note jurisdictional or scope restrictions
            5. Additional Considerations: Flag related legal issues
            """,
            
            "pdf_only": """
            Context Type: Legal Document Analysis

            Document Context:
            {context}

            Analysis Framework:
            1. Document Structure
               - Type and purpose of document
               - Formatting and organization
               - Completeness and coherence
            
            2. Key Provisions Analysis
               - Definitions and interpretations
               - Rights and obligations
               - Conditions and warranties
               - Term and termination
               - Governing law and jurisdiction
               - Remedies and enforcement
            
            3. Risk Assessment
               - Ambiguous language
               - Missing provisions
               - Conflicting terms
               - Compliance issues
               - Enforceability concerns
            
            4. Contextual Elements
               - Industry-specific considerations
               - Regulatory requirements
               - Market standard terms
               - Jurisdictional impact
            
            Document Review Guidelines:
            - Reference specific sections and clauses
            - Compare against legal standards
            - Identify unusual or non-standard provisions
            - Flag potential drafting issues
            - Note missing standard provisions
            - Assess overall legal effectiveness
            """,
            
            "combined": """
            Context Type: Comprehensive Legal Analysis

            Document Context:
            {context}

            Integrated Analysis Framework:
            1. Document-Specific Analysis
               - Detailed review of provided content
               - Identification of key provisions
               - Risk assessment and flagging
               - Structural analysis
            
            2. Legal Framework Integration
               - Applicable laws and regulations
               - Relevant case law
               - Industry standards
               - Best practices
            
            3. Contextual Considerations
               - Jurisdiction-specific requirements
               - Industry-specific regulations
               - Market standard comparisons
               - Historical precedents
            
            4. Risk and Compliance
               - Regulatory compliance assessment
               - Enforceability analysis
               - Risk mitigation strategies
               - Best practice recommendations
            
            Response Structure:
            1. Document Analysis: Specific content review
            2. Legal Framework: Applicable laws and standards
            3. Integrated Assessment: Combined analysis
            4. Recommendations: Areas for review/improvement
            5. Risk Factors: Identified concerns
            
            Special Instructions:
            - Cross-reference document provisions with legal requirements
            - Provide integrated analysis of specific and general principles
            - Highlight interactions between document terms and legal framework
            - Identify gaps between document content and legal requirements
            - Suggest improvements based on legal standards
            """
        }
        
        # Initialize LLM
        try:
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            
            self.model = ChatGroq(
                api_key=groq_api_key,
                model_name="llama3-70b-8192"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
        
        # Initialize chains
        self._initialize_chains()

    def _initialize_chains(self) -> None:
        """Initialize all chat chains with memory management"""
        try:
            # Text-only chain
            text_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message.format(
                    mode_specific_context=self.mode_contexts["text_only"]
                )),
                MessagesPlaceholder(variable_name=self.memory_key),
                ("human", "{question}")
            ])
            
            self.text_chain = (
                {
                    self.memory_key: lambda x: self.memory.load_memory_variables({})[self.memory_key],
                    "question": RunnablePassthrough()
                }
                | text_prompt 
                | self.model
                | StrOutputParser()
            )
            
            logger.info("Successfully initialized text chain")
        except Exception as e:
            logger.error(f"Failed to initialize text chain: {e}")
            raise

    def _setup_document_chains(self) -> None:
        """Setup PDF and combined chains with memory management"""
        if not self.retriever:
            logger.warning("Retriever not initialized. Skipping document chain setup.")
            return
        
        try:
            # PDF-only chain
            pdf_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message.format(
                    mode_specific_context=self.mode_contexts["pdf_only"]
                )),
                MessagesPlaceholder(variable_name=self.memory_key),
                ("human", "{question}")
            ])
            
            self.pdf_chain = (
                {
                    "context": self.retriever,
                    self.memory_key: lambda x: self.memory.load_memory_variables({})[self.memory_key],
                    "question": RunnablePassthrough()
                }
                | pdf_prompt 
                | self.model
                | StrOutputParser()
            )
            
            # Combined chain
            combined_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message.format(
                    mode_specific_context=self.mode_contexts["combined"]
                )),
                MessagesPlaceholder(variable_name=self.memory_key),
                ("human", "{question}")
            ])
            
            self.combined_chain = (
                {
                    "context": self.retriever,
                    self.memory_key: lambda x: self.memory.load_memory_variables({})[self.memory_key],
                    "question": RunnablePassthrough()
                }
                | combined_prompt 
                | self.model
                | StrOutputParser()
            )
            
            logger.info("Successfully initialized document chains")
        except Exception as e:
            logger.error(f"Failed to initialize document chains: {e}")
            raise

    def _update_memory(self, query: str, response: str) -> None:
        """
        Update conversation memory with new interaction
        
        Args:
            query (str): User's question
            response (str): Assistant's response
        """
        try:
            self.memory.chat_memory.add_message(HumanMessage(content=query))
            self.memory.chat_memory.add_message(AIMessage(content=response))
            logger.debug(f"Memory updated with new interaction")
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the chat history
        
        Returns:
            List[Dict[str, Any]]: List of chat messages
        """
        try:
            messages = self.memory.chat_memory.messages
            return [{"role": "human" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content} for msg in messages]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []

    def set_retriever(self) -> None:
        """Set up the document retriever with similarity search"""
        if self.vector_store:
            try:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": 3,
                        "score_threshold": 0.5,
                    }
                )
                self._setup_document_chains()
                logger.info("Successfully set up retriever")
            except Exception as e:
                logger.error(f"Failed to set up retriever: {e}")
                raise

    def ask_text_only(self, query: str) -> str:
        """
        Handle text-only queries with memory management
        
        Args:
            query (str): User's question
            
        Returns:
            str: Assistant's response
        """
        try:
            response = self.text_chain.invoke(query)
            self._update_memory(query, response)
            return response
        except Exception as e:
            error_msg = f"Error processing text-only query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_pdf(self, query: str) -> str:
        """
        Handle PDF-only queries with memory management
        
        Args:
            query (str): User's question
            
        Returns:
            str: Assistant's response
        """
        if not self.pdf_chain:
            return "Please upload a legal document first."
        try:
            response = self.pdf_chain.invoke(query)
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
            response = self.combined_chain.invoke(query)
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
            
            logger.info(f"Successfully processed {document_type} document")
            
        except Exception as e:
            error_msg = f"Error feeding document: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def clear(self) -> None:
        """Clear all stored documents, reset chains, and clear memory"""
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
            self.memory.clear()
            
            logger.info("Successfully cleared all resources")
            
        except Exception as e:
            error_msg = f"Error clearing resources: {str(e)}"
            logger.error(error_msg)
            print(error_msg)