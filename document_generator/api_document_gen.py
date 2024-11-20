from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, ClassVar
import os
import shutil
from datetime import datetime
import uuid
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame

# Initialize FastAPI app
app = FastAPI(title="Legal Document Generation API")

# Configuration
UPLOAD_DIR = "uploaded_documents"
DB_DIR = "vector_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Initialize models and clients
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
embedding_model.to(DEVICE)

# GroqLLM Class
class GroqLLM(LLM):
    models: ClassVar[List[str]] = [
        'llama3-70b-8192',
        'llama-3.1-70b-versatile',
        'mixtral-8x7b-32768',
        'llama3-8b-8192',
        'llama3-groq-70b-8192-tool-use-preview',
        'llama3-groq-8b-8192-tool-use-preview'
    ]
    
    model_name: str = models[3]
    client: Groq = Groq(api_key='your-key-here')
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "groq"

# Pydantic models
class Document(BaseModel):
    page_content: str
    metadata: Dict = {}

class DocumentMetadata(BaseModel):
    category: str
    document_type: str
    description: str

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    metadata: DocumentMetadata

class QueryRequest(BaseModel):
    query: str
    user_gender: str = "other"

class UserResponse(BaseModel):
    question: str
    answer: str

class DocumentGenerationRequest(BaseModel):
    document_id: str
    user_responses: List[UserResponse]

class SuggestionRequest(BaseModel):
    document_id: str
    suggestions: List[str]

# Utility functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file"""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)[:63]

def adapt_content_for_gender(content: str, user_gender: str) -> str:
    """Adapt document content based on user gender"""
    if user_gender.lower() == 'male':
        content = content.replace("[user]", "he").replace("[User]", "He")
    elif user_gender.lower() == 'female':
        content = content.replace("[user]", "she").replace("[User]", "She")
    else:
        content = content.replace("[user]", "they").replace("[User]", "They")
    return content

def generate_questions(content: str) -> List[str]:
    """Generate questions based on document content"""
    groq_llm = GroqLLM()
    prompt = f"""
    Based on the following document content, generate a list of questions to gather necessary information from the user.
    Each question should be specific and help in personalizing the document.
    
    Document content:
    {content}
    
    Generate questions:
    """
    
    response = groq_llm._call(prompt)
    return [q.strip() for q in response.split('\n') if q.strip()]

def generate_content(template_content: str, user_input: Any, user_gender: str = "other", is_suggestion: bool = False) -> str:
    """Generate document content based on template and user input"""
    groq_llm = GroqLLM()
    
    if is_suggestion:
        suggestions_text = "\n".join(user_input) if isinstance(user_input, list) else str(user_input)
        prompt = f"""
        You are a legal document drafting assistant. Refine the document based on the suggestions while maintaining legal validity.

        Original Document:
        {template_content}

        Suggestions:
        {suggestions_text}

        Generate revised document:
        """
    else:
        responses_text = "\n".join([f"Q: {r.question}\nA: {r.answer}" for r in user_input])
        prompt = f"""
        You are a legal document drafting assistant. Create a document using the template and user responses.

        Template:
        {template_content}

        User Responses:
        {responses_text}

        Generate document:
        """
    
    content = groq_llm._call(prompt)
    return adapt_content_for_gender(content, user_gender)

def create_pdf(content: str, output_path: str, line_spacing: float = 1.5):
    """Create PDF document from content"""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    frame = Frame(
        40, 40,  # x, y
        width - 80, height - 80,  # width, height
        leftPadding=0, bottomPadding=0,
        rightPadding=0, topPadding=0
    )
    
    styles = getSampleStyleSheet()
    custom_style = styles['Normal'].clone('CustomStyle')
    custom_style.fontName = 'Helvetica-Bold'
    custom_style.fontSize = 10
    custom_style.leading = 12 * line_spacing
    
    p = Paragraph(content.replace('\n', '<br/>'), custom_style)
    frame.addFromList([p], c)
    c.save()

# API endpoints
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: DocumentMetadata
):
    """Upload and process a legal document template"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    document_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        text_content = extract_text_from_pdf(file_path)
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        texts = text_splitter.split_text(text_content)
        documents = [Document(page_content=text, metadata=metadata.dict()) for text in texts]
        
        # Store in vector database
        collection_name = f"{metadata.category}_{metadata.document_type}_{document_id}"
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=os.path.join(DB_DIR, collection_name),
            embedding_function=embeddings
        )
        db.add_documents(documents)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            metadata=metadata
        )
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/query")
async def query_document(query_request: QueryRequest):
    """Query for relevant document template based on user request"""
    try:
        query_embedding = embedding_model.encode(query_request.query, convert_to_tensor=True)
        best_match = None
        highest_similarity = -1
        
        for collection_name in os.listdir(DB_DIR):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma(
                persist_directory=os.path.join(DB_DIR, collection_name),
                embedding_function=embeddings
            )
            results = db.similarity_search_with_score(query_request.query, k=1)
            
            if results and results[0][1] > highest_similarity:
                highest_similarity = results[0][1]
                best_match = {
                    'document': results[0][0],
                    'collection': collection_name
                }
        
        if not best_match:
            raise HTTPException(status_code=404, detail="No matching document found")
        
        # Generate questions for user input
        questions = generate_questions(best_match['document'].page_content)
        
        return {
            'document_id': best_match['collection'].split('_')[-1],
            'content': best_match['document'].page_content,
            'questions': questions,
            'metadata': best_match['document'].metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/generate")
async def generate_document(request: DocumentGenerationRequest):
    """Generate document based on user responses"""
    try:
        # Get document template
        collection_matches = [name for name in os.listdir(DB_DIR) if name.endswith(request.document_id)]
        if not collection_matches:
            raise HTTPException(status_code=404, detail="Document template not found")
        
        collection_name = collection_matches[0]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=os.path.join(DB_DIR, collection_name),
            embedding_function=embeddings
        )
        
        documents = db.get()
        if not documents:
            raise HTTPException(status_code=404, detail="Document content not found")
        
        # Generate document content
        content = generate_content(
            documents[0].page_content,
            request.user_responses
        )
        
        return {
            'document_id': request.document_id,
            'generated_content': content
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/suggest")
async def apply_suggestions(request: SuggestionRequest):
    """Apply user suggestions to generated document"""
    try:
        collection_matches = [name for name in os.listdir(DB_DIR) if name.endswith(request.document_id)]
        if not collection_matches:
            raise HTTPException(status_code=404, detail="Document not found")
        
        collection_name = collection_matches[0]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=os.path.join(DB_DIR, collection_name),
            embedding_function=embeddings
        )
        
        documents = db.get()
        if not documents:
            raise HTTPException(status_code=404, detail="Document content not found")
        
        # Apply suggestions
        updated_content = generate_content(
            documents[0].page_content,
            request.suggestions,
            is_suggestion=True
        )
        
        return {
            'document_id': request.document_id,
            'updated_content': updated_content
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/export")
async def export_document(document_id: str):
    """Export the generated document to PDF"""
    try:
        collection_matches = [name for name in os.listdir(DB_DIR) if name.endswith(document_id)]
        if not collection_matches:
            raise HTTPException(status_code=404, detail="Document not found")
        
        output_path = os.path.join(UPLOAD_DIR, f"generated_{document_id}.pdf")
        
        # Get the document content (you might want to store generated content separately)
        collection_name = collection_matches[0]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=os.path.join(DB_DIR, collection_name),
            embedding_function=embeddings
        )
        
        documents = db.get()
        if not documents:
            raise HTTPException(status_code=404, detail="Document content not found")
        
        create_pdf(documents[0].page_content, output_path)
        
        return {
            'document_id': document_id,
            'pdf_path': output_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)