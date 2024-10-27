

import os
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from typing import Any, List, Mapping, Optional, ClassVar , List
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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame
from typing import List, Dict, Optional

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
client = Groq(api_key='key')

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

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat_completion = client.chat.completions.create(
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

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)[:63]

def process_pdf_file(pdf_path: str, embeddings, prompt: PromptTemplate, user_gender: str):
    text_content = extract_text_from_pdf(pdf_path)
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_text(text_content)
    documents = [Document(page_content=text) for text in texts]
    collection_name = sanitize_filename(os.path.basename(pdf_path))
    db = Chroma.from_documents(documents, embeddings, collection_name=collection_name)

    groq_llm = GroqLLM()
    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    initial_query = "Please explain the context of the document, noting any information that may be misleading or incorrect."
    details_query = "List all the specific details that need to be filled in this document, emphasizing any potential inaccuracies. Ask as many questions as necessary to clarify and correct the information and fill the details, providing each question on a new line."

    summary_result = qa_chain(initial_query)
    combined_query = f"{summary_result['result']} {details_query}"
    contextual_result = qa_chain(combined_query)

    user_responses = collect_user_responses(contextual_result['result'], user_gender)
    return user_responses, text_content

def collect_user_responses(questions: str, user_gender: str) -> dict:
    user_responses = {}
    print("\nPlease provide the following details:")
    for question in questions.strip().split('\n'):
        question = question.strip()
        if question:
            question = adapt_question_for_gender(question, user_gender)
            user_input = input(f"{question}\nYour answer: ")
            user_responses[question] = user_input
            print()

    print("\nCollected Details from User:")
    for question, response in user_responses.items():
        print(f"{question}: {response}")

    return user_responses

def adapt_question_for_gender(question: str, user_gender: str) -> str:
    if user_gender.lower() == 'male':
        return question.replace("user", "he")
    elif user_gender.lower() == 'female':
        return question.replace("user", "she")
    return question


from typing import List

def allow_user_suggestions(original_content: str, user_input: List[str]) -> str:
    """
    Allow users to provide suggestions for edits to the original content.

    Parameters:
    - original_content (str): The original content to be edited.
    - user_input (List[str]): A list of initial suggestions or context (if needed).

    Returns:
    - str: The final updated document content after all user suggestions.
    """
    print("\nPlease provide your suggestions for edits. Type 'done' when finished.")

    suggestions = []
    updated_document_content = original_content  # Initialize with the original content

    while True:
        suggestion = input("Your suggestion (or 'done' to exit): ")
        if suggestion.lower() == 'done':
            break
        if suggestion:  # Only add non-empty suggestions
            suggestions.append(suggestion)

        # Generate the updated document after each suggestion
        updated_document_content = generate_content(updated_document_content, suggestions, is_suggestion=True)
        print("\nUpdated Document Content:")
        print(updated_document_content)  # Display the generated document after each suggestion

    return updated_document_content


def generate_content(template_content: str, user_input: Any, is_suggestion: bool = False) -> str:
    groq_llm = GroqLLM()

    if is_suggestion:
        suggestions_text = "\n".join(user_input) if user_input else "No suggestions provided."
        prompt = f"""
        You are a highly advanced legal document drafting assistant. Your task is to refine the provided document based on the user's suggestions.

        Original Document Content:
        {template_content}

        User Suggestions for Edits:
        {suggestions_text}

        Generate a revised legal document that incorporates all suggestions seamlessly, maintaining clarity and legal soundness. Provide only the final output without any additional commentary or explanations.
        Try to include the heading of the document too.
        Final Output:
        """
    else:
        prompt = f"""
        You are a highly advanced legal document drafting assistant. Your role is to craft precise and contextually appropriate legal documents.

        Template Content:
        {template_content}

        User Responses:
        {user_input}

        Generate a legal document that integrates the user responses, ensuring clarity, accuracy, and legal validity. Provide only the final output without any additional commentary or explanations.
        Try to include the heading of the document too.

        Final Output:
        """

    return groq_llm(prompt)



# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

# Function to process each PDF, extract heading and generate summary
def process_pdfs_in_folder(folder_path: str) -> List[Dict[str, str]]:
    data = []

    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)

                # Get the heading (first line of text)
                heading = text.split('\n')[0] if text else "No heading found"

                # Generate a brief description of the document (simplified for demonstration)
                description = f"Document about {heading}" if heading else "No description available"

                data.append({
                    "folder_name": folder_name,
                    "file_name": file,
                    "description": description,
                    "heading": heading,  # Store only the heading of the document
                    "full_text": text
                })

    return data

# Function to create embeddings for each document description
def create_embeddings(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for item in data:
        # Create tensor for each description
        item['embedding'] = embedding_model.encode(item['description'], convert_to_tensor=True)
    return data

# Function to query descriptions and retrieve the best matching file name using embeddings
def query_files_with_embeddings(data: List[Dict[str, str]], user_query: str) -> Optional[str]:
    # Create embedding for the user query
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)

    # Extract embeddings for all document descriptions and stack them into a single tensor
    descriptions = torch.stack([item['embedding'] for item in data])

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, descriptions)

    # Find the most relevant document
    best_match_idx = similarities.argmax().item()
    return data[best_match_idx]['file_name']

def create_pdf(content, output_path, line_spacing=1.5):

    use_font = 'Helvetica-Bold'

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Create a frame for the entire page
    frame = Frame(40, 40, width - 80, height - 80, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)

    # Create a style with the chosen font
    styles = getSampleStyleSheet()
    custom_style = styles['Normal'].clone('CustomStyle')
    custom_style.fontName = use_font
    custom_style.fontSize = 10  # Adjust as needed
    custom_style.leading = 12 * line_spacing  # Set leading based on line spacing

    # Create a Paragraph with the content
    p = Paragraph(content.replace('\n', '<br/>'), custom_style)

    # Draw the paragraph in the frame
    frame.addFromList([p], c)

    c.save()
from IPython.core.display import display, HTML

def main():
  #done only for will for now , need to change it for the full folder
    folder_path = '/content/will'
    output_path = '/content/output.pdf'
    # Main execution


    # Process PDFs and generate summaries
    pdf_data = process_pdfs_in_folder(folder_path)

    # Create embeddings for document descriptions
    pdf_data = create_embeddings(pdf_data)

    # Example user query
    #user_query = "i have a son aged 10 , i need to write a will to give all my assets to him"
    #user_query = "i need to write a will"
    user_query = input("Enter Your Query: ")

    matching_file_name = query_files_with_embeddings(pdf_data, user_query)

    # Find the matching file's full text content
    matching_file_content = ""
    for item in pdf_data:
        if item['file_name'] == matching_file_name:
            matching_file_content = item['full_text']
            break

    print(f"Matching File: {matching_file_name}")
    print(f"Content:\n{matching_file_content}")

    template_path =os.path.join(folder_path, f"{matching_file_name}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    SYSTEM_PROMPT = """
    You are an advanced legal document drafting assistant, designed to deliver precise and contextually relevant documents with exceptional accuracy. Your role is to engage with the user in a proactive manner, systematically gathering all essential details to create tailored legal documents.
    """

    template = f"""
    {{context}}

    Question: {{question}}

    {SYSTEM_PROMPT}
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Ask for user's gender before processing the PDF
    user_gender = input("Please specify your gender (male/female/other): ").strip().lower()

    user_responses, template_content = process_pdf_file(template_path, embeddings, prompt, user_gender)
    new_pdf_content = generate_content(template_content, user_responses)

    print("Generated PDF Content:")
    print(new_pdf_content)

    user_suggestions = allow_user_suggestions(new_pdf_content, user_responses)

    #final_document_content = generate_content(new_pdf_content, user_suggestions, is_suggestion=True)

    print("Final Document Content:")
    print(user_suggestions)

    # Ensure entire content is displayed in the cell


    # TODO: Add code to save the final_document_content to a new PDF file
    create_pdf(user_suggestions, output_path)


if __name__ == "__main__":
    main()

