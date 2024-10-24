# main.py
import streamlit as st
import tempfile
import os
from rag import LegalRag

def init_session_state():
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = LegalRag()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feeder_spinner" not in st.session_state:
        st.session_state["feeder_spinner"] = st.empty()
    if "pdf_processed" not in st.session_state:
        st.session_state["pdf_processed"] = False
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False
    if "show_file_uploader" not in st.session_state:
        st.session_state["show_file_uploader"] = True
    if "document_type" not in st.session_state:
        st.session_state["document_type"] = None
    if "context_enabled" not in st.session_state:
        st.session_state["context_enabled"] = True

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def process_file(file_uploader):
    try:
        st.session_state["assistant"].clear()
        st.session_state.messages = []

        document_type = st.selectbox(
            "Select document type",
            ["Contract", "Legal Brief", "Court Document", "Legislation", "Regulatory Filing", "Other"]
        )
        st.session_state["document_type"] = document_type

        for file in file_uploader:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.session_state["feeder_spinner"], st.spinner(f"Processing {document_type}..."):
                st.session_state["assistant"].feed(file_path, document_type)
            os.remove(file_path)

        st.session_state["pdf_processed"] = True
        st.session_state["file_uploaded"] = True
        st.session_state["show_file_uploader"] = False
        st.success(f"{document_type} processed successfully!")
        
        system_msg = f"I have processed your {document_type.lower()}. I can help you understand its contents, identify key legal provisions, and explain complex legal terminology. Remember that I provide information, not legal advice. What would you like to know about the document?"
        st.session_state.messages.append({"role": "assistant", "content": system_msg})
        with st.chat_message("assistant"):
            st.markdown(system_msg)
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.session_state["pdf_processed"] = False

def process_input(prompt, file_uploaded):
    if prompt or file_uploaded:
        if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["content"] != prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

        try:
            if not file_uploaded:
                response = st.session_state["assistant"].ask_text_only(prompt)
            elif st.session_state["pdf_processed"]:
                response = st.session_state["assistant"].ask_combined(prompt)
            else:
                response = "Please upload and process a legal document first."

            if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["content"] != response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

def reset_upload_state():
    st.session_state["show_file_uploader"] = True
    st.session_state["file_uploaded"] = False
    st.session_state["pdf_processed"] = False
    st.session_state["document_type"] = None
    # Clear conversation history when uploading new document
    st.session_state["assistant"].clear()
    st.session_state.messages = []

def main():
    st.title("Legal Document Assistant")
    
    st.markdown("""
    > **Disclaimer:** This tool provides legal information for educational purposes only. 
    > It is not a substitute for professional legal advice. Please consult with a qualified attorney for specific legal guidance.
    """)
    
    init_session_state()
    
    # Add conversation context toggle
    with st.sidebar:
        st.header("Settings")
        st.session_state["context_enabled"] = st.toggle(
            "Enable conversation memory",
            value=True,
            help="When enabled, the assistant remembers previous questions and provides more contextual responses"
        )
    
    display_messages()

    if not st.session_state["show_file_uploader"]:
        if st.button("Upload New Legal Document"):
            reset_upload_state()
            st.rerun()

    if st.session_state["show_file_uploader"]:
        file_uploader = st.file_uploader(
            "Upload legal document", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload legal documents (contracts, briefs, court documents, etc.)"
        )
        if file_uploader:
            process_file(file_uploader)

    with st.form("input_form"):
        prompt = st.text_input(
            "Ask about the legal document or general legal questions",
            key="user_input",
            help="Ask questions about document content, legal terms, or general legal concepts"
        )
        submit_button = st.form_submit_button("Submit Question")

    if submit_button:
        if not st.session_state["context_enabled"]:
            # Clear memory but keep messages for display
            st.session_state["assistant"].memory.clear()
        process_input(prompt, st.session_state["file_uploaded"])

if __name__ == "__main__":
    main()