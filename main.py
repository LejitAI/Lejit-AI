import streamlit as st
import tempfile
import time
def process_file():
    st.session_state.messages=[]
    for file in st.session_state['file_uploader']:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

            with st.session_state['feeder_spinner'], st.spinner("Uploading the document ...")
                time.delay(3)

def generate_response():
    return "Hi, I'm Lejit.ai, How can you help you today ?"

# Display all the msg stored
def display_message():
    for messages in st.session_state.messages:
        with st.chat_message(messages['role']):
            st.markdown(messages['content'])

def process_input():
    if prompt := st.chat_input("How can i help you ?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response()
        with st.chat_message("assitant"):
            st.markdown(response)

        st.session_state.messages.append({"role":"assitant", "content": response})
    
def main():
    st.title("Lejit Test Interface")

    if len(st.session_state)==0:
        st.session_state.messages = []



    st.file_uploader(
        "Upload the legal document",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
        )

    st.session_state['feeder_spinner'] = st.empty()
    
    display_message()
    process_input()


if __name__:
    main()
