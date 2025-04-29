import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from fpdf import FPDF
from datetime import datetime
import pytz
import re
from io import BytesIO

# Set API Key from Streamlit Secrets
def set_openai_api_key():
    # Fetch the OpenAI API key from Streamlit secrets
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    # Instantiate the OpenAI Chat model with the API key
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Or use another OpenAI model here
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Using the API key from the environment variable
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

def handle_userinput(user_question):
    if st.session_state.conversation:
        # Send the user's question and get a response
        response = st.session_state.conversation({'question': user_question})

        # Ensure the history is updated correctly
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

        # Display the updated chat history in a scrollable container
        display_chat_history()

def display_chat_history():
    # Create a dynamic container to update chat history
    chat_history_container = st.container()

    # Displaying all messages in order
    for message in st.session_state.chat_history:
        if len(message["content"]) > 0:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

# ===== Chat Saving to PDF =====
def estimate_multicell_height(pdf, text, width, line_height):
    lines = pdf.multi_cell(width, line_height, text, split_only=True)
    return len(lines) * line_height + 4  # +4 for padding

def save_chat_to_pdf(chat_history):
    def strip_emojis(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def remove_newlines(text):
        return re.sub(r'\s*\n\s*', ' ', text.strip())

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    page_height = 297  # A4 height in mm
    margin_top = 10
    margin_bottom = 10
    usable_height = page_height - margin_top - margin_bottom
    line_height = 8
    box_spacing = 6
    box_width = 190

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chat History", ln=True, align="C")
    pdf.set_font("Arial", '', 10)
    malaysia_time = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%B %d, %Y %H:%M")
    pdf.cell(0, 10, f"Exported on {malaysia_time} (MYT)", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)

    # Only process if chat history is available
    if chat_history:
        for entry in chat_history:
            # Ensure there are valid user and assistant messages
            user_msg = entry.get("user", "").strip()
            assistant_msg = entry.get("assistant", "").strip()

            # If either message is missing, skip the entry
            if not user_msg or not assistant_msg:
                continue

            label_user = f"You:\n{user_msg}"
            label_assistant = f"Assistant:\n{assistant_msg}"

            # Estimate heights
            user_box_height = estimate_multicell_height(pdf, label_user, box_width, line_height)
            assistant_box_height = estimate_multicell_height(pdf, label_assistant, box_width, line_height)
            total_pair_height = user_box_height + assistant_box_height + box_spacing

            # If not enough space, start new page
            if pdf.get_y() + total_pair_height > usable_height:
                pdf.add_page()

            # Render You box
            y_start = pdf.get_y()
            pdf.rect(10, y_start, box_width, user_box_height)
            pdf.set_xy(12, y_start + 2)
            pdf.multi_cell(0, line_height, label_user)
            pdf.ln(2)

            # Render Assistant box
            y_start = pdf.get_y()
            pdf.rect(10, y_start, box_width, assistant_box_height)
            pdf.set_xy(12, y_start + 2)
            pdf.set_text_color(0, 102, 204)
            pdf.multi_cell(0, line_height, label_assistant)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

        # Output PDF
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return BytesIO(pdf_bytes)

    else:
        # If no chat history exists, return an empty file or a message
        return None

# Main Application Logic
def main():
    # Set the OpenAI API key from Streamlit secrets
    set_openai_api_key()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    st.title("💻 Chat with Multiple PDFs")

    # Initialize session states for chat and history if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history as a list to accumulate the conversation
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []  # Initialize displayed messages

    # Sidebar for PDF Upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("📄 Upload your PDFs here", accept_multiple_files=True)
        process_button = st.button("Process")

        if pdf_docs and process_button:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

            # Display success message after processing is complete
            st.success("PDFs successfully processed!")

    # Disable user input until the PDFs are uploaded and processed
    if st.session_state.conversation:
        user_question = st.chat_input("💬 Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.warning("Please upload and process the PDFs before asking questions.")

    # Add Chat History to PDF
    with st.sidebar:
        if st.session_state.chat_history:
            st.download_button(
                label="📥 Download Chat History as PDF",
                data=save_chat_to_pdf(st.session_state.chat_history),
                file_name="chat_history.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No chat history to export.")

if __name__ == "__main__":
    main()
