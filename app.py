import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set API Key from Streamlit Secrets
def set_openai_api_key():
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

def get_conversation_chain(vectorstore, k=5):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": k}), memory=memory
    )

def aggregate_answers(chunks):
    combined_answer = ""
    for chunk in chunks:
        combined_answer += chunk['text'] + "\n"
    return combined_answer.strip()

def handle_userinput(user_question):
    # Ensure conversation and vectorstore are initialized
    if st.session_state.conversation and "vectorstore" in st.session_state:
        vectorstore = st.session_state.vectorstore  # Access vectorstore from session state

        if "list" in user_question.lower() or "all" in user_question.lower():
            st.session_state.conversation = get_conversation_chain(vectorstore, k=20)  # Retrieve more chunks if needed
        else:
            st.session_state.conversation = get_conversation_chain(vectorstore, k=5)  # Default chunk retrieval

        response = st.session_state.conversation({'question': user_question})
        aggregated_answer = aggregate_answers(response['chunks'])  # Aggregate the chunks before sending back

        # Append the new question and response to the existing conversation history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": aggregated_answer})

        # Display the updated chat history
        display_chat_history()

def display_chat_history():
    chat_history_container = st.container()
    for message in st.session_state.chat_history:
        if len(message["content"]) > 0:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

def main():
    set_openai_api_key()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    st.title("💻 Chat with Multiple PDFs")

    # Initialize session states for chat and history if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  # Make sure vectorstore is initialized in session state

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
                st.session_state.vectorstore = vectorstore  # Store vectorstore in session state
                st.session_state.conversation = get_conversation_chain(vectorstore)

            st.success("PDFs successfully processed!")

    # Disable user input until the PDFs are uploaded and processed
    if st.session_state.conversation:
        user_question = st.chat_input("💬 Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
        display_chat_history()
    else:
        st.warning("Please upload and process the PDFs before asking questions.")

if __name__ == "__main__":
    main()
