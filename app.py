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
        st.session_state.chat_history = response['chat_history']
        
        # Create a scrollable chat container
        with st.container():
            chat_history_container = st.empty()  # Creating a dynamic container for chat history
            
            # Append the conversation history
            for message in reversed(st.session_state.chat_history):
                if len(message.content) > 0:
                    if len(st.session_state.displayed_messages) == 0 or st.session_state.displayed_messages[-1]['user'] != user_question:
                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(user_question)
                        # Display assistant message
                        with st.chat_message("assistant"):
                            st.markdown(message.content)

                        st.session_state.displayed_messages.append({'user': user_question, 'assistant': message.content})

            # Scroll the container to the latest messages
            chat_history_container.write('')  # This will make the container scrollable when new messages are added.

def main():
    # Set the OpenAI API key from Streamlit secrets
    set_openai_api_key()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")
    st.title("💻 Chat with Multiple PDFs")

    # Initialize session states for chat and history if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
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

if __name__ == "__main__":
    main()
