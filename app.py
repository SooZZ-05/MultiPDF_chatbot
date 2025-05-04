import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np
import base64
from gtts import gTTS
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils import handle_greeting, handle_farewell, summarize_documents, is_summary_question, extract_target_doc_label, get_labeled_documents_from_any, is_wordcount_question, count_words_in_documents, save_chat_to_pdf

# Set API Key from Streamlit Secrets
def set_openai_api_key():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_document_text(docs):
    text = ""
    for doc in docs:
        name = doc.name.lower()
        if name.endswith(".pdf"):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        elif name.endswith(".docx"):
            word_doc = Document(doc)
            for para in word_doc.paragraphs:
                text += para.text + "\n"
        elif name.endswith(".txt"):
            stringio = BytesIO(doc.read())
            text += stringio.read().decode("utf-8") + "\n"
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
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 20}), memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def handle_userinput(user_question):
    greeting_reply = handle_greeting(user_question)
    if greeting_reply:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": greeting_reply})
        return

    farewell_reply = handle_farewell(user_question)
    if farewell_reply:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": farewell_reply})
        return

    if is_summary_question(user_question) and "doc_summaries" in st.session_state:
        summaries = st.session_state.doc_summaries
        target_label = extract_target_doc_label(user_question, summaries)

        if target_label:
            matched = next((s for s in summaries if s["label"].lower() == target_label.lower()), None)
            if matched:
                summary_response = f"### {matched['label']}\n{matched['summary']}"
            else:
                summary_response = f"Sorry, I couldn't find a document matching '{target_label}'."
        else:
            summary_response = "\n\n".join([f"### {s['label']}\n{s['summary']}" for s in summaries])

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": summary_response})
        return

    if is_wordcount_question(user_question) and "word_counts" in st.session_state:
        word_counts = st.session_state.word_counts
        target_label = extract_target_doc_label(user_question, word_counts)
    
        if target_label:
            matched = next((w for w in word_counts if w["label"].lower() == target_label.lower()), None)
            if matched:
                word_count_response = f"**{matched['label']}** has **{matched['word_count']}** words."
            else:
                word_count_response = f"Sorry, I couldn't find a document matching '{target_label}'."
        else:
            all_counts = [f"**{w['label']}**: {w['word_count']} words" for w in word_counts]
            word_count_response = "Here are the word counts for all documents:\n\n" + "\n".join(all_counts)
    
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": word_count_response})
        return
        
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        answer = response.get('answer', '').strip()
        source_docs = response.get('source_documents', [])

        grounded = False
        if answer and source_docs:
            embedder = OpenAIEmbeddings()
            answer_embedding = embedder.embed_query(answer)
        
            doc_texts = [doc.page_content for doc in source_docs]
            chunk_embeddings = embedder.embed_documents(doc_texts)

            doc_similarities = [
                cosine_similarity(answer_embedding, chunk_embedding)
                for chunk_embedding in chunk_embeddings
            ]

            max_similarity = max(doc_similarities)
            grounded = max_similarity >= 0.7
        
        if not grounded:
            answer = "I'm sorry, but I couldn't find an answer to that question in the documents you provided."
    
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

def auto_play_audio_streamlit(text, lang="en"):
    tts = gTTS(text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    st.audio(mp3_fp, format='audio/mp3')

def display_chat_history():
    chat_history_container = st.container()
    for i, message in enumerate(st.session_state.chat_history):
        key = f"show_audio_{i}"
        if key not in st.session_state:
            st.session_state[key] = False

        with chat_history_container:
            col1, col2 = st.columns([10, 1])
            with col1:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if st.session_state[key]:
                        auto_play_audio_streamlit(message["content"])
            with col2:
                toggle_text = "❌" if st.session_state[key] else "📢"
                if st.button(toggle_text, key=f"toggle_btn_{i}"):
                    st.session_state[key] = not st.session_state[key]

def main():
    set_openai_api_key()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:", layout="wide")
    st.title("💻 ChatBot for Laptop recommendation")

    # Initialize session states for chat and history if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "displayed_messages" not in st.session_state:
        st.session_state.displayed_messages = []

    # Sidebar for PDF Upload
    with st.sidebar:
        st.subheader("Your Documents")
        try:
            docs = st.file_uploader(
                "📄 Upload up to 3 documents (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
    
            if docs and len(docs) > 3:
                st.error("You can upload a maximum of 3 documents.")
                docs = docs[:3]  # Optional: truncate to first 3 files
            else:
                process_button = st.button("Process")
    
        except Exception as e:
            st.error(f"An error occurred during upload: {str(e)}")
            
        if docs:
            process_button = st.button("Process")
            if process_button:
                with st.spinner("Processing..."):
                    labeled_docs = get_labeled_documents_from_any(docs)
                    st.session_state.labeled_docs = labeled_docs
                    doc_summaries = summarize_documents(labeled_docs)
                    st.session_state.doc_summaries = doc_summaries
                    st.session_state.word_counts = count_words_in_documents(labeled_docs)
    
                    raw_text = get_document_text(docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
    
                st.success("PDFs successfully processed!")

        st.subheader("Chat Options")
        save_chat_button = st.button("💾 Save Chat to PDF")
        if save_chat_button and st.session_state.chat_history:
            chat_pdf = save_chat_to_pdf(st.session_state.chat_history)
            st.download_button(
                label="📥 Download Chat History PDF",
                data=chat_pdf,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )
            st.success("Chat history saved as PDF!")

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
