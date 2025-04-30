# 💬 Chat with Multiple PDFs

This Streamlit app allows users to upload and chat with multiple PDF documents using OpenAI's GPT model. It can summarize PDFs, count words, and provide intelligent, context-aware responses. The app also supports greetings/farewell detection and converts chatbot replies to speech.

---

## 🚀 Features

- 📄 **Multi-PDF Upload** – Upload multiple PDF files for document-based chat.
- 💬 **Conversational AI** – Ask questions and get responses grounded in the uploaded documents.
- 📚 **Document Summarization** – Automatically summarize the content of each uploaded PDF.
- 🔢 **Word Count** – Count words for each document or all at once.
- 🙋 **Greetings & Goodbyes** – Friendly bot replies to "hi", "bye", etc.
- 🔊 **Text-to-Speech** – Read responses aloud using gTTS.
- 📥 **Chat Export** – Download your chat conversation as a PDF.

---

## 📦 Requirements
- streamlit
- langchain
- langchain-community
- PyPDF2
- openai
- tiktoken
- faiss-cpu
- fpdf
- pytz
- gTTS
