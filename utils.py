import random
import difflib
import re
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader

# Greeting and farewell keyword libraries
GREETING_KEYWORDS = {"hi", "hello", "hey", "how are you", "good morning", "good afternoon", "greetings"}
FAREWELL_KEYWORDS = {"bye", "goodbye", "thank you", "thanks", "see you", "take care", "farewell"}

# Response libraries
GREETING_REPLIES = [
    "Hello! How can I help you today?",
    "Hi there! What would you like to know?",
    "Hey! I'm here to help you with your PDFs.",
    "Greetings! Ready when you are.",
    "Hi! Ask me anything from your documents."
]

FAREWELL_REPLIES = [
    "Goodbye! Let me know if you need help again.",
    "Take care! Come back anytime.",
    "You're welcome! Have a great day!",
    "Bye! It was nice chatting with you.",
    "Thanks! I'm here if you need me again."
]

def handle_greeting(user_input: str):
    lower_input = user_input.lower().strip()
    if any(keyword in lower_input for keyword in GREETING_KEYWORDS) and len(lower_input.split()) <= 4:
        return random.choice(GREETING_REPLIES)
    return None

def handle_farewell(user_input: str):
    lower_input = user_input.lower().strip()
    if any(keyword in lower_input for keyword in FAREWELL_KEYWORDS) and len(lower_input.split()) <= 4:
        return random.choice(FAREWELL_REPLIES)
    return None

def get_labeled_documents(pdf_docs):
    labeled_docs = []
    for i, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        doc_name = pdf.name
        label = f"Document {i+1}: {doc_name}"
        labeled_docs.append({"label": label, "text": text})
    return labeled_docs

def summarize_documents(labeled_docs):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    summaries = []
    for doc in labeled_docs:
        prompt = f"Please summarize the following document:\n\n{doc['text'][:4000]}\n\nSummary:"
        response = llm.predict(prompt)
        summaries.append({"label": doc["label"], "summary": response.strip()})
    return summaries

def is_summary_question(question: str) -> bool:
    summary_keywords = ["summary", "summarize", "overview", "main idea", "key points", "what is in", "content of"]
    return any(keyword in question.lower() for keyword in summary_keywords)

def extract_target_doc_label(question: str, docs: list, cutoff: float = 0.6) -> str:
    question_lower = question.lower()
    labels = [item['label'].lower() for item in docs]

    doc_num_match = re.search(r"(document|doc|file)\s*(\d+)", question_lower)
    if doc_num_match:
        doc_num = int(doc_num_match.group(2))
        if 1 <= doc_num <= len(docs):
            return docs[doc_num - 1]["label"]
    
    matches = difflib.get_close_matches(question_lower, labels, n=1, cutoff=cutoff)

    if matches:
        matched_label_lower = matches[0]
        for item in docs:
            if item['label'].lower() == matched_label_lower:
                return item['label']
    
    for item in docs:
        if item['label'].lower() in question_lower:
            return item['label']
    
    return None

def is_wordcount_question(question: str) -> bool:
    keywords = ["word count", "how many words", "number of words", "words in", "wordcount"]
    return any(keyword in question.lower() for keyword in keywords)

def count_words_in_documents(labeled_docs):
    word_counts = []
    for doc in labeled_docs:
        word_count = len(doc["text"].split())
        word_counts.append({
            "label": doc["label"],
            "word_count": word_count
        })
    return word_counts
