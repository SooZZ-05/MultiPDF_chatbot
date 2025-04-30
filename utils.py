import random

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
