css = """
<style>
/* Container for the chat interface */
.chat-container {
    display: flex;
    flex-direction: column;
    height: 80vh;
    overflow-y: auto;
    padding: 20px;
    background-color: #f9f9f9;
}

/* The messages container */
.chat-message {
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 80%;
    word-wrap: break-word;
    line-height: 1.5;
}

/* User message style */
.user {
    background-color: #DCF8C6;
    text-align: right;
    margin-left: auto;
}

/* Bot message style */
.bot {
    background-color: #EAEAEA;
    text-align: left;
    margin-right: auto;
}

/* Scrollable area for the messages */
.chat-container {
    overflow-y: scroll;
}

/* Input box for the user */
.input-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px;
    background-color: #fff;
    border-top: 1px solid #ddd;
}

/* Style for the text input */
.text-input {
    width: 80%;
    padding: 10px;
    border-radius: 30px;
    border: 1px solid #ccc;
    font-size: 16px;
}

/* Button to send the input */
.send-btn {
    background-color: #4CAF50;
    color: white;
    padding: 12px 20px;
    margin-left: 10px;
    border: none;
    border-radius: 50%;
    cursor: pointer;
}

.send-btn:hover {
    background-color: #45a049;
}

</style>
"""

user_template = """
<div class="chat-message user">
    <p>{{MSG}}</p>
</div>
"""

bot_template = """
<div class="chat-message bot">
    <p>{{MSG}}</p>
</div>
"""
