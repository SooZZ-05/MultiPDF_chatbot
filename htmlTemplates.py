# htmlTemplates.py
css = """
<style>
.chat-message {
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
.user {
    background-color: #DCF8C6;
    text-align: right;
}
.bot {
    background-color: #EAEAEA;
    text-align: left;
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
