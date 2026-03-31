from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv
from bson import ObjectId
from pymongo import MongoClient
from urllib.parse import quote_plus, urlparse, urlunparse
import os
import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey_change_in_production')

# MongoDB Atlas — auto-encode special characters in credentials
def _encode_mongo_uri(uri):
    """URL-encode username and password in a MongoDB URI to handle special characters."""
    try:
        parsed = urlparse(uri)
        if parsed.username and parsed.password:
            user = quote_plus(parsed.username)
            pwd  = quote_plus(parsed.password)
            host = parsed.hostname
            netloc = f"{user}:{pwd}@{host}"
            if parsed.port:
                netloc += f":{parsed.port}"
            uri = urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass
    return uri

MONGO_URI = _encode_mongo_uri(os.environ.get('MONGO_URI', ''))
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['medical_chatbot']
users_col = db['users']
conversations_col = db['conversations']

# Flask-Login & Bcrypt
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc['_id'])
        self.username = user_doc['username']

@login_manager.user_loader
def load_user(user_id):
    doc = users_col.find_one({'_id': ObjectId(user_id)})
    return User(doc) if doc else None

# AI Setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
client = Groq(api_key=GROQ_API_KEY)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Helper: Ask LLM
def ask(query, chat_history):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    history_text = ""
    for msg in chat_history:
        history_text += f"User: {msg['user']}\nAssistant: {msg['bot']}\n"

    prompt_text = f"""You are a Medical assistant for question-answering tasks.

Use the conversation history and context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.
Use ONLY the relevant parts of the context.
If the context contains unrelated information, ignore it.
If the context does not clearly answer the question, say: "I don't have enough information."

Chat History:
{history_text}

Context:
{context}

Current Question:
{query}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )
    return response.choices[0].message.content

# Auth Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_doc = users_col.find_one({'username': username})
        if user_doc and bcrypt.check_password_hash(user_doc['password'], password):
            login_user(User(user_doc))
            return redirect(url_for('chat'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif users_col.find_one({'username': username}):
            error = 'Username already taken. Please choose another.'
        else:
            hashed = bcrypt.generate_password_hash(password).decode('utf-8')
            users_col.insert_one({'username': username, 'password': hashed})
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Chat Page
@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', username=current_user.username)

# Conversation APIs
@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    conv = {
        'user_id': current_user.id,
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.datetime.utcnow()
    }
    result = conversations_col.insert_one(conv)
    return jsonify({'conversation_id': str(result.inserted_id)})

@app.route('/conversations', methods=['GET'])
@login_required
def get_conversations():
    convs = list(conversations_col.find(
        {'user_id': current_user.id},
        {'title': 1, 'created_at': 1}
    ).sort('created_at', -1))
    for c in convs:
        c['_id'] = str(c['_id'])
        c['created_at'] = c['created_at'].isoformat()
    return jsonify(convs)

@app.route('/conversation/<conv_id>', methods=['GET'])
@login_required
def get_conversation(conv_id):
    conv = conversations_col.find_one({
        '_id': ObjectId(conv_id),
        'user_id': current_user.id
    })
    if not conv:
        return jsonify({'error': 'Not found'}), 404
    conv['_id'] = str(conv['_id'])
    conv['created_at'] = conv['created_at'].isoformat()
    return jsonify(conv)

@app.route('/conversation/<conv_id>', methods=['DELETE'])
@login_required
def delete_conversation(conv_id):
    conversations_col.delete_one({'_id': ObjectId(conv_id), 'user_id': current_user.id})
    return jsonify({'success': True})

# Chat (send message)
@app.route('/get', methods=['POST'])
@login_required
def get_response():
    msg = request.form.get('msg', '')
    conv_id = request.form.get('conversation_id', '')

    # Load existing history from MongoDB
    conv = None
    if conv_id:
        try:
            conv = conversations_col.find_one({
                '_id': ObjectId(conv_id),
                'user_id': current_user.id
            })
        except Exception:
            conv = None

    # If no conversation found, create one
    if not conv:
        new_conv = {
            'user_id': current_user.id,
            'title': msg[:50],
            'messages': [],
            'created_at': datetime.datetime.utcnow()
        }
        result = conversations_col.insert_one(new_conv)
        conv_id = str(result.inserted_id)
        conv = new_conv
        conv['messages'] = []

    chat_history = conv.get('messages', [])
    answer = ask(msg, chat_history)

    # Append new exchange
    new_message = {'user': msg, 'bot': answer}
    chat_history.append(new_message)

    # Update title from first message
    title = chat_history[0]['user'][:50] if chat_history else 'New Chat'

    conversations_col.update_one(
        {'_id': ObjectId(conv_id)},
        {'$set': {'messages': chat_history, 'title': title}}
    )

    return jsonify({'answer': answer, 'conversation_id': conv_id})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)