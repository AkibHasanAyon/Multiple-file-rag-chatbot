import os
import pickle
import faiss
import numpy as np
import sqlite3
import streamlit as st
from dotenv import load_dotenv
import requests
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load environment variables from .env
load_dotenv()

# Load API keys and URL from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_API_URL = os.getenv('GEMINI_API_URL', '')

# Define the correct path for the FAISS index and metadata
BASE_DIR = os.getcwd()  # Get current working directory
INDEX_DIR = os.path.join(BASE_DIR, 'rag_data', 'faiss_index')
META_PATH = os.path.join(BASE_DIR, 'rag_data', 'metadata.pkl')

# Load the FAISS index and metadata
index_path = os.path.join(INDEX_DIR, 'faiss.index')
index = faiss.read_index(index_path)

with open(META_PATH, 'rb') as f:
    chunked_docs = pickle.load(f)

# Initialize SQLite Database for Chat History
DB_PATH = "chat_history.sqlite"

def init_chat_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()

def save_chat(role: str, message: str):
    """Save chat message into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_history (role, message) VALUES (?, ?)", (role, message))
        conn.commit()

def load_chat_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Load chat history from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT role, message, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    return [{"role": r, "message": m, "timestamp": t} for r, m, t in reversed(rows)]

def delete_chat_history():
    """Delete chat history from the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_history")
        conn.commit()

# Initialize the database
init_chat_db()

# Initialize the embeddings model
model_name = "BAAI/bge-base-en-v1.5"  # Use the model name you want to work with
encode_kwargs = {'normalize_embeddings': True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # 'cuda' if you're using a GPU
    encode_kwargs=encode_kwargs
)

# Define Function for Generating Responses with Gemini
def gemini_generate(prompt: str, history: List[Dict[str, str]] = None) -> str:
    """Send prompt to Gemini API and return response text."""
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    contents = []
    if history:
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["message"]}]} )
    
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {"contents": contents}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for non-2xx responses
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        st.error(f"Gemini API error: {e}")
        return "Sorry, there was an error processing your request."

# Streamlit chatbot UI
st.title("RAG Chatbot with Gemini")

# Initialize session state for conversation history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Load the conversation history into session state
history = load_chat_history(limit=5)
for message in history:
    st.session_state["messages"].append({"role": message['role'], "message": message['message']})

# Display the conversation
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# User input and response handling
user_input = st.chat_input("Type your message here:")

if user_input:
    # Save the user's input to the database and session state
    save_chat("user", user_input)
    st.session_state["messages"].append({"role": "user", "message": user_input})

    # Retrieve relevant chunks using FAISS
    query_vec = embeddings_model.embed_query(user_input)
    query_vec = np.array([query_vec], dtype='float32')  # Convert to numpy array
    faiss.normalize_L2(query_vec)  # Normalize the query vector

    if index.is_trained:
        scores, indices = index.search(query_vec, 5)  # Search for the top 5 matches in the FAISS index
    else:
        st.error("Index not trained yet. Please ensure the index is available.")
        indices = []

    # Construct context text from the retrieved chunks
    context_text = "\n\n".join([f"[Source: {chunked_docs[idx]['source']}] {chunked_docs[idx]['text']}" for idx in indices[0]])

    # Build the prompt for Gemini
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_input}"

    # Show loading spinner while waiting for the response
    with st.spinner("Thinking..."):
        answer = gemini_generate(prompt, st.session_state["messages"])

    # Save the assistant's response to the database and session state
    save_chat("assistant", answer)
    st.session_state["messages"].append({"role": "assistant", "message": answer})

    # Display the user input first and then the assistant's response
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(answer)

# Add Delete Chat History Button
if st.button("Delete Chat History"):
    delete_chat_history()  # Clear chat history from the database
    st.session_state["messages"] = []  # Clear chat history from the session state
    st.success("Chat history has been deleted.")
