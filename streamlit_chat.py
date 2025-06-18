import streamlit as st
import time
import random
import requests
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Monitor",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalist chat-like interface
st.markdown("""
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 8px; 
    margin-bottom: 0.8rem; 
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.chat-message.user {
    background-color: #f8f9fa;
    border-left: 3px solid #007bff;
}
.chat-message.assistant {
    background-color: #ffffff;
    border-left: 3px solid #28a745;
    position: relative;
}
.chat-message .message {
    width: 100%;
    padding: 0;
    margin: 0;
    line-height: 1.5;
}
.chat-message .timestamp {
    font-size: 0.75rem;
    color: #6c757d;
    margin-top: 0.5rem;
    opacity: 0.8;
}
.copy-button {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    background: none;
    border: none;
    color: #6c757d;
    cursor: pointer;
    font-size: 0.9rem;
    opacity: 0.6;
    transition: opacity 0.2s;
    padding: 0.2rem;
}
.copy-button:hover {
    opacity: 1;
    color: #007bff;
}
.thinking {
    background-color: #111;
    border-left: 3px solid #ffc107;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.thinking-dots {
    display: flex;
    gap: 0.2rem;
}
.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #ffc107;
    animation: pulse 1.4s infinite ease-in-out;
}
.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes pulse {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'thinking' not in st.session_state:
    st.session_state.thinking = False

# Function to display chat messages (grouped: user then assistant, latest first)
def display_messages():
    # Agrupa pares de mensagens (usuário, assistente)
    messages = st.session_state.messages
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user_msg = messages[i]
            assistant_msg = messages[i+1] if i+1 < len(messages) and messages[i+1]["role"] == "assistant" else None
            pairs.append((user_msg, assistant_msg))
            i += 2
        else:
            i += 1
    # Mostra do mais recente para o mais antigo
    for user_msg, assistant_msg in reversed(pairs):
        # Pergunta do usuário
        st.markdown(f'''
        <div class="chat-message user" style="background:#111; color:#fff; word-break:break-word;">
            <div class="message">💬 {user_msg["content"]}</div>
            <div class="timestamp">{user_msg.get("timestamp", "")}</div>
        </div>
        ''', unsafe_allow_html=True)
        # Resposta do assistente
        if assistant_msg:
            st.markdown(f'''
            <div class="chat-message assistant" style="background:#111; color:#fff; word-break:break-word;">
                <button class="copy-button" onclick="navigator.clipboard.writeText(`{assistant_msg["content"].replace("`", "\\`").replace("'", "\\'")}`)">
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="12" height="12" rx="2" stroke="white" stroke-width="1.5" fill="none"/><rect x="6" y="6" width="9" height="9" rx="2" fill="white" fill-opacity="0.1"/></svg>
                </button>
                <div class="message">🤖 {assistant_msg["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)

# Function to simulate thinking with minimalist dots animation
def simulate_thinking():
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown('''
    <div class="thinking">
        <span>Processando</span>
        <div class="thinking-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Function to generate a response by calling the API
def generate_response(prompt):
    # API endpoint
    api_url = "http://localhost:8000/api/v1/monitor"
    
    # Prepare the request payload
    payload = {
        "question": prompt
    }
    
    # Simulate thinking
    st.session_state.thinking = True
    simulate_thinking()
    
    try:
        # Make the API request
        response = requests.post(api_url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            
            return result['raw']
            
        else:
            return f"Erro: Falha na requisição da API com código {response.status_code}"
    except Exception as e:
        return f"Erro: Falha ao conectar com a API. {str(e)}"
    
    finally:
        st.session_state.thinking = False

# Title
st.title("💬 Monitor")

# Sidebar with information
with st.sidebar:
    st.title("Sobre")
    st.markdown("""
    Este é um Monitor virtual para a especialização em processos de soldagem.
    """)

# Chat input - moved to top
with st.container():
    # Create a form for the chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Digite sua pergunta:", key="user_input", height=80, disabled=st.session_state.thinking)
        submit_button = st.form_submit_button("Enviar", disabled=st.session_state.thinking)
        
    # Process the input when the form is submitted
    if submit_button and user_input and not st.session_state.thinking:
        # Get current timestamp
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Generate and display assistant response
        response = generate_response(user_input)
        
        # Add assistant response with timestamp
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M")
        })
        
        # Rerun to update the display with the new messages
        st.rerun()

# Display previous messages - moved to bottom
if st.session_state.messages:
    st.markdown("### Conversa")
    display_messages()