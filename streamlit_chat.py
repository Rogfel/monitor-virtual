import streamlit as st
import time
import random
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Monitor",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for chat-like interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
    border: 1px solid #4c5c7a;
}
.chat-message.assistant {
    background-color: #475063;
    border: 1px solid #5a6c8d;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 100%;
    padding: 0 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'thinking' not in st.session_state:
    st.session_state.thinking = False

# Function to display chat messages
def display_messages():
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user"><div class="message">🧑‍💻: {message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                # For assistant messages, we need to ensure HTML formatting is preserved
                st.markdown(f'<div class="chat-message assistant"><div class="message">🤖: {message["content"]}</div></div>', unsafe_allow_html=True)

# Function to simulate thinking with dots animation
def simulate_thinking():
    thinking_placeholder = st.empty()
    dots = ""
    for i in range(3):
        dots += "."
        thinking_placeholder.markdown(f'<div class="chat-message assistant"><div class="message">🤖: Thinking{dots}</div></div>', unsafe_allow_html=True)
        time.sleep(0.5)
    thinking_placeholder.empty()

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
            return f"Error: API request failed with status code {response.status_code}"
    except Exception as e:
        return f"Error: Failed to connect to the API. {str(e)}"
    
    st.session_state.thinking = False

# Title
st.title("💬 Monitor")

# Sidebar with information
with st.sidebar:
    st.title("Sobre")
    st.markdown("""
    Este é um Monitor virtual para a especialização em processos de soldagem.
    """)

# Display previous messages
display_messages()

# Chat input
with st.container():
    # Create a form for the chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Digite sua pergunta:", key="user_input", height=100)
        submit_button = st.form_submit_button("Enviar")
        
    # Process the input when the form is submitted
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display updated chat
        display_messages()
        
        # Generate and display assistant response
        response = generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display with the new messages
        st.rerun()