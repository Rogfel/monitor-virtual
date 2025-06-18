import streamlit as st
import time
import os
from dotenv import load_dotenv
from rag.supabase_rag import SupabaseRAG

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Chat LLM com RAG",
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

if 'rag_system' not in st.session_state:
    try:
        # Initialize the RAG system
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            st.session_state.rag_initialized = False
            st.session_state.rag_error = "Credenciais do Supabase não encontradas. Verifique seu arquivo .env."
        else:
            st.session_state.rag_system = SupabaseRAG(supabase_url, supabase_key)
            st.session_state.rag_initialized = True
    except Exception as e:
        st.session_state.rag_initialized = False
        st.session_state.rag_error = f"Erro ao inicializar o sistema RAG: {str(e)}"

# Function to display chat messages
def display_messages():
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user"><div class="message">🧑‍💻: {message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant"><div class="message">🤖: {message["content"]}</div></div>', unsafe_allow_html=True)

# Function to simulate thinking with dots animation
def simulate_thinking():
    thinking_placeholder = st.empty()
    dots = ""
    for i in range(3):
        dots += "."
        thinking_placeholder.markdown(f'<div class="chat-message assistant"><div class="message">🤖: Pensando{dots}</div></div>', unsafe_allow_html=True)
        time.sleep(0.5)
    thinking_placeholder.empty()

# Function to generate a response using the RAG system
def generate_response(prompt):
    # Simulate thinking
    st.session_state.thinking = True
    simulate_thinking()
    st.session_state.thinking = False
    
    try:
        if st.session_state.rag_initialized:
            # Use the RAG system to search for relevant documents
            results = st.session_state.rag_system.search_documents(prompt)
            
            if results and len(results) > 0:
                # Format the response with the retrieved information
                response = "Com base nos documentos encontrados:\n\n"
                
                for i, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0) * 100
                    response += f"**Documento {i}** (Relevância: {similarity:.1f}%):\n"
                    response += f"{result.get('documento', '')[:300]}...\n\n"
                
                response += f"\nEm resposta à sua pergunta: '{prompt}'\n"
                response += "Os documentos acima contêm informações relevantes que podem ajudar a responder sua questão."
                
                return response
            else:
                return f"Não encontrei informações relevantes sobre '{prompt}' nos documentos disponíveis. Tente reformular sua pergunta ou perguntar sobre outro tópico."
        else:
            return f"O sistema RAG não foi inicializado corretamente. Erro: {st.session_state.get('rag_error', 'Desconhecido')}"
    except Exception as e:
        return f"Ocorreu um erro ao gerar a resposta: {str(e)}"

# Title
st.title("💬 Chat LLM com RAG")

# Sidebar with information
with st.sidebar:
    st.title("Sobre")
    st.markdown("""
    Este é um chat que utiliza RAG (Retrieval-Augmented Generation) para 
    fornecer respostas baseadas nos documentos armazenados no Supabase.
    
    Para usar:
    1. Digite sua pergunta no campo abaixo
    2. Pressione Enter ou clique em "Enviar"
    3. Aguarde a resposta
    
    O sistema buscará informações relevantes nos documentos armazenados
    e apresentará os resultados mais relevantes para sua pergunta.
    """)
    
    # Display RAG system status
    if st.session_state.rag_initialized:
        st.success("Sistema RAG inicializado com sucesso!")
    else:
        st.error(f"Sistema RAG não inicializado. {st.session_state.get('rag_error', '')}")
        if st.button("Tentar inicializar novamente"):
            try:
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_KEY")
                
                if not supabase_url or not supabase_key:
                    st.session_state.rag_error = "Credenciais do Supabase não encontradas. Verifique seu arquivo .env."
                else:
                    st.session_state.rag_system = SupabaseRAG(supabase_url, supabase_key)
                    st.session_state.rag_initialized = True
                st.rerun()
            except Exception as e:
                st.session_state.rag_error = f"Erro ao inicializar o sistema RAG: {str(e)}"

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