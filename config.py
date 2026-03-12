# config.py - Centralized configuration for the application

# --- LLM & Model IDs ---
# Embedding model for text-based semantic search (documents, transcripts, memory, history)
# Using a smaller model to avoid GPU memory issues
# EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
# EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Vision-language model for image/frame semantic search
# CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# Audio transcription model
WHISPER_MODEL_ID = "whisper-1"
# Image generation model
DALLE_MODEL_ID = "dall-e-3"
# Main reasoning LLM (tool use, final answer)
CLAUDE_MODEL_ID = "claude-sonnet-4-20250514"
# Follow-up question LLM
# MISTRAL_MODEL_ID = "mistral-medium-latest"
MISTRAL_MODEL_ID = "mistral-small-latest"

# --- Hardware Configuration ---
# Force CPU usage for models to avoid GPU memory issues
FORCE_CPU = False

# Set environment variable to force CPU usage for PyTorch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# --- Default System Prompts ---
# Initial prompt for tool selection/analysis
INITIAL_SYSTEM_PROMPT = """Identifique a(s) melhor(es) ferramenta(s) para responder à consulta do usuário com base nas fontes de dados disponíveis (documentos, imagens, vídeos, áudios, tabelas)."""
# Final prompt for synthesizing the answer
FINAL_SYSTEM_PROMPT = """Com base no contexto fornecido e na consulta do usuário, forneça uma resposta completa e informativa. É OBRIGATÓRIO incluir referências dos documentos utilizados no formato [nome_do_documento: página] ou [fonte: página] para cada informação citada. Se a informação vier de múltiplas fontes, cite todas elas. Se não houver informações relevantes nos documentos fornecidos, indique claramente isso."""

# --- Default LLM Parameters ---
# Set to None to use the underlying API defaults
DEFAULT_MAX_TOKENS: int | None = 2000
DEFAULT_STOP_SEQUENCES: list[str] | None = None  # e.g., ["\n\nHuman:"]
DEFAULT_TEMPERATURE: float | None = 0.3
DEFAULT_TOP_P: float | None = 0.7
DEFAULT_TOP_K: int = 0  # 0 disables top-k sampling (use None only if you want to omit this parameter)

# --- Consolidated Parameters Dictionary ---
DEFAULT_PARAMETERS = {
    "max_tokens": DEFAULT_MAX_TOKENS,
    "stop_sequences": DEFAULT_STOP_SEQUENCES,
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": DEFAULT_TOP_P,
    "top_k": DEFAULT_TOP_K,
}

# --- Persona Presets Definition --- #
# These will be added to a new user's persona table on first login.
PERSONA_PRESETS = {
    "Assistente Pessoal": {
        "initial_prompt": "Você é um assistente pessoal útil e amigável. Use as ferramentas disponíveis e o contexto para responder às perguntas do usuário de forma clara e concisa.",
        "final_prompt": "Forneça uma resposta clara, amigável e concisa baseada nas informações coletadas e na consulta do usuário. Inclua referências dos documentos utilizados no formato [fonte: página] para cada informação citada.",
        "llm_params": {
            "max_tokens": 1500,
            "stop_sequences": None,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 0,  # 0 disables top-k sampling
        }
    },
    "Analista Visual": {
        "initial_prompt": "Você é um analista visual especialista. Priorize informações extraídas de imagens e quadros de vídeo para responder à consulta do usuário. Descreva elementos visuais e padrões com precisão.",
        "final_prompt": "Com base principalmente no contexto visual fornecido (imagens, quadros de vídeo), gere uma análise detalhada respondendo à consulta do usuário. Mencione detalhes visuais específicos observados. Inclua referências dos documentos utilizados no formato [fonte: página] para cada informação citada.",
        "llm_params": {
            "max_tokens": 2000,
            "stop_sequences": None,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 0,
        }
    },
    "Assistente de Pesquisa": {
        "initial_prompt": "Você é um assistente de pesquisa meticuloso. Sintetize informações de várias fontes (documentos, imagens, vídeos, áudios, tabelas) para construir uma resposta abrangente. Identifique descobertas importantes e cite fontes quando aplicável.",
        "final_prompt": "Compile as descobertas da pesquisa do contexto fornecido em um resumo bem estruturado e informativo que responda diretamente à consulta do usuário. Destaque pontos-chave de dados ou conclusões. É OBRIGATÓRIO incluir referências completas dos documentos utilizados no formato [nome_do_documento: página] para cada informação citada.",
        "llm_params": {
            "max_tokens": 2500,
            "stop_sequences": None,
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 0,
        }
    },
    "Guia Técnico": {
        "initial_prompt": "Você é um guia técnico. Concentre-se em fornecer detalhes técnicos precisos, explicações ou exemplos de código baseados na consulta do usuário e no contexto disponível.",
        "final_prompt": "Gere uma resposta técnica precisa e exata, potencialmente incluindo trechos de código ou instruções passo a passo, baseada na consulta do usuário e nas informações coletadas. Inclua referências dos documentos técnicos utilizados no formato [fonte: página] para cada informação citada.",
        "llm_params": {
            "max_tokens": 2000,
            "stop_sequences": None,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 0,
        }
    }
}

# Pixeltable Table Mapping
TABLE_MAP = {
    "document": "agents.collection",
    "image": "agents.images",
    "video": "agents.videos",
    "audio": "agents.audios",
    "tabular": "agents.tabular",
}
