# Monitor Virtual

Um agente multimodal de IA construído usando Pixeltable, Flask e vários modelos de linguagem (LLMs). O aplicativo permite fazer perguntas sobre documentos, imagens, vídeos, áudios e dados tabulares usando uma interface web.

## Recursos

- 🤖 **Agente IA Multimodal** - Integre e questione múltiplas fontes de dados
- 📄 **Suporte a Documentos** - PDF, Word, texto e muito mais
- 🖼️ **Análise de Imagens** - Processamento e busca semântica de imagens
- 🎬 **Processamento de Vídeo** - Extraia frames e analise vídeos
- 🔊 **Transcrição de Áudio** - Converta áudio em texto pesquisável
- 📊 **Dados Tabulares** - Trabalhe com planilhas e dados estruturados

## Pré-requisitos

- Python 3.9 ou superior
-至少 8GB de RAM (16GB recomendado)
- Acesso à internet para下载 modelos e APIs

## Instalação

### 1. Clone o repositório

```bash
git clone <repositorio-url>
cd monitor-virtual
```

### 2. Crie um ambiente virtual

```bash
python -m venv env
```

### 3. Ative o ambiente virtual

**Linux/macOS:**
```bash
source env/bin/activate
```

**Windows:**
```bash
env\Scripts\activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Configure o spaCy

Execute o script de configuração do spaCy:

```bash
bash setup_spacy.sh
```

Este script instala o modelo de linguagem necessário para o processamento de texto.

### 6. Configure as variáveis de ambiente

Copie o arquivo de exemplo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione suas chaves de API:

```env
# Required for Core LLM Functionality *
ANTHROPIC_API_KEY=sk-ant-api03-...  # Para raciocínio principal (Claude 3.5 Sonnet)
OPENAI_API_KEY=sk-...             # Para transcrição de áudio (Whisper) & geração de imagem (DALL-E 3)
MISTRAL_API_KEY=...               # Para sugestões de perguntas de acompanhamento (Mistral Small)

# Optional (Enable specific tools by providing keys)
NEWS_API_KEY=...                  # Habilita a ferramenta NewsAPI
# Note: yfinance and DuckDuckGo Search tools do not require API keys.

# --- !!**Authentication Mode (required to run locally)**!! ---
# Set to 'local' to bypass the WorkOS authentication used at agent.pixeltable.com and to leverage a default user.
AUTH_MODE=local
```

#### Obtendo as chaves de API:

- **Anthropic** (Claude): [anthropic.com](https://www.anthropic.com/)
- **OpenAI**: [platform.openai.com](https://platform.openai.com/)
- **Mistral AI**: [mistral.ai](https://mistral.ai/)
- **News API** (opcional): [newsapi.org](https://newsapi.org/)

## Executando o Aplicativo

### Inicie o servidor Flask

```bash
python endpoint.py
```

O servidor será iniciado em `http://localhost:5000`.

### Acesse a interface web

Abra seu navegador e vá para: **http://localhost:5000**

<!-- ## Carregando Dados

Para adicionar seus próprios documentos e dados, use o script `load_sources.py`:

```bash
python load_sources.py --file_path /caminho/para/seus/arquivos/ --table_key document
``` -->

Tipos de dados suportados:
- `document` - PDF, Word, texto
- `image` - Imagens (PNG, JPG, etc.)
- `video` - Vídeos
- `audio` - Arquivos de áudio
- `tabular` - Planilhas (Excel, CSV)

## Estrutura do Projeto

```
monitor-virtual/
├── endpoint.py              # Servidor Flask principal
├── config.py                # Configurações centralizadas
├── functions.py            # Funções auxiliares
├── load_sources.py         # Script para carregar dados
├── setup_pixeltable.py     # Configuração do Pixeltable
├── setup_spacy.sh          # Script de configuração do spaCy
├── requirements.txt        # Dependências Python
├── pyproject.toml          # Configuração do projeto
├── static/                 # Arquivos estáticos (CSS, JS, imagens)
├── templates/              # Templates HTML
└── .env                    # Variáveis de ambiente (não versionado)
```

## Solução de Problemas

### Erro de memória GPU

Se você encontrar erros de memória, o aplicativo está configurado para usar CPU por padrão. Verifique que `FORCE_CPU = True` está configurado em `config.py`.

### Erro ao carregar modelo spaCy

Execute novamente o script de configuração:
```bash
bash setup_spacy.sh
```

### Problemas com dependências

Tente reinstalar as dependências:
```bash
pip install --upgrade -r requirements.txt
```

## Licença

Apache License 2.0 - veja o arquivo LICENSE para detalhes.
