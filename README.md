# Monitor Virtual

A sophisticated AI-powered monitoring system that combines Retrieval-Augmented Generation (RAG) with a FastAPI backend and Streamlit frontend for intelligent document processing and querying.

## Features

- **RAG System**: Implements a Retrieval-Augmented Generation system using Supabase as the vector database
- **FastAPI Backend**: RESTful API endpoints for document processing and querying
- **Streamlit Interface**: User-friendly web interface for interacting with the system
- **AI Agent Integration**: Utilizes CrewAI for intelligent task processing
- **Document Processing**: Efficient document embedding and semantic search capabilities

## Prerequisites

- Python 3.7+
- Supabase account and project
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/monitor-virtual.git
cd monitor-virtual
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables in a `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Project Structure

- `main.py`: FastAPI backend implementation
- `streamlit_chat.py`: Streamlit chat interface
- `streamlit_chat_with_rag.py`: RAG-enhanced Streamlit interface
- `rag_loading.py`: RAG system implementation
- `agent.py`: AI agent implementation using CrewAI
- `config/`: Configuration files
- `data/`: Data storage directory
- `rag/`: RAG-related utilities

## Usage

### Running the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Running the Streamlit Interface

```bash
streamlit run streamlit_chat.py
```

Or with RAG enhancement:
```bash
streamlit run streamlit_chat_with_rag.py
```

## API Endpoints

- `POST /api/v1/monitor`: Process queries and return AI-generated responses
  - Request body: `{"question": "your question here"}`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Supabase for vector database capabilities
- CrewAI for AI agent implementation
- FastAPI for the backend framework
- Streamlit for the frontend interface