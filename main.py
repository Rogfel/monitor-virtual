import os
import logging
from typing import Dict
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel

from rag_loading import search
from agent import CrewAgent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Config AI agent
agent = CrewAgent()
agent.new_agent(agent_name='monitor_agent')
agent.new_task('chat_task', agent=agent.agents_dict['monitor_agent'])


# Initialize FastAPI app
app = FastAPI(
    title="Welding monitor API",
    description="API for welding monitor",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str

@app.post("/api/v1/monitor", response_model=Dict)
async def chat(QRequest: QuestionRequest):
    response = search(queries=QRequest.question, top_k=15)

    documents = [doc['document'] for doc in response["results"][0]["matches"]]
    scores = [doc['similarity'] for doc in response["results"][0]["matches"]]

    result = agent.crew().kickoff(inputs={"question": QRequest.question, "know-how": documents, "scores": scores})
    
    return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result))