from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .rag import ask


app = FastAPI(
    title="Mai Vang Chat Service",
    version="1.0.0"
)


class HistoryMessage(BaseModel):

    role: str
    content: str


class ChatRequest(BaseModel):

    question: str
    history: List[HistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):

    answer: str


@app.get("/")
def root():

    return {
        "message": "Mai Vang Chat Service is running"
    }


@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    history = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in request.history
    ]

    answer = ask(
        question=request.question,
        history=history
    )

    return {
        "answer": answer
    }