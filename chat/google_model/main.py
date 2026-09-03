import json
from typing import List
from pydantic import Field

from fastapi import File, FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .rag import ask
from .yolo_plugin import detect, format_detections


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


@app.post("/chat/image")
async def image_chat(
    image: UploadFile = File(...),
    question: str = Form(""),
    history: str = Form("[]"),
):
    image_bytes = await image.read()
    try:
        detections = detect(image_bytes)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        chat_history = json.loads(history)
        if not isinstance(chat_history, list):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="history không hợp lệ.") from exc

    detection_text = format_detections(detections)
    enriched_question = (
        f"Mô tả từ ảnh do YOLO phát hiện:\n{detection_text}\n\n"
        f"Câu hỏi của người dùng: {question or 'Hãy phân tích tình trạng cây mai trong ảnh.'}"
    )
    answer = ask(
        question=enriched_question,
        history=chat_history,
        system_prompt=(
            "Bạn là trợ lý AI chuyên về bệnh trên cây mai vàng. "
            "Hãy kết hợp kết quả nhận diện YOLO với tài liệu RAG; "
            "nếu chưa đủ chắc chắn, hãy nói rõ giới hạn kết luận."
        ),
    )
    return {"answer": answer, "detections": detections}


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