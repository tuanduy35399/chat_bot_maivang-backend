from __future__ import annotations

import json

import requests
from django.conf import settings

from .contracts import RagSuccess, ServiceFailure, YoloDetection


class RagClient:
    def __init__(self, *, text_url=None, image_url=None, timeout=None) -> None:
        self.text_url = settings.FASTAPI_CHAT_URL if text_url is None else text_url
        self.image_url = settings.FASTAPI_IMAGE_CHAT_URL if image_url is None else image_url
        self.timeout = settings.FASTAPI_TIMEOUT if timeout is None else timeout

    def ask_text(self, *, question, history, request_id):
        return self._post(self.text_url, {"question": question, "history": history}, request_id, json_payload=True)

    def ask_image(self, *, question, history, detections: tuple[YoloDetection, ...], request_id):
        wire_detections = [{
            "class_id": detection.class_id,
            "name": detection.name,
            "confidence": detection.confidence,
            "bbox_xyxy": list(detection.bbox_xyxy),
        } for detection in detections]
        return self._post(self.image_url, {
            "question": question,
            "history": json.dumps(history, ensure_ascii=False),
            "detections": json.dumps(wire_detections, ensure_ascii=False),
        }, request_id, json_payload=False)

    def _post(self, url, payload, request_id, *, json_payload):
        try:
            kwargs = {"headers": {"X-Request-ID": request_id}, "timeout": self.timeout}
            kwargs["json" if json_payload else "data"] = payload
            response = requests.post(url, **kwargs)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return ServiceFailure("rag_timeout", "AI service phản hồi quá thời gian cho phép.", 504, request_id)
        except requests.exceptions.RequestException:
            return ServiceFailure("rag_unavailable", "Không thể kết nối tới AI service.", 503, request_id)
        try:
            body = response.json()
            answer = body.get("answer") if isinstance(body, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("missing answer")
        except (TypeError, ValueError):
            return ServiceFailure("rag_invalid_response", "AI service trả về dữ liệu không hợp lệ.", 502, request_id)
        return RagSuccess(answer, request_id)
