import os
from typing import Any

import requests
from dotenv import load_dotenv



class YoloService:

    def predict(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
    ) -> dict[str, Any]:

        if not self.api_url:
            raise RuntimeError("YOLO_API_URL chưa được cấu hình.")

        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "conf": 0.25,
            "iou": 0.7,
            "imgsz": 640,
        }

        files = {
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Không thể kết nối YOLO API: {exc}"
            ) from exc

        if not response.ok:
            raise RuntimeError(
                f"YOLO API lỗi HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:
            return response.json()

        except ValueError as exc:
            raise RuntimeError(
                f"YOLO API không trả về JSON hợp lệ: "
                f"{response.text}"
            ) from exc


yolo_service = YoloService()