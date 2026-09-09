from __future__ import annotations

import math
from typing import Any

import requests
from django.conf import settings

from .contracts import ServiceFailure, YoloDetection, YoloSuccess


class YoloClient:
    def __init__(self, *, url=None, api_key=None, timeout=None) -> None:
        self.url = settings.YOLO_API_URL if url is None else url
        self.api_key = settings.YOLO_API_KEY if api_key is None else api_key
        self.timeout = settings.YOLO_API_TIMEOUT if timeout is None else timeout

    def predict(self, *, image_bytes, filename, content_type, request_id):
        headers = {"X-Request-ID": request_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = requests.post(
                self.url,
                headers=headers,
                data={
                    "conf": settings.YOLO_CONFIDENCE_THRESHOLD,
                    "iou": settings.YOLO_IOU_THRESHOLD,
                    "imgsz": settings.YOLO_IMAGE_SIZE,
                },
                files={"file": (filename, image_bytes, content_type)},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return ServiceFailure("yolo_timeout", "YOLO service phản hồi quá thời gian cho phép.", 504, request_id)
        except requests.exceptions.RequestException:
            return ServiceFailure("yolo_unavailable", "Không thể kết nối tới YOLO service.", 503, request_id)
        try:
            return YoloSuccess(self._parse_detections(response.json()), request_id)
        except (TypeError, ValueError, KeyError, OverflowError):
            return ServiceFailure("yolo_invalid_response", "YOLO service trả về dữ liệu không hợp lệ.", 502, request_id)

    @staticmethod
    def _parse_detections(payload: Any) -> tuple[YoloDetection, ...]:
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            raise ValueError("invalid YOLO payload")
        detections = []
        for image in payload["images"]:
            if not isinstance(image, dict) or not isinstance(image.get("results"), list):
                raise ValueError("invalid YOLO image result")
            for result in image["results"]:
                if not isinstance(result, dict):
                    raise ValueError("invalid YOLO detection")
                bbox = result.get("bbox_xyxy", result.get("bbox", result.get("box")))
                if isinstance(bbox, dict):
                    bbox = [bbox[key] for key in ("x1", "y1", "x2", "y2")]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError("invalid YOLO bbox")
                name = result.get("name")
                class_id = result.get("class_id", result.get("class"))
                if not isinstance(name, str):
                    raise ValueError("invalid YOLO name")
                confidence = float(result["confidence"])
                coordinates = tuple(float(value) for value in bbox)
                if not math.isfinite(confidence) or not all(
                    math.isfinite(value) for value in coordinates
                ):
                    raise ValueError("non-finite YOLO numeric value")
                detections.append(YoloDetection(
                    class_id=int(class_id), name=name,
                    confidence=confidence,
                    bbox_xyxy=coordinates,
                ))
        return tuple(detections)
