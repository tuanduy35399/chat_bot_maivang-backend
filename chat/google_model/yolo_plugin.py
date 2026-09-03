"""YOLO ONNX integration point.

Place the exported model in the project later and set YOLO_MODEL_PATH.
The adapter deliberately keeps model-specific post-processing isolated here.
"""

import io
import os
from typing import Any


class YoloOnnxPlugin:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "")
        self._session = None

    def _load(self):
        if not self.model_path:
            raise RuntimeError("YOLO_MODEL_PATH chưa được cấu hình.")
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"Không tìm thấy YOLO ONNX model: {self.model_path}")
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Thiếu dependency onnxruntime cho YOLO plugin.") from exc
            self._session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """Run inference and return normalized detections.

        Model-specific image preprocessing and output decoding belong here.
        The default implementation validates that the plugin is configured and
        intentionally raises until the model's input/output contract is known.
        """
        self._load()
        try:
            from PIL import Image
            Image.open(io.BytesIO(image_bytes)).verify()
        except ImportError as exc:
            raise RuntimeError("Thiếu dependency Pillow cho YOLO plugin.") from exc
        except Exception as exc:
            raise ValueError("File tải lên không phải ảnh hợp lệ.") from exc

        # TODO: implement the preprocessing/output decoder for the supplied
        # YOLO export. Keeping this boundary here allows replacing the model
        # without changing FastAPI or RAG code.
        raise NotImplementedError(
            "Cần cấu hình preprocessing và output decoder theo model YOLO ONNX."
        )


_detector = YoloOnnxPlugin()


def detect(image_bytes: bytes) -> list[dict[str, Any]]:
    return _detector.predict(image_bytes)


def format_detections(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "YOLO không phát hiện dấu hiệu nào đáng chú ý."
    return "\n".join(
        f"- {item.get('class_name', item.get('class_id', 'unknown'))}: "
        f"confidence={item.get('confidence', 0):.3f}"
        for item in detections
    )
