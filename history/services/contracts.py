from dataclasses import dataclass


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloSuccess:
    detections: tuple[YoloDetection, ...]
    request_id: str


@dataclass(frozen=True)
class RagSuccess:
    answer: str
    request_id: str


@dataclass(frozen=True)
class ServiceFailure:
    code: str
    message: str
    status_code: int
    request_id: str
