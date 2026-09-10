import re
import uuid

from drf_spectacular.utils import extend_schema
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatMessage, HistoryChat
from .serializers import (
    ChatRequestSerializer,
    HistoryChatDetailSerializer,
    HistoryChatListSerializer,
    ImageChatRequestSerializer,
)
from .services.contracts import RagDetection, ServiceFailure, YoloDetection
from .services.rag_client import RagClient
from .services.yolo_client import YoloClient


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _request_id(request):
    candidate = request.headers.get("X-Request-ID", "")
    return candidate if REQUEST_ID_PATTERN.fullmatch(candidate) else str(uuid.uuid4())


def _history_before_current_user_message(history, question):
    messages = list(history.messages.order_by("-created_at")[:10])
    messages.reverse()
    if messages and messages[-1].role == "user" and messages[-1].content == question:
        messages.pop()
    return [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.content.strip()
    ]


def _failure_response(failure):
    return Response(
        {"detail": failure.message, "code": failure.code, "request_id": failure.request_id},
        status=failure.status_code,
        headers={"X-Request-ID": failure.request_id},
    )


def _detection_payload(detections: tuple[RagDetection, ...]):
    return [{
        "class_id": detection.class_id,
        "name": detection.name,
        "disease_name": detection.disease_name,
        "confidence": detection.confidence,
        "bbox_xyxy": list(detection.bbox_xyxy),
    } for detection in detections]


class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        request=ChatRequestSerializer,
        responses={status.HTTP_200_OK: dict},
        description="Gửi câu hỏi tới AI và lưu câu hỏi/câu trả lời vào lịch sử chat.",
    )
    def post(self, request, pk):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]
        try:
            history = HistoryChat.objects.get(id=pk, user=request.user)
        except HistoryChat.DoesNotExist:
            return Response({"detail": "Không tìm thấy cuộc trò chuyện."}, status=status.HTTP_404_NOT_FOUND)

        ChatMessage.objects.create(history=history, role="user", content=question)
        request_id = _request_id(request)
        result = RagClient().ask_text(
            question=question,
            history=_history_before_current_user_message(history, question),
            request_id=request_id,
        )
        if isinstance(result, ServiceFailure):
            return _failure_response(result)
        ChatMessage.objects.create(history=history, role="assistant", content=result.answer)
        history.save(update_fields=["updated_at"])
        return Response(
            {
                "question": question,
                "answer": result.answer,
                "history_id": history.id,
                "detections": [],
            },
            status=status.HTTP_200_OK,
            headers={"X-Request-ID": request_id},
        )


class ImageChatView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        request=ImageChatRequestSerializer,
        responses={status.HTTP_200_OK: dict},
        description="Nhận ảnh, chuyển raw YOLO detections tới FastAPI RAG và lưu lịch sử chat.",
    )
    def post(self, request, pk):
        serializer = ImageChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data["image"]
        question = serializer.validated_data.get("question", "").strip()
        try:
            history = HistoryChat.objects.get(id=pk, user=request.user)
        except HistoryChat.DoesNotExist:
            return Response({"detail": "Không tìm thấy cuộc trò chuyện."}, status=status.HTTP_404_NOT_FOUND)

        ChatMessage.objects.create(history=history, role="user", content=question)
        request_id = _request_id(request)
        image.seek(0)
        yolo_result = YoloClient().predict(
            image_bytes=image.read(),
            filename=image.name,
            content_type=image.content_type,
            request_id=request_id,
        )
        if isinstance(yolo_result, ServiceFailure):
            return _failure_response(yolo_result)
        rag_result = RagClient().ask_image(
            question=question,
            history=_history_before_current_user_message(history, question),
            detections=yolo_result.detections,
            request_id=request_id,
        )
        if isinstance(rag_result, ServiceFailure):
            return _failure_response(rag_result)
        ChatMessage.objects.create(history=history, role="assistant", content=rag_result.answer)
        history.save(update_fields=["updated_at"])
        return Response(
            {
                "question": question,
                "answer": rag_result.answer,
                "history_id": history.id,
                "detections": _detection_payload(rag_result.detections),
            },
            status=status.HTTP_200_OK,
            headers={"X-Request-ID": request_id},
        )


class HistoryListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return HistoryChat.objects.filter(user=self.request.user).order_by("-updated_at")

    def get_serializer_class(self):
        return HistoryChatListSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class HistoryDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = HistoryChatDetailSerializer

    def get_queryset(self):
        return HistoryChat.objects.filter(user=self.request.user)
