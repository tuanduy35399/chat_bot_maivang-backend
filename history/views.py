import re
import uuid

from django.db import transaction
from django.db.models import Q
from drf_spectacular.utils import extend_schema
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import StreamingHttpResponse

from .models import ChatMessage, HistoryChat
from .serializers import (
    ChatRequestSerializer,
    HistoryChatDetailSerializer,
    HistoryChatListSerializer,
    ImageChatRequestSerializer,
)
from .services.contracts import RagDetection, ServiceFailure
from .services.rag_client import RagClient
from .services.sse import stream_operation
from .services.yolo_client import YoloClient


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_MESSAGE_CHARS = 1200


def _request_id(request):
    candidate = request.headers.get("X-Request-ID", "")
    return candidate if REQUEST_ID_PATTERN.fullmatch(candidate) else str(uuid.uuid4())


def _mark_user_message_failed(user_msg: ChatMessage) -> None:
    user_msg.processing_status = "failed"
    user_msg.has_image = False
    user_msg.detections = []
    user_msg.save(update_fields=["processing_status", "has_image", "detections"])


def _is_non_empty_history_message(message: ChatMessage) -> bool:
    return bool(message.content.strip() or message.has_image or message.detections)


def _serialize_history_message(message: ChatMessage) -> dict:
    content = message.content
    if len(content) > MAX_HISTORY_MESSAGE_CHARS:
        content = content[: MAX_HISTORY_MESSAGE_CHARS - 1] + "…"
    payload = {
        "message_id": message.id,
        "role": message.role,
        "content": content,
    }
    if message.has_image or message.detections:
        payload["has_image"] = bool(message.has_image)
        payload["detections"] = list(message.detections or [])
    return payload


def _history_before_current_user_message(
    history: HistoryChat, current_message_id: int
) -> list[dict]:
    current_msg = history.messages.only("id", "created_at").get(id=current_message_id)
    prior_completed_messages = list(
        history.messages.filter(processing_status="complete")
        .filter(
            Q(created_at__lt=current_msg.created_at)
            | Q(created_at=current_msg.created_at, id__lt=current_msg.id)
        )
        .order_by("-created_at", "-id")[:60]
    )
    prior_completed_messages.reverse()

    prior_user_ids = [msg.id for msg in prior_completed_messages if msg.role == "user"]
    linked_assistants_by_user_id: dict[int, ChatMessage] = {}
    all_linked_user_ids: set[int] = set()
    if prior_user_ids:
        for assistant_msg in history.messages.filter(
            role="assistant",
            reply_to_id__in=prior_user_ids,
        ).order_by("created_at", "id"):
            all_linked_user_ids.add(assistant_msg.reply_to_id)
            if (
                assistant_msg.processing_status == "complete"
                and assistant_msg.reply_to_id not in linked_assistants_by_user_id
            ):
                linked_assistants_by_user_id[assistant_msg.reply_to_id] = assistant_msg

    turn_units: list[list[dict]] = []
    idx = 0
    total = len(prior_completed_messages)
    while idx < total:
        msg = prior_completed_messages[idx]
        if msg.role == "user":
            linked_assistant = linked_assistants_by_user_id.get(msg.id)
            if linked_assistant is not None:
                unit = [
                    _serialize_history_message(item)
                    for item in (msg, linked_assistant)
                    if _is_non_empty_history_message(item)
                ]
                if unit:
                    turn_units.append(unit)
                idx += 1
                continue
            if (
                msg.id not in all_linked_user_ids
                and idx + 1 < total
                and prior_completed_messages[idx + 1].role == "assistant"
                and prior_completed_messages[idx + 1].reply_to_id is None
            ):
                unit = [
                    _serialize_history_message(item)
                    for item in (msg, prior_completed_messages[idx + 1])
                    if _is_non_empty_history_message(item)
                ]
                if unit:
                    turn_units.append(unit)
                idx += 2
                continue
            idx += 1
        elif msg.role == "assistant":
            if msg.reply_to_id is None and _is_non_empty_history_message(msg):
                turn_units.append([_serialize_history_message(msg)])
            idx += 1
        else:
            idx += 1

    selected_units: list[list[dict]] = []
    message_count = 0
    for unit in reversed(turn_units):
        if message_count + len(unit) > MAX_HISTORY_MESSAGES:
            break
        selected_units.append(unit)
        message_count += len(unit)
    selected_units.reverse()
    return [message for unit in selected_units for message in unit]


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


def _stream_response(stream, request_id: str) -> StreamingHttpResponse:
    response = StreamingHttpResponse(stream, content_type="text/event-stream; charset=utf-8")
    response["Cache-Control"] = "no-cache"
    response["Connection"] = "keep-alive"
    response["X-Accel-Buffering"] = "no"
    response["X-Request-ID"] = request_id
    return response


def _stream_failure_payload(failure, request_id: str) -> dict:
    return {
        "type": "error",
        "code": failure.code,
        "message": failure.message,
        "request_id": request_id,
    }


def _stream_exception_payload(error: Exception, request_id: str) -> dict:
    del error
    return {
        "type": "error",
        "code": "internal_error",
        "message": "Không thể xử lý yêu cầu.",
        "request_id": request_id,
    }


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

        user_msg = ChatMessage.objects.create(
            history=history,
            role="user",
            content=question,
            processing_status="pending",
        )
        try:
            request_id = _request_id(request)
            result = RagClient().ask_text(
                question=question,
                history=_history_before_current_user_message(history, user_msg.id),
                request_id=request_id,
            )
            if isinstance(result, ServiceFailure):
                _mark_user_message_failed(user_msg)
                return _failure_response(result)
            with transaction.atomic():
                assistant_msg = ChatMessage(
                    history=history,
                    role="assistant",
                    content=result.answer,
                    processing_status="complete",
                    reply_to=user_msg,
                )
                assistant_msg.full_clean()
                assistant_msg.save()
                user_msg.processing_status = "complete"
                user_msg.save(update_fields=["processing_status"])
                history.save(update_fields=["updated_at"])
        except Exception:
            _mark_user_message_failed(user_msg)
            raise
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

        user_msg = ChatMessage.objects.create(
            history=history,
            role="user",
            content=question,
            processing_status="pending",
        )
        try:
            request_id = _request_id(request)
            image.seek(0)
            yolo_result = YoloClient().predict(
                image_bytes=image.read(),
                filename=image.name,
                content_type=image.content_type,
                request_id=request_id,
            )
            if isinstance(yolo_result, ServiceFailure):
                _mark_user_message_failed(user_msg)
                return _failure_response(yolo_result)
            rag_result = RagClient().ask_image(
                question=question,
                history=_history_before_current_user_message(history, user_msg.id),
                detections=yolo_result.detections,
                request_id=request_id,
            )
            if isinstance(rag_result, ServiceFailure):
                _mark_user_message_failed(user_msg)
                return _failure_response(rag_result)
            detections_payload = _detection_payload(rag_result.detections)
            with transaction.atomic():
                assistant_msg = ChatMessage(
                    history=history,
                    role="assistant",
                    content=rag_result.answer,
                    processing_status="complete",
                    reply_to=user_msg,
                )
                assistant_msg.full_clean()
                assistant_msg.save()
                user_msg.has_image = True
                user_msg.detections = detections_payload
                user_msg.processing_status = "complete"
                user_msg.save(update_fields=["has_image", "detections", "processing_status"])
                history.save(update_fields=["updated_at"])
        except Exception:
            _mark_user_message_failed(user_msg)
            raise
        return Response(
            {
                "question": question,
                "answer": rag_result.answer,
                "history_id": history.id,
                "detections": detections_payload,
            },
            status=status.HTTP_200_OK,
            headers={"X-Request-ID": request_id},
        )


class ChatStreamView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"]
        try:
            history = HistoryChat.objects.get(id=pk, user=request.user)
        except HistoryChat.DoesNotExist:
            return Response({"detail": "Không tìm thấy cuộc trò chuyện."}, status=status.HTTP_404_NOT_FOUND)

        request_id = _request_id(request)
        user_msg = ChatMessage.objects.create(
            history=history,
            role="user",
            content=question,
            processing_status="pending",
        )

        def operation():
            try:
                result = RagClient().ask_text(
                    question=question,
                    history=_history_before_current_user_message(history, user_msg.id),
                    request_id=request_id,
                )
                if isinstance(result, ServiceFailure):
                    _mark_user_message_failed(user_msg)
                    return _stream_failure_payload(result, request_id)
                with transaction.atomic():
                    assistant_msg = ChatMessage(
                        history=history,
                        role="assistant",
                        content=result.answer,
                        processing_status="complete",
                        reply_to=user_msg,
                    )
                    assistant_msg.full_clean()
                    assistant_msg.save()
                    user_msg.processing_status = "complete"
                    user_msg.save(update_fields=["processing_status"])
                    history.save(update_fields=["updated_at"])
                return {
                    "type": "complete",
                    "answer": result.answer,
                    "history_id": history.id,
                    "detections": [],
                    "request_id": request_id,
                }
            except Exception as error:
                _mark_user_message_failed(user_msg)
                return _stream_exception_payload(error, request_id)

        return _stream_response(
            stream_operation(
                operation,
                progress_payloads=(
                    {
                        "type": "progress",
                        "stage": "preparing",
                        "message": "Đang chuẩn bị câu hỏi...",
                        "request_id": request_id,
                    },
                    {
                        "type": "progress",
                        "stage": "processing",
                        "message": "MaiCare đang phân tích thông tin...",
                        "request_id": request_id,
                    },
                ),
                error_payload=lambda error: _stream_exception_payload(error, request_id),
            ),
            request_id,
        )


class ImageChatStreamView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, pk):
        serializer = ImageChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data["image"]
        question = serializer.validated_data.get("question", "").strip()
        try:
            history = HistoryChat.objects.get(id=pk, user=request.user)
        except HistoryChat.DoesNotExist:
            return Response({"detail": "Không tìm thấy cuộc trò chuyện."}, status=status.HTTP_404_NOT_FOUND)

        request_id = _request_id(request)
        image.seek(0)
        image_bytes = image.read()
        filename = image.name
        content_type = image.content_type
        user_msg = ChatMessage.objects.create(
            history=history,
            role="user",
            content=question,
            processing_status="pending",
        )

        def operation():
            try:
                yolo_result = YoloClient().predict(
                    image_bytes=image_bytes,
                    filename=filename,
                    content_type=content_type,
                    request_id=request_id,
                )
                if isinstance(yolo_result, ServiceFailure):
                    _mark_user_message_failed(user_msg)
                    return _stream_failure_payload(yolo_result, request_id)
                rag_result = RagClient().ask_image(
                    question=question,
                    history=_history_before_current_user_message(history, user_msg.id),
                    detections=yolo_result.detections,
                    request_id=request_id,
                )
                if isinstance(rag_result, ServiceFailure):
                    _mark_user_message_failed(user_msg)
                    return _stream_failure_payload(rag_result, request_id)
                detections_payload = _detection_payload(rag_result.detections)
                with transaction.atomic():
                    assistant_msg = ChatMessage(
                        history=history,
                        role="assistant",
                        content=rag_result.answer,
                        processing_status="complete",
                        reply_to=user_msg,
                    )
                    assistant_msg.full_clean()
                    assistant_msg.save()
                    user_msg.has_image = True
                    user_msg.detections = detections_payload
                    user_msg.processing_status = "complete"
                    user_msg.save(update_fields=["has_image", "detections", "processing_status"])
                    history.save(update_fields=["updated_at"])
                return {
                    "type": "complete",
                    "question": question,
                    "answer": rag_result.answer,
                    "history_id": history.id,
                    "detections": detections_payload,
                    "request_id": request_id,
                }
            except Exception as error:
                _mark_user_message_failed(user_msg)
                return _stream_exception_payload(error, request_id)

        return _stream_response(
            stream_operation(
                operation,
                progress_payloads=(
                    {
                        "type": "progress",
                        "stage": "preparing",
                        "message": "Đang chuẩn bị ảnh...",
                        "request_id": request_id,
                    },
                    {
                        "type": "progress",
                        "stage": "processing",
                        "message": "MaiCare đang phân tích hình ảnh...",
                        "request_id": request_id,
                    },
                ),
                error_payload=lambda error: _stream_exception_payload(error, request_id),
            ),
            request_id,
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
