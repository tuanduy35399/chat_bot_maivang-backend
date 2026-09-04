import json

from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .models import HistoryChat, ChatMessage
from .serializers import (
    HistoryChatListSerializer,
    HistoryChatDetailSerializer,
    ChatMessageSerializer,
    ChatRequestSerializer,
    ImageChatRequestSerializer,
)

from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema
import requests


class ChatView(APIView):

    permission_classes = [
        IsAuthenticated
    ]

    @extend_schema(
        request=ChatRequestSerializer,
        responses={status.HTTP_200_OK: dict},
        description="Gửi câu hỏi tới AI và lưu câu hỏi/câu trả lời vào lịch sử chat.",
    )
    def post(self, request, pk):

        serializer = ChatRequestSerializer(
            data=request.data
        )

        serializer.is_valid(
            raise_exception=True
        )

        question = serializer.validated_data["question"]

        try:
            history = HistoryChat.objects.get(
                id=pk,
                user=request.user
            )

        except HistoryChat.DoesNotExist:

            return Response(
                {
                    "detail": "Không tìm thấy cuộc trò chuyện."
                },
                status=status.HTTP_404_NOT_FOUND
            )

        #Lưu câu hỏi của user
        ChatMessage.objects.create(
            history=history,
            role="user",
            content=question
        )

        # Lấy 10 message gần nhất
        messages = list(
            history.messages
            .order_by("-created_at")[:10]
        )

        messages.reverse()

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        # Gọi qua bên FastAPI
        fastapi_url = settings.FASTAPI_CHAT_URL

        try:
            response = requests.post(
                fastapi_url,
                json={
                    "question": question,
                    "history": chat_history,
                },
                timeout=settings.FASTAPI_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            return Response(
                {
                    "detail": "AI service phản hồi quá thời gian cho phép."
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except requests.exceptions.RequestException:
            return Response(
                {
                    "detail": "Không thể kết nối tới AI service."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except ValueError:
            return Response(
                {
                    "detail": "AI service trả về dữ liệu không hợp lệ."
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        answer = data.get("answer") if isinstance(data, dict) else None

        if not answer:

            return Response(
                {
                    "detail": "AI service không trả về câu trả lời."
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        ChatMessage.objects.create(
            history=history,
            role="assistant",
            content=answer
        )
        history.save(update_fields=["updated_at"])

        return Response(
            {
                "question": question,
                "answer": answer,
                "history_id": history.id
            },
            status=status.HTTP_200_OK
        )
class ImageChatView(APIView):

    permission_classes = [IsAuthenticated]

    parser_classes = [
        MultiPartParser,
        FormParser
    ]

    @extend_schema(
        request=ImageChatRequestSerializer,
        responses={status.HTTP_200_OK: dict},
        description=(
            "Nhận ảnh từ FE, gửi trực tiếp tới YOLO API, "
            "sau đó gửi kết quả YOLO cùng câu hỏi tới FastAPI RAG."
        ),
    )
    def post(self, request, pk):

        serializer = ImageChatRequestSerializer(
            data=request.data
        )

        serializer.is_valid(
            raise_exception=True
        )

        image = serializer.validated_data["image"]

        question = (
            serializer.validated_data
            .get("question", "")
            .strip()
        )

        if not question:
            question = (
                "Hãy phân tích tình trạng "
                "cây mai trong ảnh."
            )

        # =========================
        # LẤY HISTORY
        # =========================

        try:
            history = HistoryChat.objects.get(
                id=pk,
                user=request.user
            )

        except HistoryChat.DoesNotExist:

            return Response(
                {
                    "detail": "Không tìm thấy cuộc trò chuyện."
                },
                status=status.HTTP_404_NOT_FOUND
            )

        # =========================
        # LƯU MESSAGE USER
        # =========================

        ChatMessage.objects.create(
            history=history,
            role="user",
            content=question
        )

        # =========================
        # LẤY 10 MESSAGE GẦN NHẤT
        # =========================

        messages = list(
            history.messages
            .order_by("-created_at")[:10]
        )

        messages.reverse()

        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        # =========================
        # ĐỌC IMAGE
        # =========================

        print("========== IMAGE CHAT ==========")

        print("1. Đã nhận image:", image.name)

        image.seek(0)
        image_bytes = image.read()

        print(
            "2. Đã đọc image:",
            len(image_bytes),
            "bytes"
        )

        # =========================
        # GỌI YOLO
        # =========================

        print("3. Chuẩn bị gọi YOLO")
        print("YOLO URL:", settings.YOLO_API_URL)

        try:

            headers = {}

            if settings.YOLO_API_KEY:
                headers["Authorization"] = (
                    f"Bearer {settings.YOLO_API_KEY}"
                )

            yolo_response = requests.post(
                settings.YOLO_API_URL,

                headers=headers,

                data={
                    "conf": 0.25,
                    "iou": 0.7,
                    "imgsz": 640,
                },

                files={
                    "file": (
                        image.name,
                        image_bytes,
                        image.content_type,
                    )
                },

                timeout=settings.YOLO_API_TIMEOUT,
            )

            print(
                "4. YOLO status:",
                yolo_response.status_code
            )

            print(
                "5. YOLO response:",
                yolo_response.text
            )

            yolo_response.raise_for_status()

            yolo_result = yolo_response.json()

        except requests.exceptions.Timeout:

            return Response(
                {
                    "detail": (
                        "YOLO service phản hồi "
                        "quá thời gian cho phép."
                    )
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT
            )

        except requests.exceptions.RequestException as exc:

            return Response(
                {
                    "detail": (
                        "Không thể kết nối tới "
                        "YOLO service."
                    ),
                    "error": str(exc),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        except ValueError:

            return Response(
                {
                    "detail": (
                        "YOLO service trả về "
                        "dữ liệu không hợp lệ."
                    )
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # =========================
        # EXTRACT DETECTIONS
        # =========================

        detections = [
            {
                "name": result["name"],
                "confidence": result["confidence"],
            }
            for image_data in yolo_result.get("images", [])
            for result in image_data.get("results", [])
        ]

        print(
            "6. Detections:",
            detections
        )

        # =========================
        # GỌI FASTAPI RAG
        # =========================

        try:

            fastapi_response = requests.post(

                settings.FASTAPI_IMAGE_CHAT_URL,

                data={
                    "question": question,

                    "history": json.dumps(
                        chat_history,
                        ensure_ascii=False
                    ),

                    "detections": json.dumps(
                        detections,
                        ensure_ascii=False
                    ),
                },

                timeout=settings.FASTAPI_TIMEOUT,
            )

            print(
                "7. FastAPI status:",
                fastapi_response.status_code
            )

            print(
                "8. FastAPI response:",
                fastapi_response.text
            )

            fastapi_response.raise_for_status()

            data = fastapi_response.json()

        except requests.exceptions.Timeout:

            return Response(
                {
                    "detail": (
                        "AI service phản hồi "
                        "quá thời gian cho phép."
                    )
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT
            )

        except requests.exceptions.RequestException as exc:

            return Response(
                {
                    "detail": (
                        "Không thể kết nối tới "
                        "AI service."
                    ),
                    "error": str(exc),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        except ValueError:

            return Response(
                {
                    "detail": (
                        "AI service trả về "
                        "dữ liệu không hợp lệ."
                    )
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # =========================
        # LẤY ANSWER
        # =========================

        answer = (
            data.get("answer")
            if isinstance(data, dict)
            else None
        )

        if not answer:

            return Response(
                {
                    "detail": (
                        "AI service không trả về "
                        "câu trả lời."
                    )
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # =========================
        # LƯU ANSWER
        # =========================

        ChatMessage.objects.create(
            history=history,
            role="assistant",
            content=answer
        )

        history.save(
            update_fields=["updated_at"]
        )

        # =========================
        # RESPONSE
        # =========================

        return Response(
            {
                "question": question,
                "answer": answer,
                "history_id": history.id,
                "detections": detections,
            },
            status=status.HTTP_200_OK
        )

class HistoryListCreateView(generics.ListCreateAPIView):

    permission_classes = [
        IsAuthenticated
    ]

    def get_queryset(self):
        return HistoryChat.objects.filter(
            user=self.request.user
        ).order_by("-updated_at")

    def get_serializer_class(self):
        return HistoryChatListSerializer

    def perform_create(self, serializer):
        serializer.save(
            user=self.request.user
        )
class HistoryDetailView(generics.RetrieveUpdateDestroyAPIView):

    permission_classes = [
        IsAuthenticated
    ]

    serializer_class = HistoryChatDetailSerializer

    def get_queryset(self):
        return HistoryChat.objects.filter(
            user=self.request.user
        )