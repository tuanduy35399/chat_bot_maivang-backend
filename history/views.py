import json

from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

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

    @extend_schema(
        request=ImageChatRequestSerializer,
        responses={status.HTTP_200_OK: dict},
        description="Gửi ảnh (và tùy chọn câu hỏi) qua YOLO rồi RAG.",
    )
    def post(self, request, pk):
        serializer = ImageChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            history = HistoryChat.objects.get(id=pk, user=request.user)
        except HistoryChat.DoesNotExist:
            return Response(
                {"detail": "Không tìm thấy cuộc trò chuyện."},
                status=status.HTTP_404_NOT_FOUND,
            )

        image = serializer.validated_data["image"]
        question = serializer.validated_data.get("question", "").strip()
        question = question or "Hãy phân tích tình trạng cây mai trong ảnh."

        ChatMessage.objects.create(history=history, role="user", content=question)
        messages = list(history.messages.order_by("-created_at")[:10])
        messages.reverse()
        chat_history = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        try:
            response = requests.post(
                settings.FASTAPI_IMAGE_CHAT_URL,
                files={"image": (image.name, image.file, image.content_type)},
                data={"question": question, "history": json.dumps(chat_history)},
                timeout=settings.FASTAPI_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            return Response(
                {"detail": "AI service phản hồi quá thời gian cho phép."},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except requests.exceptions.RequestException:
            return Response(
                {"detail": "Không thể kết nối tới AI service."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except ValueError:
            return Response(
                {"detail": "AI service trả về dữ liệu không hợp lệ."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        answer = data.get("answer") if isinstance(data, dict) else None
        if not answer:
            return Response(
                {"detail": "AI service không trả về câu trả lời."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        ChatMessage.objects.create(history=history, role="assistant", content=answer)
        history.save(update_fields=["updated_at"])
        return Response(
            {"question": question, "answer": answer, "history_id": history.id,
             "detections": data.get("detections", [])},
            status=status.HTTP_200_OK,
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