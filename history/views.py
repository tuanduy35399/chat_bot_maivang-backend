import requests

from django.conf import settings
from django.db import transaction
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from .models import HistoryChat, ChatMessage
from .serializers import (
    ChatRequestSerializer,
    HistoryChatListSerializer,
    HistoryChatDetailSerializer,
)

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response


class ChatView(APIView):

    permission_classes = [
        IsAuthenticated
    ]

    def post(self, request, pk):

        serializer = ChatRequestSerializer(
            data=request.data
        )

        serializer.is_valid(
            raise_exception=True
        )

        question = serializer.validated_data["question"]

        # 1. Chỉ lấy history của user hiện tại
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

        # Chỉ gửi các tin nhắn trước câu hỏi hiện tại cho AI.
        messages = list(
            history.messages
            .order_by("-created_at")[:10]
        )

        messages.reverse()

        # Chuyển thành history cho FastAPI
        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        # Gọi FastAPI
        fastapi_url = getattr(
            settings,
            "FASTAPI_CHAT_URL",
            "http://localhost:9000/chat"
        )

        try:
            response = requests.post(
                fastapi_url,
                json={
                    "question": question,
                    "history": chat_history
                },
                timeout=120
            )

            response.raise_for_status()
        except requests.exceptions.RequestException:
            return Response(
                {
                    "detail": "Không thể kết nối tới AI service."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        try:
            data = response.json()
        except ValueError:
            return Response(
                {
                    "detail": "AI service trả về dữ liệu không hợp lệ."
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # Lấy answer
        answer = data.get("answer") if isinstance(data, dict) else None

        if not answer:

            return Response(
                {
                    "detail": "AI service không trả về câu trả lời."
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # Chỉ lưu cặp tin nhắn sau khi AI trả lời thành công.
        with transaction.atomic():
            ChatMessage.objects.create(
                history=history,
                role="user",
                content=question
            )
            ChatMessage.objects.create(
                history=history,
                role="assistant",
                content=answer
            )
            update_fields = ["updated_at"]
            if not history.title:
                history.title = question[:255]
                update_fields.append("title")
            history.save(update_fields=update_fields)

        # 8. Trả về app
        return Response(
            {
                "question": question,
                "answer": answer,
                "history_id": history.id
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