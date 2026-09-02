from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from .models import HistoryChat, ChatMessage
from .serializers import (
    HistoryChatListSerializer,
    HistoryChatDetailSerializer,
    ChatMessageSerializer
)

from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response


class ChatView(APIView):

    permission_classes = [
        IsAuthenticated
    ]

    def post(self, request, pk):

        serializer = ChatMessageSerializer(
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

        # 2. Lưu câu hỏi của user
        ChatMessage.objects.create(
            history=history,
            role="user",
            content=question
        )

        # 3. Lấy 10 message gần nhất
        messages = list(
            history.messages
            .order_by("-created_at")[:10]
        )

        messages.reverse()

        # 4. Chuyển thành history cho FastAPI
        chat_history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        # 5. Gọi FastAPI
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

            data = response.json()

        except requests.exceptions.RequestException:

            return Response(
                {
                    "detail": "Không thể kết nối tới AI service."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # 6. Lấy answer
        answer = data.get("answer")

        if not answer:

            return Response(
                {
                    "detail": "AI service không trả về câu trả lời."
                },
                status=status.HTTP_502_BAD_GATEWAY
            )

        # 7. Lưu câu trả lời
        ChatMessage.objects.create(
            history=history,
            role="assistant",
            content=answer
        )

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