from rest_framework import serializers
from .models import HistoryChat, ChatMessage


class ChatRequestSerializer(serializers.Serializer):

    question = serializers.CharField(
        required=True
    )


class ImageChatRequestSerializer(serializers.Serializer):

    image = serializers.ImageField(required=True)
    question = serializers.CharField(required=False, allow_blank=True)


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "role",
            "content",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "created_at",
        ]


class HistoryChatListSerializer(serializers.ModelSerializer):

    class Meta:
        model = HistoryChat

        fields = [
            "id",
            "title",
            "created_at",
            "updated_at",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]

class HistoryChatDetailSerializer(serializers.ModelSerializer):

    messages = ChatMessageSerializer(
        many=True,
        read_only=True
    )

    class Meta:
        model = HistoryChat

        fields = [
            "id",
            "title",
            "created_at",
            "updated_at",
            "messages",
        ]

        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "messages",
        ]