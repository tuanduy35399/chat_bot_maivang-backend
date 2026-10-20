from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class HistoryChat(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_histories",
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        default="",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or f"Chat #{self.id}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]
    PROCESSING_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("complete", "Complete"),
        ("failed", "Failed"),
    ]

    history = models.ForeignKey(
        HistoryChat,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField(blank=True, default="")
    has_image = models.BooleanField(default=False)
    detections = models.JSONField(default=list, blank=True)
    processing_status = models.CharField(
        max_length=12,
        choices=PROCESSING_STATUS_CHOICES,
        default="complete",  # Backfill tin nhắn cũ; request mới tạo với pending ở Task 2.
    )
    reply_to = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="assistant_replies",
    )  # Chỉ assistant mới có reply_to trỏ tới user message cùng HistoryChat.
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]

    def clean(self):
        super().clean()
        if self.reply_to is not None:
            if self.role != "assistant":
                raise ValidationError({"reply_to": "Only assistant messages can have reply_to."})
            if self.reply_to.role != "user":
                raise ValidationError({"reply_to": "reply_to must link to a user message."})
            if self.reply_to.history_id != self.history_id:
                raise ValidationError({"reply_to": "reply_to must link to a message in the same chat history."})

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
