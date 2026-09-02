from django.db import models
from django.conf import settings
from backend.settings import AUTH_USER_MODEL
# Create your models here.
class HistoryChat(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_histories"
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
        ("user", "User",),
        ("assistant", "Assistant"),
    ]
    
    history= models.ForeignKey(
        HistoryChat,
        on_delete= models.CASCADE,
        related_name="messages",                           
    )
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ["created_at"]
    def __str__(self):
        return f"{self.role}: {self.content[:50]}"