from django.contrib import admin
from history.models import ChatMessage, HistoryChat

# Register your models here.
class HistoryChatAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "user",
        "title",
        "created_at",
        "updated_at",
    ]

    list_filter = [
        "created_at",
        "updated_at",
    ]

    search_fields = [
        "title",
        "user__username",
        "user__email",
    ]

    ordering = [
        "-updated_at",
    ]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "history",
        "role",
        "content_preview",
        "created_at",
    ]

    list_filter = [
        "role",
        "created_at",
    ]

    search_fields = [
        "content",
        "history__title",
        "history__user__username",
    ]

    ordering = [
        "-created_at",
    ]

    @admin.display(description="Content")
    def content_preview(self, obj):
        return obj.content[:100]
    