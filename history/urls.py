from django.urls import path

from .views import (
    HistoryListCreateView,
    HistoryDetailView,
    ChatView,
    ImageChatView,
    ChatStreamView,
    ImageChatStreamView,
)


urlpatterns = [

    path(
        "",
        HistoryListCreateView.as_view(),
        name="history-list-create"
    ),

    path(
        "<int:pk>/",
        HistoryDetailView.as_view(),
        name="history-detail"
    ),

    path(
        "<int:pk>/chat/",
        ChatView.as_view(),
        name="history-chat"
    ),

    path(
        "<int:pk>/chat/stream/",
        ChatStreamView.as_view(),
        name="history-chat-stream"
    ),

    path(
        "<int:pk>/chat/image/",
        ImageChatView.as_view(),
        name="history-image-chat"
    ),

    path(
        "<int:pk>/chat/image/stream/",
        ImageChatStreamView.as_view(),
        name="history-image-chat-stream"
    ),
]
