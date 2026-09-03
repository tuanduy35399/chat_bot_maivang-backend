from django.urls import path

from .views import (
    HistoryListCreateView,
    HistoryDetailView,
    ChatView,
    ImageChatView,
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
        "<int:pk>/chat/image/",
        ImageChatView.as_view(),
        name="history-image-chat"
    ),
]