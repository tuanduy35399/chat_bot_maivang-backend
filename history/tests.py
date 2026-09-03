from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat


class ChatViewTests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="tester",
            email="tester@example.com",
            name="Tester",
            password="password123",
        )
        self.history = HistoryChat.objects.create(
            user=self.user,
            title="",
        )
        self.client.force_authenticate(self.user)

    @patch("history.views.requests.post")
    def test_chat_saves_user_and_assistant_messages(self, post):
        response = Mock()
        response.json.return_value = {"answer": "Hãy kiểm tra lá cây."}
        response.raise_for_status.return_value = None
        post.return_value = response

        result = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Lá cây bị vàng phải làm sao?"},
            format="json",
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            list(
                self.history.messages.values_list("role", "content")
            ),
            [
                ("user", "Lá cây bị vàng phải làm sao?"),
                ("assistant", "Hãy kiểm tra lá cây."),
            ],
        )
        self.history.refresh_from_db()
        self.assertEqual(self.history.title, "Lá cây bị vàng phải làm sao?")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["history"], [])

    @patch("history.views.requests.post")
    def test_chat_does_not_save_messages_when_ai_is_unavailable(self, post):
        import requests

        post.side_effect = requests.exceptions.Timeout

        result = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Cây bị bệnh gì?"},
            format="json",
        )

        self.assertEqual(result.status_code, 503)
        self.assertFalse(ChatMessage.objects.exists())
