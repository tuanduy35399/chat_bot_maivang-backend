from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat


class ChatViewTests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="chat-user",
            name="Chat User",
            email="chat-user@example.com",
            password="Strong-password-123!",
        )
        self.other_user = get_user_model().objects.create_user(
            username="other-user",
            name="Other User",
            email="other-user@example.com",
            password="Strong-password-123!",
        )
        self.history = HistoryChat.objects.create(
            user=self.user,
            title="Tư vấn bệnh cây mai",
        )
        self.client.force_authenticate(user=self.user)

    @patch("history.views.requests.post")
    def test_chat_calls_fastapi_and_saves_both_messages(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            "answer": "Bạn nên kiểm tra lá và điều kiện tưới nước."
        }

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Lá mai bị vàng thì phải làm sao?"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["history_id"], self.history.id)
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 2)
        self.assertEqual(
            list(
                ChatMessage.objects.filter(history=self.history).values_list(
                    "role", "content"
                )
            ),
            [
                ("user", "Lá mai bị vàng thì phải làm sao?"),
                ("assistant", "Bạn nên kiểm tra lá và điều kiện tưới nước."),
            ],
        )
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["question"], "Lá mai bị vàng thì phải làm sao?")
        self.assertEqual(payload["history"][0]["role"], "user")

    @patch("history.views.requests.post")
    def test_chat_returns_503_when_fastapi_is_unavailable(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Cây mai bị bệnh gì?"},
            format="json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 1)
        self.assertEqual(ChatMessage.objects.get(history=self.history).role, "user")

    def test_chat_does_not_expose_another_users_history(self):
        other_history = HistoryChat.objects.create(user=self.other_user)

        response = self.client.post(
            reverse("history-chat", args=[other_history.id]),
            {"question": "Xin chào"},
            format="json",
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(ChatMessage.objects.filter(history=other_history).count(), 0)
