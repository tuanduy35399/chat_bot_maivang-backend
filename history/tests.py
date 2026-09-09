import os
import runpy
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from backend import settings as backend_settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import SimpleTestCase
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat
from .services.contracts import RagSuccess, ServiceFailure, YoloDetection, YoloSuccess


def image_file():
    image = Image.new("RGB", (1, 1), color="green")
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    stream.name = "leaf.png"
    stream.content_type = "image/png"
    return stream


class SettingsDefaultsTests(SimpleTestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_settings_defaults_yolo_timeout_when_environment_is_missing(self, _load_dotenv):
        values = runpy.run_path(backend_settings.__file__)

        self.assertEqual(values["YOLO_API_TIMEOUT"], 30.0)


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

    @patch("history.views.RagClient.ask_text")
    def test_chat_calls_fastapi_and_saves_both_messages(self, mock_ask):
        mock_ask.return_value = RagSuccess(
            answer="Bạn nên kiểm tra lá và điều kiện tưới nước.",
            request_id="req-text-1",
        )

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
        mock_ask.assert_called_once()
        payload = mock_ask.call_args.kwargs
        self.assertEqual(payload["question"], "Lá mai bị vàng thì phải làm sao?")
        self.assertEqual(payload["history"], [])

    @patch("history.views.RagClient.ask_text")
    def test_chat_returns_503_when_fastapi_is_unavailable(self, mock_ask):
        mock_ask.return_value = ServiceFailure(
            code="rag_unavailable",
            message="Không thể kết nối tới AI service.",
            status_code=503,
            request_id="req-text-1",
        )

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Cây mai bị bệnh gì?"},
            format="json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 1)
        self.assertEqual(ChatMessage.objects.get(history=self.history).role, "user")

    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_image_view_forwards_blank_question_and_excludes_current_user_history(
        self, mock_predict, mock_ask_image
    ):
        detection = YoloDetection(
            class_id=4,
            name="dom-la",
            confidence=0.89,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        mock_predict.return_value = YoloSuccess(
            detections=(detection,), request_id="req-image-1"
        )
        mock_ask_image.return_value = RagSuccess(
            answer="Lá có dấu hiệu đốm lá.", request_id="req-image-1"
        )

        response = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file()},
            format="multipart",
            HTTP_X_REQUEST_ID="req-image-1",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["question"], "")
        self.assertEqual(response.data["detections"], [{
            "class_id": 4,
            "name": "dom-la",
            "confidence": 0.89,
            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
        }])
        self.assertEqual(mock_predict.call_args.kwargs["request_id"], "req-image-1")
        rag_kwargs = mock_ask_image.call_args.kwargs
        self.assertEqual(rag_kwargs["question"], "")
        self.assertEqual(rag_kwargs["history"], [])
        self.assertEqual(rag_kwargs["request_id"], "req-image-1")
        self.assertEqual(rag_kwargs["detections"], (detection,))

    @patch("history.views.YoloClient.predict")
    def test_image_view_returns_safe_yolo_failure_without_assistant_message(self, mock_predict):
        mock_predict.return_value = ServiceFailure(
            code="yolo_unavailable",
            message="Không thể kết nối tới YOLO service.",
            status_code=503,
            request_id="req-image-1",
        )

        response = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file()},
            format="multipart",
            HTTP_X_REQUEST_ID="req-image-1",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "yolo_unavailable")
        self.assertEqual(response.data["request_id"], "req-image-1")
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 1)

    def test_chat_does_not_expose_another_users_history(self):
        other_history = HistoryChat.objects.create(user=self.other_user)

        response = self.client.post(
            reverse("history-chat", args=[other_history.id]),
            {"question": "Xin chào"},
            format="json",
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(ChatMessage.objects.filter(history=other_history).count(), 0)
