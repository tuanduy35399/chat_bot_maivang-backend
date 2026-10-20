import os
import re
import runpy
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from backend import settings as backend_settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat
from .services.contracts import RagDetection, RagSuccess, ServiceFailure, YoloDetection, YoloSuccess


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

    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_settings_default_id_token_audience_is_empty(self, _load_dotenv):
        values = runpy.run_path(backend_settings.__file__)
        self.assertEqual(values["RAG_ID_TOKEN_AUDIENCE"], "")

    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_settings_default_does_not_require_database_url(self, _load_dotenv):
        values = runpy.run_path(backend_settings.__file__)
        self.assertFalse(values["DJANGO_REQUIRE_DATABASE_URL"])
        self.assertEqual(values["DATABASES"]["default"]["ENGINE"], "django.db.backends.sqlite3")

    @patch.dict(
        os.environ,
        {"DJANGO_REQUIRE_DATABASE_URL": "true"},
        clear=True,
    )
    @patch("dotenv.load_dotenv")
    def test_settings_require_database_url_without_value_raises(self, _load_dotenv):
        with self.assertRaises(Exception) as raised:
            runpy.run_path(backend_settings.__file__)
        self.assertIn("DATABASE_URL", str(raised.exception))

    @patch.dict(
        os.environ,
        {
            "CSRF_TRUSTED_ORIGINS": "https://maivang-api-xyz.run.app",
            "DJANGO_TRUST_X_FORWARDED_PROTO": "true",
        },
        clear=True,
    )
    @patch("dotenv.load_dotenv")
    def test_settings_cloud_run_proxy_and_csrf(self, _load_dotenv):
        values = runpy.run_path(backend_settings.__file__)
        self.assertEqual(
            values["CSRF_TRUSTED_ORIGINS"],
            ["https://maivang-api-xyz.run.app"],
        )
        self.assertEqual(
            values["SECURE_PROXY_SSL_HEADER"],
            ("HTTP_X_FORWARDED_PROTO", "https"),
        )


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

    @patch("history.views.RagClient.ask_text")
    def test_subsequent_rag_history_omits_persisted_blank_image_user_message(self, mock_ask):
        blank_image_message = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="",
        )
        ChatMessage.objects.create(
            history=self.history,
            role="assistant",
            content="Kết quả nhận diện ảnh trước đó.",
        )
        mock_ask.return_value = RagSuccess(
            answer="Bạn nên kiểm tra thêm độ ẩm đất.",
            request_id="req-text-2",
        )

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Cần chăm thế nào tiếp?"},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            ChatMessage.objects.filter(pk=blank_image_message.pk, content="").exists()
        )
        self.assertEqual(
            mock_ask.call_args.kwargs["history"],
            [{"role": "assistant", "content": "Kết quả nhận diện ảnh trước đó."}],
        )

    @patch("history.views.RagClient.ask_text")
    def test_chat_replaces_request_id_rejected_by_downstream_service(self, mock_ask):
        mock_ask.return_value = RagSuccess(
            answer="Bạn nên kiểm tra lá.", request_id="generated-request-id"
        )

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Lá có bị bệnh không?"},
            format="json",
            HTTP_X_REQUEST_ID="trace:1",
        )

        request_id = mock_ask.call_args.kwargs["request_id"]
        self.assertEqual(request_id, "trace:1")
        self.assertRegex(request_id, re.compile(r"^[A-Za-z0-9._:-]{1,128}$"))
        self.assertEqual(response["X-Request-ID"], request_id)

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
            answer="Lá có dấu hiệu đốm lá.",
            request_id="req-image-1",
            detections=(RagDetection(
                class_id=4,
                name="dom-la",
                disease_name="Đốm lá",
                confidence=0.89,
                bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
            ),),
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
            "disease_name": "Đốm lá",
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


class ChatMessageModelTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="model-user",
            name="Model User",
            email="model-user@example.com",
            password="Strong-password-123!",
        )
        self.history = HistoryChat.objects.create(
            user=self.user,
            title="Cuộc trò chuyện thử nghiệm",
        )

    def test_chat_message_evidence_defaults_and_reply_link(self):
        # Default values on creation
        msg = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Xin chào",
        )
        self.assertFalse(msg.has_image)
        self.assertEqual(msg.detections, [])
        self.assertEqual(msg.processing_status, "complete")
        self.assertIsNone(msg.reply_to)

        # Image message with blank content and detections passes full_clean
        img_msg = ChatMessage(
            history=self.history,
            role="user",
            content="",
            has_image=True,
            detections=[{"class_id": 4, "name": "dom-la", "confidence": 0.89}],
        )
        img_msg.full_clean()
        img_msg.save()
        self.assertTrue(img_msg.has_image)
        self.assertEqual(len(img_msg.detections), 1)

        # Blank text-only message also passes full_clean
        blank_msg = ChatMessage(
            history=self.history,
            role="user",
            content="",
        )
        blank_msg.full_clean()

        # Valid assistant reply_to pointing to user message in same history passes full_clean
        reply_msg = ChatMessage(
            history=self.history,
            role="assistant",
            content="Cây mai bị đốm lá.",
            reply_to=img_msg,
        )
        reply_msg.full_clean()
        reply_msg.save()
        self.assertEqual(reply_msg.reply_to, img_msg)
        self.assertEqual(list(img_msg.assistant_replies.all()), [reply_msg])

        # Invariant 1: User message cannot have reply_to
        invalid_user_msg = ChatMessage(
            history=self.history,
            role="user",
            content="Hỏi tiếp",
            reply_to=img_msg,
        )
        with self.assertRaises(ValidationError):
            invalid_user_msg.full_clean()

        # Invariant 2: Assistant message cannot reply to another assistant message
        invalid_asst_to_asst = ChatMessage(
            history=self.history,
            role="assistant",
            content="Trả lời phụ",
            reply_to=reply_msg,
        )
        with self.assertRaises(ValidationError):
            invalid_asst_to_asst.full_clean()

        # Invariant 3: Assistant message cannot reply to a user message from a different HistoryChat
        other_history = HistoryChat.objects.create(
            user=self.user,
            title="Cuộc trò chuyện khác",
        )
        other_user_msg = ChatMessage.objects.create(
            history=other_history,
            role="user",
            content="Tin nhắn ở đoạn chat khác",
        )
        cross_chat_reply = ChatMessage(
            history=self.history,
            role="assistant",
            content="Trả lời nhầm đoạn chat",
            reply_to=other_user_msg,
        )
        with self.assertRaises(ValidationError):
            cross_chat_reply.full_clean()

        # Ordering check: ordering by created_at, id
        self.assertEqual(ChatMessage._meta.ordering, ["created_at", "id"])

    def test_legacy_messages_have_complete_processing_status_and_null_reply_to(self):
        msg = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Tin nhắn cũ",
        )
        self.assertEqual(msg.processing_status, "complete")
        self.assertIsNone(msg.reply_to)

    def test_migration_backfill_unambiguous_and_ambiguous_pairs(self):
        import importlib
        migration_module = importlib.import_module("history.migrations.0002_chatmessage_evidence_status_reply")
        backfill_reply_to = migration_module.backfill_reply_to
        from django.apps import apps

        # History 1: Unambiguous adjacent pair (u1, a1)
        h1 = HistoryChat.objects.create(user=self.user, title="H1")
        h1_u1 = ChatMessage.objects.create(history=h1, role="user", content="u1")
        h1_a1 = ChatMessage.objects.create(history=h1, role="assistant", content="a1")

        # History 2: Multiple unambiguous pairs (u1, a1) and (u2, a2)
        h2 = HistoryChat.objects.create(user=self.user, title="H2")
        h2_u1 = ChatMessage.objects.create(history=h2, role="user", content="u1")
        h2_a1 = ChatMessage.objects.create(history=h2, role="assistant", content="a1")
        h2_u2 = ChatMessage.objects.create(history=h2, role="user", content="u2")
        h2_a2 = ChatMessage.objects.create(history=h2, role="assistant", content="a2")

        # History 3: Ambiguous consecutive user messages (u1, u2, a1)
        h3 = HistoryChat.objects.create(user=self.user, title="H3")
        h3_u1 = ChatMessage.objects.create(history=h3, role="user", content="u1")
        h3_u2 = ChatMessage.objects.create(history=h3, role="user", content="u2")
        h3_a1 = ChatMessage.objects.create(history=h3, role="assistant", content="a1")

        # History 4: Ambiguous consecutive assistant messages (u1, a1, a2)
        h4 = HistoryChat.objects.create(user=self.user, title="H4")
        h4_u1 = ChatMessage.objects.create(history=h4, role="user", content="u1")
        h4_a1 = ChatMessage.objects.create(history=h4, role="assistant", content="a1")
        h4_a2 = ChatMessage.objects.create(history=h4, role="assistant", content="a2")

        # History 5: Mixed history - ambiguous (u1, a1, a2) followed by unambiguous (u2, a3)
        h5 = HistoryChat.objects.create(user=self.user, title="H5")
        h5_u1 = ChatMessage.objects.create(history=h5, role="user", content="u1")
        h5_a1 = ChatMessage.objects.create(history=h5, role="assistant", content="a1")
        h5_a2 = ChatMessage.objects.create(history=h5, role="assistant", content="a2")
        h5_u2 = ChatMessage.objects.create(history=h5, role="user", content="u2")
        h5_a3 = ChatMessage.objects.create(history=h5, role="assistant", content="a3")

        # History 6: Assistant already has reply_to set - should not be modified
        h6 = HistoryChat.objects.create(user=self.user, title="H6")
        h6_u1 = ChatMessage.objects.create(history=h6, role="user", content="u1")
        h6_u2 = ChatMessage.objects.create(history=h6, role="user", content="u2")
        h6_a1 = ChatMessage.objects.create(history=h6, role="assistant", content="a1", reply_to=h6_u1)

        # Run migration backfill function
        backfill_reply_to(apps, None)

        # Assertions
        h1_a1.refresh_from_db()
        self.assertEqual(h1_a1.reply_to_id, h1_u1.id)

        h2_a1.refresh_from_db()
        h2_a2.refresh_from_db()
        self.assertEqual(h2_a1.reply_to_id, h2_u1.id)
        self.assertEqual(h2_a2.reply_to_id, h2_u2.id)

        h3_a1.refresh_from_db()
        self.assertIsNone(h3_a1.reply_to_id)

        h4_a1.refresh_from_db()
        h4_a2.refresh_from_db()
        self.assertIsNone(h4_a1.reply_to_id)
        self.assertIsNone(h4_a2.reply_to_id)

        h5_a1.refresh_from_db()
        h5_a2.refresh_from_db()
        h5_a3.refresh_from_db()
        self.assertIsNone(h5_a1.reply_to_id)
        self.assertIsNone(h5_a2.reply_to_id)
        self.assertEqual(h5_a3.reply_to_id, h5_u2.id)

        h6_a1.refresh_from_db()
        self.assertEqual(h6_a1.reply_to_id, h6_u1.id)
