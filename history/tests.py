import os
import re
import runpy
from wsgiref.util import is_hop_by_hop
from threading import Event
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from backend import settings as backend_settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import SimpleTestCase, TestCase, TransactionTestCase
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat
from .services.contracts import RagDetection, RagSuccess, ServiceFailure, YoloDetection, YoloSuccess
from .services.sse import encode_sse_event, keep_alive_frame, stream_operation


def image_file():
    image = Image.new("RGB", (1, 1), color="green")
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    stream.name = "leaf.png"
    stream.content_type = "image/png"
    return stream


def inline_stream_operation(operation, *, progress_payloads, error_payload, **_kwargs):
    for payload in progress_payloads:
        yield encode_sse_event(payload)
    try:
        yield encode_sse_event(operation())
    except Exception as error:
        yield encode_sse_event(error_payload(error))


class SseFrameTests(SimpleTestCase):
    def test_encode_sse_event_serializes_json_and_terminates_frame(self):
        frame = encode_sse_event({
            "type": "progress",
            "stage": "preparing",
            "message": "Đang chuẩn bị câu hỏi...",
        })

        self.assertEqual(
            frame,
            'data: {"type":"progress","stage":"preparing","message":"Đang chuẩn bị câu hỏi..."}\n\n',
        )

    def test_keep_alive_frame_is_an_sse_comment(self):
        self.assertEqual(keep_alive_frame(), ": keep-alive\n\n")

    def test_stream_operation_emits_progress_heartbeat_and_terminal_event(self):
        started = Event()
        release = Event()

        def operation():
            started.set()
            release.wait(timeout=1)
            return {"type": "complete", "answer": "Đã xong."}

        with patch("history.services.sse.close_old_connections") as close_connections:
            stream = iter(stream_operation(
                operation,
                progress_payloads=(
                    {"type": "progress", "stage": "preparing"},
                    {"type": "progress", "stage": "processing"},
                ),
                heartbeat_seconds=0.01,
                error_payload=lambda error: {"type": "error", "message": str(error)},
            ))

            self.assertEqual(next(stream), 'data: {"type":"progress","stage":"preparing"}\n\n')
            self.assertEqual(next(stream), 'data: {"type":"progress","stage":"processing"}\n\n')
            self.assertEqual(next(stream), ": keep-alive\n\n")
            self.assertTrue(started.wait(timeout=1))
            release.set()
            self.assertEqual(next(stream), 'data: {"type":"complete","answer":"Đã xong."}\n\n')
            self.assertGreaterEqual(close_connections.call_count, 2)

    def test_stream_operation_converts_worker_exception_to_error_event(self):
        def operation():
            raise RuntimeError("RAG failed")

        frames = list(stream_operation(
            operation,
            progress_payloads=(),
            error_payload=lambda error: {"type": "error", "message": str(error)},
        ))

        self.assertEqual(frames, ['data: {"type":"error","message":"RAG failed"}\n\n'])

    def test_stream_operation_maps_unserializable_terminal_result_to_error(self):
        frames = list(stream_operation(
            lambda: {"type": "complete", "answer": object()},
            progress_payloads=(),
            error_payload=lambda error: {"type": "error", "message": str(error)},
        ))

        self.assertEqual(frames[0][:7], "data: {")
        self.assertIn('"type":"error"', frames[0])

    def test_stream_operation_rejects_non_positive_heartbeat(self):
        with self.assertRaises(ValueError):
            list(stream_operation(
                lambda: {"type": "complete"},
                progress_payloads=(),
                heartbeat_seconds=0,
                error_payload=lambda error: {"type": "error", "message": str(error)},
            ))


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
        messages = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))
        self.assertEqual(len(messages), 2)
        user_msg, assistant_msg = messages
        self.assertEqual(
            [(user_msg.role, user_msg.content), (assistant_msg.role, assistant_msg.content)],
            [
                ("user", "Lá mai bị vàng thì phải làm sao?"),
                ("assistant", "Bạn nên kiểm tra lá và điều kiện tưới nước."),
            ],
        )
        self.assertEqual(user_msg.processing_status, "complete")
        self.assertEqual(assistant_msg.processing_status, "complete")
        self.assertEqual(assistant_msg.reply_to_id, user_msg.id)
        mock_ask.assert_called_once()
        payload = mock_ask.call_args.kwargs
        self.assertEqual(payload["question"], "Lá mai bị vàng thì phải làm sao?")
        self.assertEqual(payload["history"], [])

    @patch("history.views.RagClient.ask_text")
    def test_chat_stream_emits_progress_and_complete_after_persistence(self, mock_ask):
        mock_ask.return_value = RagSuccess(
            answer="Bạn nên kiểm tra độ ẩm đất.",
            request_id="req-stream-1",
        )

        with patch("history.views.stream_operation", inline_stream_operation):
            response = self.client.post(
                reverse("history-chat-stream", args=[self.history.id]),
                {"question": "Lá mai bị vàng thì làm sao?"},
                format="json",
                HTTP_X_REQUEST_ID="req-stream-1",
            )

        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertLess(body.index('"stage":"preparing"'), body.index('"stage":"processing"'))
        self.assertLess(body.index('"stage":"processing"'), body.index('"type":"complete"'))
        self.assertIn('"answer":"Bạn nên kiểm tra độ ẩm đất."', body)
        self.assertIn('"history_id":%d' % self.history.id, body)
        self.assertTrue(response["Content-Type"].startswith("text/event-stream"))
        self.assertFalse(
            [name for name, _ in response.items() if is_hop_by_hop(name)],
            "WSGI rejects application-supplied hop-by-hop headers",
        )
        self.assertEqual(response["X-Request-ID"], "req-stream-1")
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 2)

    @patch("history.views.RagClient.ask_text")
    def test_chat_stream_emits_error_and_marks_message_failed(self, mock_ask):
        mock_ask.return_value = ServiceFailure(
            code="rag_timeout",
            message="AI service phản hồi quá thời gian cho phép.",
            status_code=504,
            request_id="req-stream-error",
        )

        with patch("history.views.stream_operation", inline_stream_operation):
            response = self.client.post(
                reverse("history-chat-stream", args=[self.history.id]),
                {"question": "Cây mai bị bệnh gì?"},
                format="json",
                HTTP_X_REQUEST_ID="req-stream-error",
            )

        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('"type":"error"', body)
        self.assertIn('"code":"rag_timeout"', body)
        self.assertEqual(ChatMessage.objects.get(history=self.history).processing_status, "failed")

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
        failed_msg = ChatMessage.objects.get(history=self.history)
        self.assertEqual(failed_msg.role, "user")
        self.assertEqual(failed_msg.processing_status, "failed")

    @patch("history.views.RagClient.ask_text")
    def test_subsequent_rag_history_omits_persisted_blank_image_user_message(self, mock_ask):
        blank_image_message = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="",
        )
        legacy_assistant = ChatMessage.objects.create(
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
            [
                {
                    "message_id": legacy_assistant.id,
                    "role": "assistant",
                    "content": "Kết quả nhận diện ảnh trước đó.",
                }
            ],
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
        messages = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))
        self.assertEqual(len(messages), 2)
        user_msg, assistant_msg = messages
        self.assertTrue(user_msg.has_image)
        self.assertEqual(
            user_msg.detections,
            [{
                "class_id": 4,
                "name": "dom-la",
                "disease_name": "Đốm lá",
                "confidence": 0.89,
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            }],
        )
        self.assertEqual(user_msg.processing_status, "complete")
        self.assertEqual(assistant_msg.processing_status, "complete")
        self.assertEqual(assistant_msg.reply_to_id, user_msg.id)

    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_image_chat_stream_emits_complete_with_detections(self, mock_predict, mock_ask_image):
        detection = YoloDetection(
            class_id=4,
            name="dom-la",
            confidence=0.89,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        mock_predict.return_value = YoloSuccess(
            detections=(detection,), request_id="req-image-stream"
        )
        mock_ask_image.return_value = RagSuccess(
            answer="Lá có dấu hiệu đốm lá.",
            request_id="req-image-stream",
            detections=(RagDetection(
                class_id=4,
                name="dom-la",
                disease_name="Đốm lá",
                confidence=0.89,
                bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
            ),),
        )

        with patch("history.views.stream_operation", inline_stream_operation):
            response = self.client.post(
                reverse("history-image-chat-stream", args=[self.history.id]),
                {"image": image_file(), "question": "Bệnh này là gì?"},
                format="multipart",
                HTTP_X_REQUEST_ID="req-image-stream",
            )

        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('"stage":"preparing"', body)
        self.assertIn('"stage":"processing"', body)
        self.assertIn('"type":"complete"', body)
        self.assertIn('"disease_name":"Đốm lá"', body)
        user_msg = ChatMessage.objects.get(history=self.history, role="user")
        self.assertTrue(user_msg.has_image)
        self.assertEqual(user_msg.processing_status, "complete")

    @patch("history.views.YoloClient.predict")
    def test_image_chat_stream_emits_yolo_error_without_assistant_message(self, mock_predict):
        mock_predict.return_value = ServiceFailure(
            code="yolo_unavailable",
            message="Không thể kết nối tới YOLO service.",
            status_code=503,
            request_id="req-image-stream-error",
        )

        with patch("history.views.stream_operation", inline_stream_operation):
            response = self.client.post(
                reverse("history-image-chat-stream", args=[self.history.id]),
                {"image": image_file()},
                format="multipart",
                HTTP_X_REQUEST_ID="req-image-stream-error",
            )

        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn('"type":"error"', body)
        self.assertIn('"code":"yolo_unavailable"', body)
        self.assertEqual(ChatMessage.objects.filter(history=self.history).count(), 1)
        self.assertEqual(ChatMessage.objects.get(history=self.history).processing_status, "failed")

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
        failed_msg = ChatMessage.objects.get(history=self.history)
        self.assertEqual(failed_msg.processing_status, "failed")
        self.assertFalse(failed_msg.has_image)
        self.assertEqual(failed_msg.detections, [])

    def test_chat_does_not_expose_another_users_history(self):
        other_history = HistoryChat.objects.create(user=self.other_user)

        response = self.client.post(
            reverse("history-chat", args=[other_history.id]),
            {"question": "Xin chào"},
            format="json",
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(ChatMessage.objects.filter(history=other_history).count(), 0)

    @patch("history.views.RagClient.ask_text")
    def test_identical_questions_do_not_drop_earlier_turn(self, mock_ask):
        mock_ask.side_effect = [
            RagSuccess(
                answer="Câu trả lời lần 1.",
                request_id="req-ident-1",
            ),
            RagSuccess(
                answer="Câu trả lời lần 2.",
                request_id="req-ident-2",
            ),
        ]
        same_question = "Lá mai bị vàng thì phải làm sao?"

        first_res = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": same_question},
            format="json",
        )
        self.assertEqual(first_res.status_code, 200)
        u1, a1 = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))

        second_res = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": same_question},
            format="json",
        )
        self.assertEqual(second_res.status_code, 200)

        second_call_history = mock_ask.call_args_list[1].kwargs["history"]
        self.assertEqual(
            second_call_history,
            [
                {
                    "message_id": u1.id,
                    "role": "user",
                    "content": same_question,
                },
                {
                    "message_id": a1.id,
                    "role": "assistant",
                    "content": "Câu trả lời lần 1.",
                },
            ],
        )

    @patch("history.views.RagClient.ask_text")
    def test_concurrent_turn_snapshot_excludes_pending_and_includes_completed_pair(self, mock_ask):
        from .views import _history_before_current_user_message

        # Request 1 starts (u1 pending)
        u1 = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Câu hỏi 1 đang chạy",
            processing_status="pending",
        )
        # Request 2 starts concurrently while Request 1 is still pending (u2 created after u1)
        u2 = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Câu hỏi 2 chạy đồng thời",
            processing_status="pending",
        )

        # Case A: If Request 2 snapshots history while Request 1 is still pending, u1 is excluded
        self.assertEqual(_history_before_current_user_message(self.history, u2.id), [])

        # Even if a lone user row somehow had processing_status="complete" without a reply, it must not leak as a lone user message
        lone_user = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Tin nhắn mồ côi",
            processing_status="complete",
        )
        u_after_lone = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Câu hỏi sau tin mồ côi",
            processing_status="pending",
        )
        self.assertEqual(_history_before_current_user_message(self.history, u_after_lone.id), [])
        lone_user.delete()
        u_after_lone.delete()

        # Case B: Request 1 finishes AFTER u2 was created -> a1 has created_at > u2.created_at
        a1 = ChatMessage(
            history=self.history,
            role="assistant",
            content="Trả lời cho câu hỏi 1 (lưu sau khi u2 đã tạo).",
            processing_status="complete",
            reply_to=u1,
        )
        a1.full_clean()
        a1.save()
        u1.processing_status = "complete"
        u1.save(update_fields=["processing_status"])

        # Now when Request 2 reads history for u2.id, it must find (u1, a1) via reply_to despite a1.created_at > u2.created_at
        self.assertEqual(
            _history_before_current_user_message(self.history, u2.id),
            [
                {
                    "message_id": u1.id,
                    "role": "user",
                    "content": "Câu hỏi 1 đang chạy",
                },
                {
                    "message_id": a1.id,
                    "role": "assistant",
                    "content": "Trả lời cho câu hỏi 1 (lưu sau khi u2 đã tạo).",
                },
            ],
        )

        # Case C: End-to-end interleaving during ChatView.post where an earlier pending request stays excluded
        in_flight_u = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Lượt đang pending chưa xong",
            processing_status="pending",
        )
        mock_ask.return_value = RagSuccess(
            answer="Trả lời lượt mới.",
            request_id="req-conc-3",
        )
        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Lượt mới gửi khi lượt trước vẫn pending"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        # Only (u1, a1) should be in history; u2 and in_flight_u are pending and must be excluded
        self.assertEqual(
            mock_ask.call_args.kwargs["history"],
            [
                {
                    "message_id": u1.id,
                    "role": "user",
                    "content": "Câu hỏi 1 đang chạy",
                },
                {
                    "message_id": a1.id,
                    "role": "assistant",
                    "content": "Trả lời cho câu hỏi 1 (lưu sau khi u2 đã tạo).",
                },
            ],
        )
        in_flight_u.refresh_from_db()
        self.assertEqual(in_flight_u.processing_status, "pending")

    @patch("history.views.RagClient.ask_text")
    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_successful_blank_image_is_preserved(self, mock_predict, mock_ask_image, mock_ask_text):
        detection = YoloDetection(
            class_id=4,
            name="dom-la",
            confidence=0.89,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        mock_predict.side_effect = [
            YoloSuccess(detections=(detection,), request_id="req-img-det"),
            YoloSuccess(detections=(), request_id="req-img-nodet"),
        ]
        mock_ask_image.side_effect = [
            RagSuccess(
                answer="Lá có dấu hiệu đốm lá.",
                request_id="req-img-det",
                detections=(
                    RagDetection(
                        class_id=4,
                        name="dom-la",
                        disease_name="Đốm lá",
                        confidence=0.89,
                        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                    ),
                ),
            ),
            RagSuccess(
                answer="Ảnh chưa phát hiện dấu hiệu bệnh rõ ràng.",
                request_id="req-img-nodet",
                detections=(),
            ),
        ]
        mock_ask_text.return_value = RagSuccess(
            answer="Bạn có thể phun thuốc đặc trị đốm lá.",
            request_id="req-followup",
        )

        # 1. Blank image WITH detection
        res1 = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file()},
            format="multipart",
        )
        self.assertEqual(res1.status_code, 200)

        # 2. Blank image WITHOUT detection (detections == [])
        res2 = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file(), "question": ""},
            format="multipart",
        )
        self.assertEqual(res2.status_code, 200)

        msgs = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))
        self.assertEqual(len(msgs), 4)
        u_det, a_det, u_nodet, a_nodet = msgs

        # 3. Follow-up text question checks that BOTH blank image turns are preserved in history
        res3 = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Vậy bệnh ở ảnh đầu tiên xịt thuốc gì?"},
            format="json",
        )
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(
            mock_ask_text.call_args.kwargs["history"],
            [
                {
                    "message_id": u_det.id,
                    "role": "user",
                    "content": "",
                    "has_image": True,
                    "detections": [
                        {
                            "class_id": 4,
                            "name": "dom-la",
                            "disease_name": "Đốm lá",
                            "confidence": 0.89,
                            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                        }
                    ],
                },
                {
                    "message_id": a_det.id,
                    "role": "assistant",
                    "content": "Lá có dấu hiệu đốm lá.",
                },
                {
                    "message_id": u_nodet.id,
                    "role": "user",
                    "content": "",
                    "has_image": True,
                    "detections": [],
                },
                {
                    "message_id": a_nodet.id,
                    "role": "assistant",
                    "content": "Ảnh chưa phát hiện dấu hiệu bệnh rõ ràng.",
                },
            ],
        )

    @patch("history.views.RagClient.ask_text")
    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_failed_image_turn_is_excluded(self, mock_predict, mock_ask_image, mock_ask_text):
        detection = YoloDetection(
            class_id=4,
            name="dom-la",
            confidence=0.89,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        # 1. Image turn failing at YOLO with 503
        # 2. Image turn succeeding at YOLO but failing at RAG with 504
        mock_predict.side_effect = [
            ServiceFailure("yolo_unavailable", "YOLO 503", 503, "req-fail-1"),
            YoloSuccess(detections=(detection,), request_id="req-fail-2"),
        ]
        mock_ask_image.return_value = ServiceFailure(
            "rag_timeout",
            "AI service phản hồi quá thời gian cho phép.",
            504,
            "req-fail-2",
        )
        # 3. Text turn failing at RAG with 504, then 4. Text turn succeeding
        mock_ask_text.side_effect = [
            ServiceFailure("rag_timeout", "RAG 504", 504, "req-fail-3"),
            RagSuccess(answer="Câu trả lời thành công.", request_id="req-ok-4"),
        ]

        res_yolo_503 = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file(), "question": "Ảnh lỗi YOLO 503"},
            format="multipart",
        )
        self.assertEqual(res_yolo_503.status_code, 503)

        res_rag_504 = self.client.post(
            reverse("history-image-chat", args=[self.history.id]),
            {"image": image_file(), "question": ""},
            format="multipart",
        )
        self.assertEqual(res_rag_504.status_code, 504)

        res_text_504 = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Câu hỏi lỗi RAG 504"},
            format="json",
        )
        self.assertEqual(res_text_504.status_code, 504)

        failed_msgs = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))
        self.assertEqual(len(failed_msgs), 3)
        for msg in failed_msgs:
            self.assertEqual(msg.processing_status, "failed")
            self.assertFalse(msg.has_image)
            self.assertEqual(msg.detections, [])

        res_ok = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Câu hỏi tiếp theo sau các lượt lỗi"},
            format="json",
        )
        self.assertEqual(res_ok.status_code, 200)
        self.assertEqual(mock_ask_text.call_args_list[1].kwargs["history"], [])

    @patch("history.views.RagClient.ask_text")
    def test_history_limit_is_twenty_messages_not_twenty_pairs(self, mock_ask):
        created_pairs = []
        for idx in range(1, 12):
            u = ChatMessage.objects.create(
                history=self.history,
                role="user",
                content=f"Câu hỏi {idx}",
                processing_status="complete",
            )
            a = ChatMessage(
                history=self.history,
                role="assistant",
                content=f"Trả lời {idx}",
                processing_status="complete",
                reply_to=u,
            )
            a.full_clean()
            a.save()
            created_pairs.append((u, a))

        mock_ask.return_value = RagSuccess(answer="Trả lời 12", request_id="req-12")
        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Câu hỏi 12"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        forwarded_history = mock_ask.call_args.kwargs["history"]
        self.assertEqual(len(forwarded_history), 20)
        expected_messages = []
        for u, a in created_pairs[1:]:
            expected_messages.append({"message_id": u.id, "role": "user", "content": u.content})
            expected_messages.append({"message_id": a.id, "role": "assistant", "content": a.content})
        self.assertEqual(forwarded_history, expected_messages)

    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_no_transaction_held_during_external_calls_and_serializer_exposes_metadata(
        self, mock_predict, mock_ask_image
    ):
        baseline_atomic_depth = len(connection.atomic_blocks)
        atomic_depths_during_full_clean = []
        original_full_clean = ChatMessage.full_clean

        def check_predict(**_kwargs):
            self.assertEqual(len(connection.atomic_blocks), baseline_atomic_depth)
            return YoloSuccess(
                detections=(
                    YoloDetection(
                        class_id=4,
                        name="dom-la",
                        confidence=0.91,
                        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                    ),
                ),
                request_id="req-atomic-check",
            )

        def check_ask_image(**_kwargs):
            self.assertEqual(len(connection.atomic_blocks), baseline_atomic_depth)
            return RagSuccess(
                answer="Kết quả kiểm tra transaction.",
                request_id="req-atomic-check",
                detections=(
                    RagDetection(
                        class_id=4,
                        name="dom-la",
                        disease_name="Đốm lá",
                        confidence=0.91,
                        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                    ),
                ),
            )

        def spy_full_clean(instance, *args, **kwargs):
            atomic_depths_during_full_clean.append(len(connection.atomic_blocks))
            return original_full_clean(instance, *args, **kwargs)

        mock_predict.side_effect = check_predict
        mock_ask_image.side_effect = check_ask_image

        with patch.object(ChatMessage, "full_clean", autospec=True, side_effect=spy_full_clean):
            res = self.client.post(
                reverse("history-image-chat", args=[self.history.id]),
                {"image": image_file(), "question": "Kiểm tra ảnh"},
                format="multipart",
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(atomic_depths_during_full_clean, [baseline_atomic_depth + 1])

        detail_res = self.client.get(reverse("history-detail", args=[self.history.id]))
        self.assertEqual(detail_res.status_code, 200)
        self.assertEqual(len(detail_res.data["messages"]), 2)
        user_item, assistant_item = detail_res.data["messages"]
        self.assertTrue(user_item["has_image"])
        self.assertEqual(user_item["processing_status"], "complete")
        self.assertEqual(len(user_item["detections"]), 1)
        self.assertFalse(assistant_item["has_image"])
        self.assertEqual(assistant_item["processing_status"], "complete")
        self.assertEqual(assistant_item["detections"], [])

    @patch("history.views.RagClient.ask_text")
    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_unexpected_exception_marks_pending_user_message_failed(
        self, mock_predict, mock_ask_image, mock_ask_text
    ):
        baseline_atomic_depth = len(connection.atomic_blocks)

        def raise_in_ask_text(**_kwargs):
            self.assertEqual(len(connection.atomic_blocks), baseline_atomic_depth)
            raise RuntimeError("Token fetch or unexpected client error in ask_text")

        def raise_in_predict(**_kwargs):
            self.assertEqual(len(connection.atomic_blocks), baseline_atomic_depth)
            raise RuntimeError("Unexpected client error in YoloClient.predict")

        mock_ask_text.side_effect = raise_in_ask_text
        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("history-chat", args=[self.history.id]),
                {"question": "Câu hỏi gây lỗi ngoại lệ"},
                format="json",
            )

        mock_predict.side_effect = raise_in_predict
        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("history-image-chat", args=[self.history.id]),
                {"image": image_file(), "question": "Ảnh gây lỗi ngoại lệ ở YOLO"},
                format="multipart",
            )

        # Also test exception raised during RagClient.ask_image after YOLO succeeded
        mock_predict.side_effect = None
        mock_predict.return_value = YoloSuccess(
            detections=(
                YoloDetection(
                    class_id=4,
                    name="dom-la",
                    confidence=0.9,
                    bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                ),
            ),
            request_id="req-exc-img",
        )
        mock_ask_image.side_effect = RuntimeError("Unexpected error in ask_image")
        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("history-image-chat", args=[self.history.id]),
                {"image": image_file(), "question": "Ảnh gây lỗi ngoại lệ ở RAG"},
                format="multipart",
            )

        # Also test exception raised inside the final save atomic block (e.g., full_clean failure)
        mock_ask_text.side_effect = None
        mock_ask_text.return_value = RagSuccess(answer="Trả lời", request_id="req-exc-save")
        with patch.object(ChatMessage, "full_clean", side_effect=ValidationError("Save validation failed")):
            with self.assertRaises(ValidationError):
                self.client.post(
                    reverse("history-chat", args=[self.history.id]),
                    {"question": "Câu hỏi lỗi lúc lưu assistant"},
                    format="json",
                )

        msgs = list(ChatMessage.objects.filter(history=self.history).order_by("created_at", "id"))
        self.assertEqual(len(msgs), 4)
        for msg in msgs:
            self.assertEqual(msg.role, "user")
            self.assertEqual(msg.processing_status, "failed")
            self.assertFalse(msg.has_image)
            self.assertEqual(msg.detections, [])

    def test_history_limit_does_not_split_completed_pair_on_odd_message_count(self):
        from .views import _history_before_current_user_message

        # 10 completed pairs plus one legacy assistant: keep nine full pairs and the tail.
        pairs = []
        for idx in range(1, 11):
            u = ChatMessage.objects.create(
                history=self.history,
                role="user",
                content=f"Câu hỏi cặp {idx}",
                processing_status="complete",
            )
            a = ChatMessage(
                history=self.history,
                role="assistant",
                content=f"Trả lời cặp {idx}",
                processing_status="complete",
                reply_to=u,
            )
            a.full_clean()
            a.save()
            pairs.append((u, a))

        legacy_tail = ChatMessage.objects.create(
            history=self.history,
            role="assistant",
            content="Tin assistant legacy đứng lẻ phía sau",
            processing_status="complete",
        )
        current_u = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Câu hỏi hiện tại",
            processing_status="pending",
        )

        history_payload = _history_before_current_user_message(self.history, current_u.id)
        # Drop pair 1 as a whole unit: nine pairs and one legacy message.
        self.assertEqual(len(history_payload), 19)
        self.assertEqual(history_payload[0]["role"], "user")
        self.assertEqual(history_payload[0]["message_id"], pairs[1][0].id)
        expected = []
        for u, a in pairs[1:]:
            expected.append({"message_id": u.id, "role": "user", "content": u.content})
            expected.append({"message_id": a.id, "role": "assistant", "content": a.content})
        expected.append(
            {
                "message_id": legacy_tail.id,
                "role": "assistant",
                "content": legacy_tail.content,
            }
        )
        self.assertEqual(history_payload, expected)

    def test_long_history_message_is_clipped_for_router_budget(self):
        from .views import _history_before_current_user_message

        previous_user = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="x" * 1600,
            processing_status="complete",
        )
        previous_assistant = ChatMessage.objects.create(
            history=self.history,
            role="assistant",
            content="y" * 1600,
            processing_status="complete",
            reply_to=previous_user,
        )
        current = ChatMessage.objects.create(
            history=self.history,
            role="user",
            content="Câu hỏi hiện tại",
            processing_status="pending",
        )

        payload = _history_before_current_user_message(self.history, current.id)

        self.assertEqual([item["message_id"] for item in payload], [previous_user.id, previous_assistant.id])
        self.assertEqual([len(item["content"]) for item in payload], [1200, 1200])


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
        with self.assertRaises(ValidationError) as cm:
            invalid_user_msg.full_clean()
        self.assertIn("reply_to", cm.exception.message_dict)

        # Invariant 2: Assistant message cannot reply to another assistant message
        invalid_asst_to_asst = ChatMessage(
            history=self.history,
            role="assistant",
            content="Trả lời phụ",
            reply_to=reply_msg,
        )
        with self.assertRaises(ValidationError) as cm:
            invalid_asst_to_asst.full_clean()
        self.assertIn("reply_to", cm.exception.message_dict)

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
        with self.assertRaises(ValidationError) as cm:
            cross_chat_reply.full_clean()
        self.assertIn("reply_to", cm.exception.message_dict)

        # Invariant 4: Non-existent reply_to_id
        non_existent_reply = ChatMessage(
            history=self.history,
            role="assistant",
            content="Trả lời tin nhắn không tồn tại",
            reply_to_id=999999,
        )
        with self.assertRaises(ValidationError) as cm:
            non_existent_reply.full_clean()
        self.assertIn("reply_to", cm.exception.message_dict)

        # Invariant 5: Unsaved in-memory HistoryChat cross-reference
        unsaved_history_1 = HistoryChat(user=self.user, title="Unsaved 1")
        unsaved_history_2 = HistoryChat(user=self.user, title="Unsaved 2")
        unsaved_user_msg = ChatMessage(history=unsaved_history_1, role="user", content="u")
        cross_unsaved_reply = ChatMessage(
            history=unsaved_history_2,
            role="assistant",
            content="cross unsaved",
            reply_to=unsaved_user_msg,
        )
        with self.assertRaises(ValidationError) as cm:
            cross_unsaved_reply.clean()
        self.assertIn("reply_to", cm.exception.message_dict)

        # Valid unsaved in-memory HistoryChat with matching history passes clean()
        valid_unsaved_reply = ChatMessage(
            history=unsaved_history_1,
            role="assistant",
            content="valid unsaved",
            reply_to=unsaved_user_msg,
        )
        valid_unsaved_reply.clean()

        # Invariant 6: Invalid processing_status choice raises ValidationError
        invalid_status_msg = ChatMessage(
            history=self.history,
            role="user",
            content="Trạng thái không hợp lệ",
            processing_status="unknown_status",
        )
        with self.assertRaises(ValidationError) as cm:
            invalid_status_msg.full_clean()
        self.assertIn("processing_status", cm.exception.message_dict)

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


class ChatMessageMigrationTests(TransactionTestCase):
    def tearDown(self):
        # Restore test database to latest migrations
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(executor.loader.graph.leaf_nodes())
        super().tearDown()

    def test_migration_0001_to_0002_backfill_and_defaults(self):
        migrate_from = [("history", "0001_initial")]
        migrate_to = [("history", "0002_chatmessage_evidence_status_reply")]

        executor = MigrationExecutor(connection)
        executor.migrate(migrate_from)

        old_state = executor.loader.project_state(migrate_from)
        old_apps = old_state.apps
        OldUser = old_apps.get_model("user", "User")
        OldHistoryChat = old_apps.get_model("history", "HistoryChat")
        OldChatMessage = old_apps.get_model("history", "ChatMessage")

        user = OldUser.objects.create(username="mig-user", email="mig@example.com")

        # 1. Unambiguous legacy pair: user -> assistant
        h1 = OldHistoryChat.objects.create(user=user, title="H1")
        u1 = OldChatMessage.objects.create(history=h1, role="user", content="hello")
        a1 = OldChatMessage.objects.create(history=h1, role="assistant", content="world")

        # 2. Ambiguous legacy sequence: user -> user -> assistant
        h2 = OldHistoryChat.objects.create(user=user, title="H2")
        u2_1 = OldChatMessage.objects.create(history=h2, role="user", content="q1")
        u2_2 = OldChatMessage.objects.create(history=h2, role="user", content="q2")
        a2_1 = OldChatMessage.objects.create(history=h2, role="assistant", content="ans")

        # Migrate forward to 0002
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(migrate_to)

        new_state = executor.loader.project_state(migrate_to)
        new_apps = new_state.apps
        NewChatMessage = new_apps.get_model("history", "ChatMessage")

        # Verify unambiguous user message
        mig_u1 = NewChatMessage.objects.get(id=u1.id)
        self.assertEqual(mig_u1.processing_status, "complete")
        self.assertFalse(mig_u1.has_image)
        self.assertEqual(mig_u1.detections, [])
        self.assertIsNone(mig_u1.reply_to_id)

        # Verify unambiguous assistant message receives reply_to link
        mig_a1 = NewChatMessage.objects.get(id=a1.id)
        self.assertEqual(mig_a1.processing_status, "complete")
        self.assertFalse(mig_a1.has_image)
        self.assertEqual(mig_a1.detections, [])
        self.assertEqual(mig_a1.reply_to_id, u1.id)

        # Verify ambiguous sequence rows remain unlinked
        mig_u2_1 = NewChatMessage.objects.get(id=u2_1.id)
        self.assertEqual(mig_u2_1.processing_status, "complete")
        self.assertFalse(mig_u2_1.has_image)
        self.assertEqual(mig_u2_1.detections, [])
        self.assertIsNone(mig_u2_1.reply_to_id)

        mig_u2_2 = NewChatMessage.objects.get(id=u2_2.id)
        self.assertEqual(mig_u2_2.processing_status, "complete")
        self.assertFalse(mig_u2_2.has_image)
        self.assertEqual(mig_u2_2.detections, [])
        self.assertIsNone(mig_u2_2.reply_to_id)

        mig_a2_1 = NewChatMessage.objects.get(id=a2_1.id)
        self.assertEqual(mig_a2_1.processing_status, "complete")
        self.assertFalse(mig_a2_1.has_image)
        self.assertEqual(mig_a2_1.detections, [])
        self.assertIsNone(mig_a2_1.reply_to_id)
