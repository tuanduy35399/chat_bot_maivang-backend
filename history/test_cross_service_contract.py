"""Offline Django boundary evidence for the seven supported RAG scenarios."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

from PIL import Image
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APITestCase

from .models import ChatMessage, HistoryChat
from .services.contracts import RagDetection, RagSuccess, YoloDetection, YoloSuccess


CASES = (
    "text_only",
    "image_only_disease",
    "image_only_healthy",
    "image_only_empty",
    "related_text_image",
    "irrelevant_text_image",
    "text_refers_to_empty_image",
)


def _image_file():
    image = Image.new("RGB", (1, 1), color="green")
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    stream.name = "leaf.png"
    stream.content_type = "image/png"
    return stream


class CrossServiceContractTests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="contract-user",
            name="Contract User",
            email="contract-user@example.com",
            password="Strong-password-123!",
        )
        self.history = HistoryChat.objects.create(user=self.user, title="Contract matrix")
        self.client.force_authenticate(user=self.user)

    @patch("history.views.RagClient.ask_text")
    def test_text_only_contract_uses_text_route_and_keeps_legacy_four_field_shape(
        self, mock_ask_text
    ):
        """Catches the text route omitting the legacy empty detections field."""
        request_id = "trace:contract:text-only"
        mock_ask_text.return_value = RagSuccess(
            answer="Cần kiểm tra lá và xử lý sớm.", request_id=request_id
        )

        response = self.client.post(
            reverse("history-chat", args=[self.history.id]),
            {"question": "Cách trị rỉ sắt?"},
            format="json",
            HTTP_X_REQUEST_ID=request_id,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Request-ID"], request_id)
        self.assertEqual(
            set(response.data), {"question", "answer", "history_id", "detections"}
        )
        self.assertEqual(response.data["detections"], [])
        self.assertNotIn("code", response.data)
        self.assertEqual(mock_ask_text.call_args.kwargs, {
            "question": "Cách trị rỉ sắt?",
            "history": [],
            "request_id": request_id,
        })
        self.assertEqual(
            list(ChatMessage.objects.filter(history=self.history).values_list("role", "content")),
            [
                ("user", "Cách trị rỉ sắt?"),
                ("assistant", "Cần kiểm tra lá và xử lý sớm."),
            ],
        )

    @patch("history.views.RagClient.ask_image")
    @patch("history.views.YoloClient.predict")
    def test_contract_matrix_keeps_legacy_public_shape_and_canonical_detections(
        self, mock_predict, mock_ask_image
    ):
        raw_disease = (
            YoloDetection(4, "rust", 0.91, (1.0, 2.0, 3.0, 4.0)),
            YoloDetection(4, "rust", 0.72, (5.0, 6.0, 7.0, 8.0)),
        )
        raw_healthy = (YoloDetection(1, "healthy", 0.98, (9.0, 10.0, 11.0, 12.0)),)

        for case in CASES[1:]:
            with self.subTest(case=case):
                question = {
                    "text_only": "Cách trị rỉ sắt?",
                    "related_text_image": "Đốm lá trong ảnh chữa thế nào?",
                    "irrelevant_text_image": "Mai vàng cần bao nhiêu nắng?",
                    "text_refers_to_empty_image": "Bệnh trong ảnh là gì?",
                }.get(case, "")
                raw_detections = raw_healthy if case == "image_only_healthy" else raw_disease
                if case in {"image_only_empty", "text_refers_to_empty_image"}:
                    raw_detections = ()
                canonical = tuple(
                    RagDetection(
                        class_id=item.class_id,
                        name=item.name,
                        disease_name="Lá khỏe" if item.class_id == 1 else "Rỉ sắt",
                        confidence=item.confidence,
                        bbox_xyxy=item.bbox_xyxy,
                    )
                    for item in raw_detections
                )
                request_id = f"trace:contract:{case}"
                mock_predict.return_value = YoloSuccess(raw_detections, request_id)
                mock_ask_image.return_value = RagSuccess(
                    answer=f"answer:{case}", request_id=request_id, detections=canonical
                )

                response = self.client.post(
                    reverse("history-image-chat", args=[self.history.id]),
                    {"image": _image_file(), "question": question},
                    format="multipart",
                    HTTP_X_REQUEST_ID=request_id,
                )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response["X-Request-ID"], request_id)
                self.assertEqual(set(response.data), {"question", "answer", "history_id", "detections"})
                self.assertNotIn("code", response.data)
                self.assertEqual(response.data["question"], question)
                self.assertEqual(response.data["answer"], f"answer:{case}")
                self.assertEqual(response.data["detections"], [
                    {
                        "class_id": item.class_id,
                        "name": item.name,
                        "disease_name": item.disease_name,
                        "confidence": item.confidence,
                        "bbox_xyxy": list(item.bbox_xyxy),
                    }
                    for item in canonical
                ])
                self.assertEqual(mock_predict.call_args.kwargs["request_id"], request_id)
                self.assertEqual(mock_ask_image.call_args.kwargs["request_id"], request_id)
                self.assertEqual(mock_ask_image.call_args.kwargs["question"], question)
                self.assertEqual(mock_ask_image.call_args.kwargs["detections"], raw_detections)
