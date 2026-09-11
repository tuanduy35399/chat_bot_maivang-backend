import json
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .services.contracts import ServiceFailure, YoloDetection
from .services.id_token import audience_from_service_url, fetch_google_id_token
from .services.rag_client import RagClient


class RagClientTests(SimpleTestCase):
    @patch("history.services.rag_client.requests.post")
    def test_ask_image_serializes_raw_detections_and_forwards_request_id(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "answer": "Câu trả lời",
            "detections": [{
                "class_id": 4,
                "name": "dom-la",
                "disease_name": "Đốm lá",
                "confidence": 0.89,
                "bbox_xyxy": [1, 2, 3, 4],
            }],
        }
        mock_post.return_value = response
        detection = YoloDetection(4, "dom-la", 0.89, (1.0, 2.0, 3.0, 4.0))

        result = RagClient(image_url="http://rag/image", timeout=3).ask_image(
            question="", history=[], detections=(detection,), request_id="req-1"
        )

        self.assertEqual(result.answer, "Câu trả lời")
        self.assertEqual(result.detections[0].name, "dom-la")
        self.assertEqual(result.detections[0].disease_name, "Đốm lá")
        self.assertEqual(result.detections[0].bbox_xyxy, (1.0, 2.0, 3.0, 4.0))
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["X-Request-ID"], "req-1")
        self.assertEqual(kwargs["data"]["question"], "")
        self.assertEqual(json.loads(kwargs["data"]["detections"]), [{
            "class_id": 4, "name": "dom-la", "confidence": 0.89,
            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
        }])

    @patch("history.services.rag_client.requests.post")
    def test_ask_image_hides_upstream_failure_body(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("secret upstream body")

        result = RagClient(image_url="http://rag/image", timeout=3).ask_image(
            question="", history=[], detections=(), request_id="req-1"
        )

        self.assertIsInstance(result, ServiceFailure)
        self.assertEqual(result.code, "rag_unavailable")
        self.assertNotIn("secret", result.message)


class IdTokenHelperTests(SimpleTestCase):
    def test_audience_from_service_url_strips_path(self):
        self.assertEqual(
            audience_from_service_url("https://maivang-rag-xyz.run.app/chat"),
            "https://maivang-rag-xyz.run.app",
        )
        self.assertEqual(
            audience_from_service_url("https://maivang-rag-xyz.run.app/chat/image"),
            "https://maivang-rag-xyz.run.app",
        )

    @patch("history.services.id_token.id_token.fetch_id_token", return_value="tok-123")
    def test_fetch_google_id_token_passes_audience(self, mock_fetch):
        token = fetch_google_id_token("https://maivang-rag-xyz.run.app")
        self.assertEqual(token, "tok-123")
        self.assertEqual(mock_fetch.call_args.args[1], "https://maivang-rag-xyz.run.app")
