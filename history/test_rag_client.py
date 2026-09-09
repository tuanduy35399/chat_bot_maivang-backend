import json
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .services.contracts import ServiceFailure, YoloDetection
from .services.rag_client import RagClient


class RagClientTests(SimpleTestCase):
    @patch("history.services.rag_client.requests.post")
    def test_ask_image_serializes_raw_detections_and_forwards_request_id(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"answer": "Câu trả lời"}
        mock_post.return_value = response
        detection = YoloDetection(4, "dom-la", 0.89, (1.0, 2.0, 3.0, 4.0))

        result = RagClient(image_url="http://rag/image", timeout=3).ask_image(
            question="", history=[], detections=(detection,), request_id="req-1"
        )

        self.assertEqual(result.answer, "Câu trả lời")
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
