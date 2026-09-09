from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .services.contracts import ServiceFailure
from .services.yolo_client import YoloClient


class YoloClientTests(SimpleTestCase):
    @patch("history.services.yolo_client.requests.post")
    def test_predict_preserves_every_box_and_forwards_request_id(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "images": [{"results": [
                {"class_id": 4, "name": "dom-la", "confidence": 0.89,
                 "bbox_xyxy": [1, 2, 3, 4]},
                {"class_id": 4, "name": "dom-la", "confidence": 0.42,
                 "bbox_xyxy": [5, 6, 7, 8]},
            ]}]
        }
        mock_post.return_value = response

        result = YoloClient(url="http://yolo", timeout=3).predict(
            image_bytes=b"x", filename="leaf.jpg", content_type="image/jpeg",
            request_id="req-1",
        )

        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].bbox_xyxy, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(result.detections[1].confidence, 0.42)
        self.assertEqual(mock_post.call_args.kwargs["headers"]["X-Request-ID"], "req-1")

    @patch("history.services.yolo_client.requests.post")
    def test_predict_maps_invalid_json_to_safe_error(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("secret upstream body")
        mock_post.return_value = response

        result = YoloClient(url="http://yolo", timeout=3).predict(
            image_bytes=b"x", filename="leaf.jpg", content_type="image/jpeg",
            request_id="req-1",
        )

        self.assertIsInstance(result, ServiceFailure)
        self.assertEqual(result.code, "yolo_invalid_response")
        self.assertEqual(result.request_id, "req-1")
        self.assertNotIn("secret", result.message)
