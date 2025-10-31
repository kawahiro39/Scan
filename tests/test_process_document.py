import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main


def _encode_blank_image(width: int = 100, height: int = 100) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = 255
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode test image")
    return io.BytesIO(buffer.tobytes()).getvalue()


class FakeDocResponse:
    def __init__(self, pages=None) -> None:
        self.error = SimpleNamespace(message="")
        self.full_text_annotation = SimpleNamespace(text="", pages=pages or [])
        self.text_annotations = []


class FakeVisionClient:
    def __init__(self, doc_vertices: list[tuple[float, float]] | None) -> None:
        self.doc_vertices = doc_vertices
        self.doc_text_detection_called = False

    def document_text_detection(self, image, image_context=None):
        self.doc_text_detection_called = True
        if not self.doc_vertices or len(self.doc_vertices) != 4:
            return FakeDocResponse(pages=[])  # Simulate no pages found

        vertices = [SimpleNamespace(x=x, y=y) for x, y in self.doc_vertices]
        bounding_box = SimpleNamespace(vertices=vertices)
        page = SimpleNamespace(bounding_box=bounding_box)
        return FakeDocResponse(pages=[page])


class ProcessDocumentDetectionLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_bytes = _encode_blank_image()

    def _patch_vision(self, client: FakeVisionClient):
        fake_vision = SimpleNamespace(
            ImageAnnotatorClient=lambda: client,
            Image=lambda content: SimpleNamespace(content=content),
        )
        return patch.object(main, "vision", fake_vision)

    def test_vision_api_is_used_when_successful(self):
        # Test that if Vision API succeeds, it's used immediately
        vision_api_points = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
        client = FakeVisionClient(vision_api_points)

        with self._patch_vision(client), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as contour_mock, patch("main.four_point_transform") as transform_mock:
            transform_mock.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            main.process_document(self.image_bytes)

            self.assertTrue(client.doc_text_detection_called)
            ai_detect_mock.assert_not_called()
            contour_mock.assert_not_called()
            transform_mock.assert_called_once()
            _, passed_points = transform_mock.call_args[0]
            expected_points = np.array(vision_api_points, dtype=np.float32)
            np.testing.assert_array_equal(
                passed_points, main.order_points(expected_points)
            )

    def test_ai_model_is_used_when_vision_api_fails(self):
        # Test fallback to AI model when Vision API fails
        client = FakeVisionClient(None)  # Vision API fails

        with self._patch_vision(client), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as contour_mock, patch("main.four_point_transform") as transform_mock:
            ai_mock_points = np.array(
                [[5.0, 5.0], [95.0, 5.0], [95.0, 95.0], [5.0, 95.0]], dtype=np.float32
            )
            ai_detect_mock.return_value = ai_mock_points
            transform_mock.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            main.process_document(self.image_bytes)

            self.assertTrue(client.doc_text_detection_called)
            ai_detect_mock.assert_called_once()
            contour_mock.assert_not_called()
            transform_mock.assert_called_once()
            _, passed_points = transform_mock.call_args[0]
            np.testing.assert_array_equal(passed_points, ai_mock_points)

    def test_opencv_is_used_when_vision_api_and_ai_fail(self):
        # Test fallback to OpenCV when both Vision API and AI fail
        client = FakeVisionClient(None)  # Vision API fails

        with self._patch_vision(client), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as contour_mock, patch("main.four_point_transform") as transform_mock:
            ai_detect_mock.return_value = None  # AI fails
            opencv_points = np.array(
                [[2.0, 2.0], [98.0, 2.0], [98.0, 98.0], [2.0, 98.0]], dtype=np.float32
            )
            contour_mock.return_value = opencv_points
            transform_mock.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            main.process_document(self.image_bytes)

            self.assertTrue(client.doc_text_detection_called)
            ai_detect_mock.assert_called_once()
            contour_mock.assert_called_once()
            transform_mock.assert_called_once()
            _, passed_points = transform_mock.call_args[0]
            np.testing.assert_array_equal(passed_points, opencv_points)

    def test_color_mode_enhancement_completes_without_cv2_error(self):
        with patch("main.ENABLE_CLOUD_VISION", False), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as detect_mock, patch("main.four_point_transform") as transform_mock:
            ai_detect_mock.return_value = None
            detect_mock.return_value = np.array(
                [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]], dtype=np.float32
            )
            transform_mock.return_value = np.full((32, 32, 3), 180, dtype=np.uint8)

            processed_bytes, _ = main.process_document(
                self.image_bytes, color_mode="color"
            )

            self.assertIsInstance(processed_bytes, bytes)
            detect_mock.assert_called_once()
            transform_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
