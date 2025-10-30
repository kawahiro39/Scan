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
    def __init__(self) -> None:
        self.error = SimpleNamespace(message="")
        self.full_text_annotation = SimpleNamespace(text="", pages=[])
        self.text_annotations = []


class FakeVisionClient:
    def __init__(self, crop_vertices: list[tuple[float, float]] | None) -> None:
        self.crop_vertices = crop_vertices
        self.crop_called = False
        self.last_crop_context = None

    def document_text_detection(self, image, image_context=None):
        return FakeDocResponse()

    def crop_hints(self, image, image_context=None):
        self.crop_called = True
        self.last_crop_context = image_context

        if not self.crop_vertices or len(self.crop_vertices) < 4:
            return SimpleNamespace(crop_hints_annotation=SimpleNamespace(crop_hints=[]))

        vertices = [SimpleNamespace(x=x, y=y) for x, y in self.crop_vertices]
        bounding_poly = SimpleNamespace(vertices=vertices)
        crop_hint = SimpleNamespace(bounding_poly=bounding_poly)
        crop_hints_annotation = SimpleNamespace(crop_hints=[crop_hint])
        return SimpleNamespace(crop_hints_annotation=crop_hints_annotation)


class ProcessDocumentCropHintsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_bytes = _encode_blank_image()

    def _patch_vision(self, client: FakeVisionClient):
        fake_vision = SimpleNamespace(
            ImageAnnotatorClient=lambda: client,
            Image=lambda content: SimpleNamespace(content=content),
        )
        return patch.object(main, "vision", fake_vision)

    def test_crop_hint_polygon_used_before_contour(self):
        crop_vertices = [
            (-10.0, -5.0),
            (120.0, -3.0),
            (130.0, 130.0),
            (-25.0, 140.0),
        ]
        client = FakeVisionClient(crop_vertices)

        with self._patch_vision(client), patch("main.ENABLE_CLOUD_VISION", True), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as detect_mock, patch("main.four_point_transform") as transform_mock:
            ai_detect_mock.return_value = None  # Simulate AI failing
            detect_mock.return_value = None
            transform_mock.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            processed_bytes, extracted_text = main.process_document(self.image_bytes)

        self.assertIsInstance(processed_bytes, bytes)
        self.assertEqual(extracted_text, "")
        self.assertTrue(client.crop_called)
        self.assertEqual(
            client.last_crop_context,
            {"crop_hints_params": {"aspect_ratios": [1.0]}},
        )
        detect_mock.assert_not_called()
        transform_mock.assert_called_once()

        _, points = transform_mock.call_args[0]
        expected_points = np.array(
            [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(points, expected_points)

    def test_crop_hint_fallback_to_contour_when_invalid(self):
        crop_vertices = [
            (10.0, 10.0),
            (80.0, 15.0),
            (75.0, 80.0),
        ]
        client = FakeVisionClient(crop_vertices)

        with self._patch_vision(client), patch("main.ENABLE_CLOUD_VISION", True), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as detect_mock, patch("main.four_point_transform") as transform_mock:
            ai_detect_mock.return_value = None  # Simulate AI failing
            detect_mock.return_value = np.array(
                [[1.0, 1.0], [98.0, 2.0], [97.0, 97.0], [2.0, 98.0]], dtype=np.float32
            )
            transform_mock.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            processed_bytes, extracted_text = main.process_document(self.image_bytes)

        self.assertIsInstance(processed_bytes, bytes)
        self.assertEqual(extracted_text, "")
        self.assertTrue(client.crop_called)
        detect_mock.assert_called_once()
        transform_mock.assert_called_once()

    def test_color_mode_enhancement_completes_without_cv2_error(self):
        with patch("main.ENABLE_CLOUD_VISION", False), patch(
            "main.detect_document_with_ai"
        ) as ai_detect_mock, patch(
            "main.detect_document_contour"
        ) as detect_mock, patch("main.four_point_transform") as transform_mock:
            ai_detect_mock.return_value = None  # Simulate AI failing
            detect_mock.return_value = np.array(
                [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]], dtype=np.float32
            )
            transform_mock.return_value = np.full((32, 32, 3), 180, dtype=np.uint8)

            processed_bytes, extracted_text = main.process_document(
                self.image_bytes, color_mode="color"
            )

        self.assertIsInstance(processed_bytes, bytes)
        self.assertEqual(extracted_text, "")
        detect_mock.assert_called_once()
        transform_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
