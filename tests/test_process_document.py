from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

import main


def _create_test_image_bytes() -> bytes:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (90, 90), (255, 255, 255), -1)
    success, buffer = cv2.imencode(".png", image)
    if not success:  # pragma: no cover - safeguard
        raise RuntimeError("Failed to encode test image")
    return buffer.tobytes()


class ProcessDocumentCropHintTests(TestCase):
    def setUp(self) -> None:
        self.image_bytes = _create_test_image_bytes()
        main.ENABLE_CLOUD_VISION = True

    def test_crop_hint_polygon_used_before_contour(self) -> None:
        crop_vertices = [
            SimpleNamespace(x=80, y=10),
            SimpleNamespace(x=90, y=90),
            SimpleNamespace(x=10, y=90),
            SimpleNamespace(x=10, y=10),
        ]

        crop_hint = SimpleNamespace(
            bounding_poly=SimpleNamespace(vertices=crop_vertices)
        )

        document_response = SimpleNamespace(
            error=SimpleNamespace(message=""),
            full_text_annotation=SimpleNamespace(text="", pages=[]),
            text_annotations=[],
        )

        crop_response = SimpleNamespace(
            error=SimpleNamespace(message=""),
            crop_hints_annotation=SimpleNamespace(crop_hints=[crop_hint]),
        )

        class FakeClient:
            def document_text_detection(self, image, image_context):
                return document_response

            def crop_hints(self, image, image_context):
                return crop_response

        fake_vision = SimpleNamespace(
            ImageAnnotatorClient=lambda: FakeClient(),
            Image=lambda content=None: SimpleNamespace(content=content),
        )

        with patch.object(main, "vision", fake_vision), patch(
            "main.detect_document_contour"
        ) as mock_contour, patch("main.four_point_transform") as mock_transform:
            mock_contour.side_effect = AssertionError("Contour fallback should not run")

            def _fake_transform(image, points):
                self.assertEqual(points.shape, (4, 2))
                np.testing.assert_array_equal(
                    points,
                    np.array(
                        [[10.0, 10.0], [80.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
                        dtype=np.float32,
                    ),
                )
                return np.zeros((10, 10, 3), dtype=np.uint8)

            mock_transform.side_effect = _fake_transform

            result_bytes, extracted_text = main.process_document(self.image_bytes)

            self.assertTrue(result_bytes)
            self.assertEqual(extracted_text, "")
            mock_transform.assert_called_once()

    def test_crop_hint_error_falls_back_to_contour(self) -> None:
        document_response = SimpleNamespace(
            error=SimpleNamespace(message=""),
            full_text_annotation=SimpleNamespace(text="", pages=[]),
            text_annotations=[],
        )

        class FakeClient:
            def document_text_detection(self, image, image_context):
                return document_response

            def crop_hints(self, image, image_context):
                raise main.GoogleAPICallError("test failure")

        crop_polygon = np.array(
            [[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32
        )

        fake_vision = SimpleNamespace(
            ImageAnnotatorClient=lambda: FakeClient(),
            Image=lambda content=None: SimpleNamespace(content=content),
        )

        with patch.object(main, "vision", fake_vision), patch(
            "main.detect_document_contour", return_value=crop_polygon
        ) as mock_contour, patch("main.four_point_transform") as mock_transform:
            mock_transform.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

            result_bytes, _ = main.process_document(self.image_bytes)

            self.assertTrue(result_bytes)
            mock_contour.assert_called_once()
            mock_transform.assert_called_once()
