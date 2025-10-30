import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import main

# Mock the torch and smp modules to avoid actual library dependencies for this test
torch_mock = MagicMock()
torch_mock.sigmoid.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = np.ones((512, 512), dtype=np.float32)

class TestAIDetection(unittest.TestCase):

    def setUp(self):
        # Create a dummy blank image for input
        self.blank_image = np.zeros((800, 600, 3), dtype=np.uint8)

    @patch('main.torch', new=torch_mock)
    @patch('main.smp', new=MagicMock())
    @patch('main.T', new=MagicMock())
    def test_detect_document_with_ai_success(self):
        """
        Test that AI detection function correctly processes a mocked model output
        and returns valid document points.
        """
        # --- Mocking Setup ---
        # Configure the mocked model and its output
        mock_model = MagicMock()
        # Create a sample mask: a white square in the middle of a black background
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[100:400, 100:400] = 1.0

        # Configure the series of mock calls to produce this mask
        mock_output_tensor = MagicMock()
        mock_sigmoid_tensor = MagicMock()
        mock_squeeze_tensor = MagicMock()
        mock_cpu_tensor = MagicMock()

        mock_model.return_value = mock_output_tensor
        torch_mock.sigmoid.return_value = mock_sigmoid_tensor
        mock_sigmoid_tensor.squeeze.return_value = mock_squeeze_tensor
        mock_squeeze_tensor.cpu.return_value = mock_cpu_tensor
        mock_cpu_tensor.numpy.return_value = mask

        main.AI_MODEL = mock_model

        # --- Execution ---
        points = main.detect_document_with_ai(self.blank_image)

        # --- Assertions ---
        self.assertIsNotNone(points)
        self.assertEqual(points.shape, (4, 2))

        # We expect the points to be an approximation of the square we defined in the mask,
        # scaled to the original image size.
        # The exact values depend on cv2.minAreaRect, so we check for plausibility.
        self.assertTrue(np.all(points >= 0))
        # Check that x-coordinates are within image width (600) and y-coords within height (800)
        self.assertTrue(np.all(points[:, 0] < 600))
        self.assertTrue(np.all(points[:, 1] < 800))


    def test_detect_document_with_ai_model_unavailable(self):
        """
        Test that the function returns None if the AI model is not loaded.
        """
        main.AI_MODEL = None
        points = main.detect_document_with_ai(self.blank_image)
        self.assertIsNone(points)

    def test_detect_document_with_ai_no_contours_found(self):
        """
        Test that the function returns None if the model produces an empty mask.
        """
        mock_model = MagicMock()
        # Produce an all-black (empty) mask
        mask = np.zeros((512, 512), dtype=np.float32)

        mock_output_tensor = MagicMock()
        mock_sigmoid_tensor = MagicMock()
        mock_squeeze_tensor = MagicMock()
        mock_cpu_tensor = MagicMock()

        mock_model.return_value = mock_output_tensor
        torch_mock.sigmoid.return_value = mock_sigmoid_tensor
        mock_sigmoid_tensor.squeeze.return_value = mock_squeeze_tensor
        mock_squeeze_tensor.cpu.return_value = mock_cpu_tensor
        mock_cpu_tensor.numpy.return_value = mask

        main.AI_MODEL = mock_model

        points = main.detect_document_with_ai(self.blank_image)
        self.assertIsNone(points)

if __name__ == "__main__":
    unittest.main()
