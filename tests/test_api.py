from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
import base64

client = TestClient(app)

def test_scan_endpoint_returns_correct_keys():
    """
    Tests that the /scan endpoint returns a JSON response with the keys
    'extracted_text' and 'pdf_base64'.
    """
    fake_image_bytes = b"fakeimagedata"
    fake_corrected_bytes = b"correctedimagedata"
    fake_pdf_uri = f"data:application/pdf;base64,{base64.b64encode(b'fakepdf').decode('utf-8')}"

    with patch("main.fetch_image_from_url", return_value=fake_image_bytes), \
         patch("main.process_document", return_value=(fake_corrected_bytes, "extracted text")), \
         patch("main.generate_pdf_response", return_value=fake_pdf_uri):

        response = client.post(
            "/scan",
            data={"image_url": "http://fakeurl.com/image.jpg", "color_mode": "mono"}
        )

        assert response.status_code == 200
        json_data = response.json()
        assert "extracted_text" in json_data
        assert "pdf_base64" in json_data
        assert "pdf_data_uri" not in json_data
        assert json_data["pdf_base64"] == fake_pdf_uri
