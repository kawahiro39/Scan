# Document Scan API

This is an API for scanning documents from images. It uses Google Cloud Vision and OpenCV to automatically correct the image orientation and perspective, perform OCR to extract text, and return a corrected PDF.

## API Usage

### Endpoint

- `POST /scan`

### Request Body

The API expects a JSON payload with a single key:

- `image_base64` (string): The base64-encoded string of the image you want to process.

**Example JSON Body:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
}
```

### Response Body

On success, the API returns a JSON object with the following keys:

- `extracted_text` (string): The full text extracted from the document.
- `pdf_base64` (string): A base64-encoded string representing the corrected and cropped document in PDF format.

**Example JSON Response:**
```json
{
  "extracted_text": "This is a sample text extracted from the document.",
  "pdf_base64": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFI..."
}
```

### Example Usage with cURL

You can use `cURL` to test the API. Make sure to replace `[YOUR_BASE64_STRING]` with the actual base64-encoded string of your image.

```bash
curl -X POST "http://localhost:8080/scan" \
-H "Content-Type: application/json" \
-d '{
  "image_base64": "[YOUR_BASE64_STRING]"
}'
```

To get the base64 string of an image file (e.g., `my_document.png`) on macOS or Linux, you can use the following command:

```bash
base64 -i my_document.png
```

### Error Handling

- If the request body is invalid or the base64 string is malformed, the API will return an `HTTP 400 Bad Request` error.
- If a document cannot be found in the image, the API will return an `HTTP 400 Bad Request` with the detail: `"書類が見つかりません"` (Document not found).
- If an unexpected server error occurs, the API will return an `HTTP 500 Internal Server Error`.
