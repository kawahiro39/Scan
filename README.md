# Document Scan API

This is an API for scanning documents from images. It uses Google Cloud Vision and OpenCV to automatically correct the image orientation and perspective, perform OCR to extract text, and return a corrected PDF.

## API Usage

### Endpoint

- `POST /scan`

### Request Body

The API expects the request body to be in `multipart/form-data` format. You must provide either a file upload or an image URL.

- `file` (File, optional): The image file of the document to process.
- `image_url` (string, optional): A publicly accessible URL pointing to the image to process.

At least one of the parameters must be supplied. When both are provided, the uploaded file takes precedence.

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

You can use `cURL` to test the API with a local file.

```bash
curl -X POST "http://localhost:8080/scan" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/document.jpg"
```

### Error Handling

- If neither a valid file nor an `image_url` is provided, or the uploaded file is empty, the API will return an `HTTP 400 Bad Request` error.
- If a document cannot be found in the image, the API will return an `HTTP 400 Bad Request` with the detail: `"書類が見つかりません"` (Document not found).
- If an unexpected server error occurs, the API will return an `HTTP 500 Internal Server Error`.
