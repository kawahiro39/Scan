from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import math
from google.cloud import vision
from PIL import Image
import cv2
import numpy as np
import requests
from typing import Optional

# Pydanticモデルを定義してリクエストボディの型を検証
class ScanRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(
    title="Document Scan API",
    description="An API to scan documents using Google Cloud Vision and OpenCV.",
    version="0.1.0"
)

def process_document(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Processes a document image using a single Google Cloud Vision API call.
    This function performs OCR, orientation correction, and perspective correction.

    Args:
        image_bytes: The raw bytes of the image file.

    Returns:
        A tuple containing:
        - The bytes of the perspective-corrected document image.
        - The extracted text (OCR) from the document.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    # Use document_text_detection to get both text and document boundaries in one call.
    response = client.document_text_detection(image=image)

    if not response.text_annotations:
        raise HTTPException(status_code=400, detail="書類が見つかりません")

    # The first annotation is the entire text block, which gives us the OCR text
    # and the bounding box for the entire document.
    annotation = response.text_annotations[0]
    extracted_text = annotation.description
    document_bounds = annotation.bounding_poly.vertices

    # Convert the original image bytes to an OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # The vertices from the Vision API are absolute pixel coordinates.
    src_pts = np.array([[v.x, v.y] for v in document_bounds], dtype=np.float32)

    # Define the destination rectangle's dimensions based on the bounding box.
    # This automatically handles orientation.
    width_top = np.sqrt(((src_pts[0][0] - src_pts[1][0]) ** 2) + ((src_pts[0][1] - src_pts[1][1]) ** 2))
    width_bottom = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
    height_right = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Compute the perspective transform matrix and apply it to get the top-down view.
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))

    # Convert the corrected OpenCV image back to bytes.
    is_success, buffer = cv2.imencode(".png", warped)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode corrected image.")

    return buffer.tobytes(), extracted_text

def generate_pdf_response(image_bytes: bytes) -> str:
    """
    Generates a PDF from the image, sizes it appropriately (A4 vs. business card),
    and returns it as a base64 encoded string.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    # A4 dimensions in points (72 dpi)
    A4_WIDTH, A4_HEIGHT = 595, 842
    # Business card dimensions (approx)
    CARD_WIDTH, CARD_HEIGHT = 252, 144 # 3.5 x 2 inches

    aspect_ratio = img_width / img_height
    a4_aspect_ratio = A4_WIDTH / A4_HEIGHT

    # Determine if the image is closer to A4 or business card aspect ratio
    # This is a heuristic and might need refinement
    if abs(aspect_ratio - a4_aspect_ratio) < abs(aspect_ratio - (CARD_WIDTH / CARD_HEIGHT)):
        # Treat as A4
        pdf_width, pdf_height = A4_WIDTH, A4_HEIGHT
    else:
        # Treat as Business Card
        pdf_width, pdf_height = CARD_WIDTH, CARD_HEIGHT

    # Create a new PDF with the same dimensions as the target size
    pdf_buffer = io.BytesIO()
    canvas = Image.new('RGB', (pdf_width, pdf_height), 'white')

    # Resize image to fit into the canvas while maintaining aspect ratio
    img.thumbnail((pdf_width, pdf_height), Image.Resampling.LANCZOS)

    # Paste the image onto the center of the canvas
    paste_x = (pdf_width - img.width) // 2
    paste_y = (pdf_height - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))

    canvas.save(pdf_buffer, 'PDF', resolution=100.0)
    pdf_buffer.seek(0)

    pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
    return pdf_base64

@app.post("/scan")
async def scan_document(request: ScanRequest):
    """
    Receives an image via base64 string or URL, processes it, and returns the result.
    """
    try:
        # Manual validation of input fields
        if (request.image_base64 and request.image_url) or \
           (not request.image_base64 and not request.image_url):
            raise HTTPException(
                status_code=422,
                detail='Exactly one of "image_base64" or "image_url" must be provided.'
            )

        image_bytes = None
        if request.image_base64:
            try:
                image_bytes = base64.b64decode(request.image_base64)
            except base64.binascii.Error:
                raise HTTPException(status_code=400, detail="Invalid base64 string.")

        elif request.image_url:
            try:
                response = requests.get(request.image_url, stream=True)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                image_bytes = response.content
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")

        # The Pydantic validator ensures that image_bytes will be set.

        # Process the document in a single, efficient step
        corrected_image_bytes, extracted_text = process_document(image_bytes)

        # Generate the PDF response from the corrected image
        pdf_base64 = generate_pdf_response(corrected_image_bytes)

        # Return the final response
        return {
            "extracted_text": extracted_text,
            "pdf_base64": pdf_base64
        }

    except HTTPException as e:
        # Re-raise HTTPException to let FastAPI handle it
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Scan API"}
