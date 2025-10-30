from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import base64
import io
import logging
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import requests
from requests import RequestException

try:
    from google.api_core.exceptions import GoogleAPICallError
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import vision
except ImportError:  # pragma: no cover - optional dependency
    vision = None  # type: ignore
    GoogleAPICallError = DefaultCredentialsError = Exception  # type: ignore

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(
    title="Document Scan API",
    description="An API to scan documents using Google Cloud Vision and OpenCV.",
    version="0.1.0"
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

ENABLE_CLOUD_VISION = (
    vision is not None
    and os.getenv("ENABLE_CLOUD_VISION", "true").lower() in {"1", "true", "yes"}
)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders four points in the order: top-left, top-right, bottom-right, bottom-left."""

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_document_contour(img: np.ndarray) -> Optional[np.ndarray]:
    """Detects the largest 4-point contour that likely represents the document."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2).astype(np.float32))

    return None


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies a perspective transform using four source points."""

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom), 1)

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = max(int(height_right), int(height_left), 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def enhance_document_image(img: np.ndarray) -> np.ndarray:
    """Improves contrast and whiteness to emulate a scanned document."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_color


def fetch_image_from_url(image_url: str) -> bytes:
    """Downloads an image from the provided URL and returns its bytes."""

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
    except RequestException as exc:  # pragma: no cover - network errors
        raise HTTPException(
            status_code=400,
            detail=f"画像のダウンロードに失敗しました: {exc}",
        ) from exc

    if not response.content:
        raise HTTPException(status_code=400, detail="ダウンロードした画像が空です。")

    return response.content


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
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした。")

    extracted_text = ""
    document_points: Optional[np.ndarray] = None

    global ENABLE_CLOUD_VISION

    if ENABLE_CLOUD_VISION:
        try:
            client = vision.ImageAnnotatorClient()
            response = client.document_text_detection(image=vision.Image(content=image_bytes))

            if getattr(response, "error", None) and response.error.message:
                logger.warning("Cloud Vision API error: %s", response.error.message)
            elif response.text_annotations:
                annotation = response.text_annotations[0]
                extracted_text = annotation.description
                vertices = annotation.bounding_poly.vertices
                if len(vertices) >= 4:
                    document_points = np.array(
                        [[v.x, v.y] for v in vertices[:4]],
                        dtype=np.float32,
                    )
        except (DefaultCredentialsError, GoogleAPICallError) as vision_error:
            message = str(vision_error)
            logger.warning("Cloud Vision API unavailable: %s", message)
            if "does not have permission to write logs" in message:
                ENABLE_CLOUD_VISION = False
        except Exception as vision_error:  # pragma: no cover - safeguard
            logger.warning("Unexpected Cloud Vision error: %s", vision_error)
            if "does not have permission to write logs" in str(vision_error):
                ENABLE_CLOUD_VISION = False

    if document_points is not None:
        document_points = order_points(document_points)
    else:
        document_points = detect_document_contour(img)

    if document_points is None:
        height, width = img.shape[:2]
        document_points = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

    warped = four_point_transform(img, document_points)
    enhanced = enhance_document_image(warped)

    # Convert the corrected OpenCV image back to bytes.
    is_success, buffer = cv2.imencode(".png", enhanced)
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
async def scan_document(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """
    Receives an image file via Form-data, processes it, and returns the result.
    """
    try:
        if file is not None:
            # Read the uploaded file into memory
            image_bytes = await file.read()

            if not image_bytes:
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        elif image_url:
            image_bytes = fetch_image_from_url(image_url)
        else:
            raise HTTPException(
                status_code=400,
                detail="A file or image_url parameter must be provided.",
            )

        # Process the document
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
