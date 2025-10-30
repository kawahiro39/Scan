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
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
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


def normalize_color_mode(mode: Optional[str]) -> str:
    """Validates and normalizes the requested color mode."""

    if not mode:
        return "mono"

    normalized = mode.lower()
    if normalized not in {"mono", "color"}:
        raise HTTPException(
            status_code=400,
            detail="color_mode must be either 'mono' or 'color'.",
        )

    return normalized


# Global variable for the AI model to avoid reloading it on every request
# It will be initialized on the first call to detect_document_with_ai
AI_MODEL = None

def detect_document_with_ai(img: np.ndarray) -> Optional[np.ndarray]:
    """Detects a document in an image using a semantic segmentation model."""
    global AI_MODEL

    # Initialize the model on the first run
    if AI_MODEL is None:
        try:
            # Using a lightweight and fast model. We can try larger models if accuracy is not enough.
            AI_MODEL = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1, # We are only detecting one class: document
            )
            AI_MODEL.eval()
            # If a GPU is available, use it.
            if torch.cuda.is_available():
                AI_MODEL.to('cuda')
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Pre-process the image
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert OpenCV BGR image to RGB PIL Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = AI_MODEL(input_tensor)

    # Process the output mask
    mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # Resize mask to original image size
    original_height, original_width = img.shape[:2]
    mask_resized = cv2.resize(mask, (original_width, original_height))

    # Threshold the mask to get a binary image
    _, binary_mask = cv2.threshold((mask_resized * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # Find the largest contour in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    # If the contour has 4 vertices, we assume it's our document
    if len(approx) == 4:
        # The points might not be in the correct order, so we order them
        return order_points(approx.reshape(4, 2).astype(np.float32))

    # If not exactly 4 points, try to find the minimum area rectangle
    # This can help with slightly irregular shapes
    try:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return order_points(box.astype(np.float32))
    except Exception:
         return None


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

    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edged = cv2.Canny(gray, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
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


def enhance_document_image(img: np.ndarray, mode: str = "mono") -> np.ndarray:
    """Enhances the warped document image based on the requested color mode."""

    normalized_mode = (mode or "mono").lower()

    if normalized_mode == "color":
        # Apply white balance using LAB color space and CLAHE on the luminance channel.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        balanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Perform a gentle contrast stretch to bring out colors.
        balanced_float = balanced.astype(np.float32)
        mean_color = balanced_float.mean(axis=(0, 1))
        mean = np.empty_like(balanced_float)
        mean[:] = mean_color
        enhanced_color = cv2.addWeighted(balanced_float, 1.15, mean, -0.15, 0)
        enhanced_color = np.clip(enhanced_color, 0, 255).astype(np.uint8)
        return enhanced_color

    # Default to the monochrome pipeline.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)

    denoised = cv2.fastNlMeansDenoising(equalized, None, 30, 7, 21)

    enhanced = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8,
    )

    enhanced = cv2.medianBlur(enhanced, 3)
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


def process_document(image_bytes: bytes, color_mode: str = "mono") -> tuple[bytes, str]:
    """
    Processes a document image using a single Google Cloud Vision API call.
    This function performs OCR, orientation correction, and perspective correction.

    Args:
        image_bytes: The raw bytes of the image file.
        color_mode: Processing mode, either "mono" for binarized output or "color" for
            color-preserving enhancement.

    Returns:
        A tuple containing:
        - The bytes of the perspective-corrected document image.
        - The extracted text (OCR) from the document.
    """
    normalized_mode = normalize_color_mode(color_mode)

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
            image = vision.Image(content=image_bytes)
            response = client.document_text_detection(
                image=image,
                image_context={"language_hints": ["ja", "en"]},
            )

            if getattr(response, "error", None) and response.error.message:
                logger.warning("Cloud Vision API error: %s", response.error.message)
            else:
                extracted_text = (
                    getattr(response, "full_text_annotation", None).text
                    if getattr(response, "full_text_annotation", None)
                    and response.full_text_annotation.text
                    else ""
                )

                if not extracted_text and response.text_annotations:
                    extracted_text = response.text_annotations[0].description

                document_points = None

                def _vertices_to_array(vertices: list) -> Optional[np.ndarray]:
                    if len(vertices) < 4:
                        return None
                    arr = np.array([[v.x, v.y] for v in vertices[:4]], dtype=np.float32)
                    if arr.shape != (4, 2):
                        return None
                    return arr

                if (
                    getattr(response, "full_text_annotation", None)
                    and response.full_text_annotation.pages
                ):
                    page = response.full_text_annotation.pages[0]
                    page_points = []
                    for block in page.blocks:
                        for vertex in block.bounding_box.vertices:
                            page_points.append([vertex.x, vertex.y])

                    if len(page_points) >= 4:
                        rect = cv2.minAreaRect(np.array(page_points, dtype=np.float32))
                        document_points = cv2.boxPoints(rect).astype(np.float32)

                if document_points is not None and document_points.shape != (4, 2):
                    document_points = None

                if document_points is None and response.text_annotations:
                    annotation = response.text_annotations[0]
                    if annotation.bounding_poly:
                        vertices = annotation.bounding_poly.vertices
                        candidate = _vertices_to_array(vertices)
                        if candidate is not None:
                            document_points = candidate

                if document_points is None:
                    crop_response = client.crop_hints(
                        image=image,
                        image_context={
                            "crop_hints_params": {
                                "aspect_ratios": [1.0]
                            }
                        },
                    )

                    crop_annotation = getattr(
                        crop_response, "crop_hints_annotation", None
                    )
                    crop_hints = (
                        getattr(crop_annotation, "crop_hints", None)
                        if crop_annotation
                        else None
                    )
                    if crop_hints:
                        crop_hint = crop_hints[0]
                        bounding_poly = getattr(crop_hint, "bounding_poly", None)
                        if bounding_poly and getattr(bounding_poly, "vertices", None):
                            candidate = _vertices_to_array(bounding_poly.vertices)
                            if candidate is not None:
                                document_points = candidate
        except (DefaultCredentialsError, GoogleAPICallError) as vision_error:
            message = str(vision_error)
            logger.warning("Cloud Vision API unavailable: %s", message)
            if "does not have permission to write logs" in message:
                ENABLE_CLOUD_VISION = False
        except Exception as vision_error:  # pragma: no cover - safeguard
            logger.warning("Unexpected Cloud Vision error: %s", vision_error)
            if "does not have permission to write logs" in str(vision_error):
                ENABLE_CLOUD_VISION = False

    # The result from Google Vision API should be retrieved before AI or contour detection
    # The existing complex logic for parsing Vision API response already populates `document_points`
    # if it finds valid geometry. We will respect that.

    # If Vision API did not provide points, try the AI model
    if document_points is None:
        document_points = detect_document_with_ai(img)

    # If both Vision and AI fail, fall back to OpenCV contour detection
    if document_points is None:
        document_points = detect_document_contour(img)

    # Final check and normalization of points
    if document_points is not None:
        document_points = document_points.astype(np.float32)
        height, width = img.shape[:2]
        document_points[:, 0] = np.clip(document_points[:, 0], 0, width - 1)
        document_points[:, 1] = np.clip(document_points[:, 1], 0, height - 1)
        document_points = order_points(document_points)

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
    enhanced = enhance_document_image(warped, normalized_mode)

    # Convert the corrected OpenCV image back to bytes.
    is_success, buffer = cv2.imencode(".png", enhanced)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode corrected image.")

    return buffer.tobytes(), extracted_text

def generate_pdf_response(image_bytes: bytes, color_mode: str = "mono") -> str:
    """
    Generates a PDF from the image, sizes it appropriately (A4 vs. business card),
    and returns it as a data URI containing a base64 encoded string. The PDF is
    rendered in grayscale for "mono" mode and RGB for "color" mode.
    """
    img = Image.open(io.BytesIO(image_bytes))
    normalized_mode = (color_mode or "mono").lower()

    if normalized_mode == "color":
        img = img.convert("RGB")
        canvas_mode = "RGB"
        background_color = "white"
    else:
        img = img.convert("L")
        canvas_mode = "L"
        background_color = 255

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
    canvas = Image.new(canvas_mode, (pdf_width, pdf_height), background_color)

    # Resize image to fit into the canvas while maintaining aspect ratio
    img.thumbnail((pdf_width, pdf_height), Image.Resampling.LANCZOS)

    # Paste the image onto the center of the canvas
    paste_x = (pdf_width - img.width) // 2
    paste_y = (pdf_height - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))

    canvas.save(pdf_buffer, 'PDF', resolution=100.0)
    pdf_buffer.seek(0)

    pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
    return f"data:application/pdf;base64,{pdf_base64}"

@app.post("/scan")
async def scan_document(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    color_mode: str = Form("mono"),
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

        normalized_color_mode = normalize_color_mode(color_mode)

        # Process the document
        corrected_image_bytes, extracted_text = process_document(
            image_bytes, normalized_color_mode
        )

        # Generate the PDF response from the corrected image
        pdf_data_uri = generate_pdf_response(
            corrected_image_bytes, normalized_color_mode
        )

        # Return the final response
        return {
            "extracted_text": extracted_text,
            "pdf_data_uri": pdf_data_uri
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
