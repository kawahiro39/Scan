from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import math
from google.cloud import vision
from PIL import Image
import cv2
import numpy as np

# Pydanticモデルを定義してリクエストボディの型を検証
class ScanRequest(BaseModel):
    image_base64: str

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(
    title="Document Scan API",
    description="An API to scan documents using Google Cloud Vision and OpenCV.",
    version="0.1.0"
)

def correct_image_orientation(image_bytes: bytes) -> bytes:
    """
    Corrects the orientation of an image based on text detection from Google Cloud Vision API.

    Args:
        image_bytes: The raw bytes of the image.

    Returns:
        The bytes of the orientation-corrected image.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    # Detect text properties to determine orientation
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        # The first text annotation is the full text block
        # Its bounding poly gives information about the page orientation
        page_orientation = texts[0].bounding_poly

        # Determine the rotation angle
        # The first annotation is the entire detected text block.
        # We can calculate the orientation of this block to correct for page rotation.
        vertices = page_orientation.vertices
        if len(vertices) == 4:
            # The vector from top-left to top-right vertex defines the orientation
            dy = vertices[1].y - vertices[0].y
            dx = vertices[1].x - vertices[0].x

            # Calculate the angle of this vector. This handles both minor skew
            # and major rotations (90, 180, 270 degrees).
            angle = math.degrees(math.atan2(dy, dx))

            # Only perform rotation if the angle is significant enough to indicate
            # that the image is not upright.
            if abs(angle) > 1: # Threshold of 1 degree
                img = Image.open(io.BytesIO(image_bytes))
                # Pillow's rotate function rotates counter-clockwise.
                # A negative angle will correct the clockwise tilt.
                rotated_img = img.rotate(-angle, expand=True)

                output_buffer = io.BytesIO()
                rotated_img.save(output_buffer, format=img.format or "PNG")
                return output_buffer.getvalue()

    return image_bytes

def detect_and_correct_perspective(image_bytes: bytes) -> bytes:
    """
    Detects a document in the image and corrects its perspective.
    Returns the corrected image bytes. If no document is found, raises an HTTPException.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Find the object with the largest area, assuming it's the document.
    largest_area = 0
    document_bounds = None

    img_for_size = Image.open(io.BytesIO(image_bytes))
    w, h = img_for_size.size

    for obj in objects:
        if len(obj.bounding_poly.normalized_vertices) == 4:
            # Calculate the area of the polygon to find the largest one
            poly = obj.bounding_poly.normalized_vertices
            # Shoelace formula to calculate polygon area
            area = 0.5 * abs(sum(poly[i].x * w * (poly[(i + 1) % 4].y * h) - poly[(i + 1) % 4].x * w * (poly[i].y * h) for i in range(4)))

            if area > largest_area:
                largest_area = area
                document_bounds = poly

    if not document_bounds:
        raise HTTPException(status_code=400, detail="書類が見つかりません")

    # Convert the image bytes to an OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # Get source points from Vision API response
    src_pts = np.array([[v.x * w, v.y * h] for v in document_bounds], dtype=np.float32)

    # Define the destination points (a rectangle)
    # Get the width and height of the new image
    width_ad = np.sqrt(((src_pts[0][0] - src_pts[1][0]) ** 2) + ((src_pts[0][1] - src_pts[1][1]) ** 2))
    width_bc = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
    max_width = max(int(width_ad), int(width_bc))

    height_ab = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
    height_cd = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
    max_height = max(int(height_ab), int(height_cd))

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Get the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))

    # Convert the corrected image back to bytes
    is_success, buffer = cv2.imencode(".png", warped)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode corrected image.")

    return buffer.tobytes()

def extract_text(image_bytes: bytes) -> str:
    """
    Performs OCR on the given image bytes and returns the extracted text.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""

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
    Receives a base64 encoded image, processes it, and returns the result.
    Processing steps (to be implemented):
    1. Decode base64 image.
    2. Correct image orientation.
    3. Detect document and apply perspective correction.
    4. Perform OCR to extract text.
    5. Generate a PDF file.
    6. Return the PDF (base64) and the extracted text.
    """
    try:
        # Base64文字列が正しいかどうかの簡易的なチェック
        if not request.image_base64 or len(request.image_base64) % 4 != 0:
            raise HTTPException(status_code=400, detail="Invalid base64 string provided.")

        # Base64デコード試行
        image_bytes = base64.b64decode(request.image_base64)

        # Step 2: Correct image orientation
        corrected_image_bytes = correct_image_orientation(image_bytes)

        # Step 3: Detect document and apply perspective correction
        corrected_perspective_bytes = detect_and_correct_perspective(corrected_image_bytes)

        # Step 4: Perform OCR to extract text
        extracted_text = extract_text(corrected_perspective_bytes)

        # Step 5: Generate a PDF file
        pdf_base64 = generate_pdf_response(corrected_perspective_bytes)

        # Step 6: Return the final response
        return {
            "extracted_text": extracted_text,
            "pdf_base64": pdf_base64
        }

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 string.")
    except Exception as e:
        # 予期せぬエラーのハンドリング
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Scan API"}
