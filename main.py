"""
Document Scanner API
FastAPI application for scanning documents and generating PDFs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import cv2
import numpy as np
import io
from datetime import datetime

from document_scanner import DocumentScanner
from pdf_generator import PDFGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Document Scanner API",
    description="API for automatic document boundary detection, cropping, and PDF generation",
    version="1.0.0"
)

# Initialize scanner and PDF generator
scanner = DocumentScanner()
pdf_generator = PDFGenerator()


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": "Document Scanner API",
        "version": "1.0.0",
        "endpoints": {
            "/scan": "POST - Scan document image and return as PDF",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """
    Health check endpoint for Cloud Run
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/scan")
async def scan_document(
    file: UploadFile = File(..., description="Image file to scan (JPEG, PNG, etc.)"),
    enhance: Optional[bool] = Form(True, description="Apply image enhancement"),
    page_size: Optional[str] = Form("A4", description="PDF page size (A4 or letter)")
):
    """
    Scan a document image, detect boundaries, crop, and return as PDF
    
    Args:
        file: Uploaded image file
        enhance: Whether to enhance the image (default: True)
        page_size: PDF page size - "A4" or "letter" (default: "A4")
        
    Returns:
        PDF file with scanned document
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure the file is a valid image."
            )
        
        # Scan the document
        scanned_image, status_message = scanner.scan(image, enhance=enhance)
        
        if scanned_image is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to scan document: {status_message}"
            )
        
        # Create PDF generator with specified page size
        pdf_gen = PDFGenerator(page_size=page_size)
        
        # Generate PDF
        pdf_buffer = io.BytesIO()
        success = pdf_gen.generate_single_page_pdf(scanned_image, pdf_buffer)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate PDF"
            )
        
        # Reset buffer position
        pdf_buffer.seek(0)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scanned_document_{timestamp}.pdf"
        
        # Return PDF as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Scan-Status": status_message
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )


@app.post("/scan/preview")
async def scan_preview(
    file: UploadFile = File(..., description="Image file to scan"),
    enhance: Optional[bool] = Form(True, description="Apply image enhancement")
):
    """
    Scan a document and return the processed image (not PDF) for preview
    
    Args:
        file: Uploaded image file
        enhance: Whether to enhance the image (default: True)
        
    Returns:
        Processed image as JPEG
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        # Scan the document
        scanned_image, status_message = scanner.scan(image, enhance=enhance)
        
        if scanned_image is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to scan document: {status_message}"
            )
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', scanned_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Convert to bytes
        image_bytes = io.BytesIO(buffer.tobytes())
        
        # Return image as streaming response
        return StreamingResponse(
            image_bytes,
            media_type="image/jpeg",
            headers={"X-Scan-Status": status_message}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
