from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import base64
import io
import math
# from google.cloud import vision
# from PIL import Image
# import cv2
# import numpy as np

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI(
    title="Document Scan API (Diagnostic Mode)",
    description="An API to inspect incoming requests from Bubble.",
    version="0.1.0-diag"
)

@app.post("/scan")
async def scan_document(request: Request):
    """
    (Diagnostic Endpoint)
    Receives a request and returns its headers and body to diagnose the
    structure of the data sent by the client (Bubble).
    """
    try:
        # Get headers
        headers = dict(request.headers)

        # Get raw body
        body_bytes = await request.body()
        body_str = ""
        try:
            # Try to decode as utf-8 to see if it's readable text/form-data
            body_str = body_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If it's not utf-8, it's likely binary data, so represent it as base64
            body_str = base64.b64encode(body_bytes).decode('utf-8')

        return JSONResponse(content={
            "message": "Diagnostic data received. Please send this JSON back to the support agent.",
            "headers": headers,
            "body": body_str
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during diagnosis: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Scan API (Diagnostic Mode)"}

# --- ORIGINAL CODE (COMMENTED OUT) ---
# def process_document(image_bytes: bytes) -> tuple[bytes, str]:
#     ...
# def generate_pdf_response(image_bytes: bytes) -> str:
#     ...
# @app.post("/scan")
# async def scan_document_original(file: UploadFile = File(...)):
#     ...
