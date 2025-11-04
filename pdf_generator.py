"""
PDF Generator Module
Converts scanned images to PDF format
"""

import io
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import numpy as np
import cv2


class PDFGenerator:
    """
    A class to generate PDF files from scanned document images
    """
    
    def __init__(self, page_size: str = "A4"):
        """
        Initialize PDF generator
        
        Args:
            page_size: Page size for PDF ("A4" or "letter")
        """
        self.page_size = A4 if page_size.upper() == "A4" else letter
        
    def opencv_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image (BGR) to PIL Image (RGB)
        
        Args:
            cv_image: OpenCV image in BGR format
            
        Returns:
            PIL Image in RGB format
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def generate_pdf(self, images: list, output_buffer: io.BytesIO, 
                    fit_to_page: bool = True) -> bool:
        """
        Generate a PDF file from a list of images
        
        Args:
            images: List of images (OpenCV format or PIL Image)
            output_buffer: BytesIO buffer to write PDF to
            fit_to_page: Whether to fit images to page size
            
        Returns:
            True if successful, False otherwise
        """
        if not images:
            return False
        
        try:
            # Create PDF canvas
            pdf_canvas = canvas.Canvas(output_buffer, pagesize=self.page_size)
            page_width, page_height = self.page_size
            
            for img in images:
                # Convert to PIL if it's an OpenCV image
                if isinstance(img, np.ndarray):
                    pil_img = self.opencv_to_pil(img)
                else:
                    pil_img = img
                
                # Get image dimensions
                img_width, img_height = pil_img.size
                
                if fit_to_page:
                    # Calculate scaling to fit page while maintaining aspect ratio
                    width_ratio = page_width / img_width
                    height_ratio = page_height / img_height
                    scale = min(width_ratio, height_ratio)
                    
                    # Calculate new dimensions
                    new_width = img_width * scale
                    new_height = img_height * scale
                    
                    # Center the image on the page
                    x_offset = (page_width - new_width) / 2
                    y_offset = (page_height - new_height) / 2
                    
                    # Draw image on canvas
                    pdf_canvas.drawImage(
                        ImageReader(pil_img),
                        x_offset,
                        y_offset,
                        width=new_width,
                        height=new_height,
                        preserveAspectRatio=True
                    )
                else:
                    # Draw image at original size
                    pdf_canvas.drawImage(
                        ImageReader(pil_img),
                        0,
                        page_height - img_height,
                        width=img_width,
                        height=img_height
                    )
                
                # Create new page for next image
                pdf_canvas.showPage()
            
            # Save the PDF
            pdf_canvas.save()
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False
    
    def generate_single_page_pdf(self, image: np.ndarray, 
                                 output_buffer: io.BytesIO) -> bool:
        """
        Generate a single-page PDF from an image
        
        Args:
            image: OpenCV image or PIL Image
            output_buffer: BytesIO buffer to write PDF to
            
        Returns:
            True if successful, False otherwise
        """
        return self.generate_pdf([image], output_buffer, fit_to_page=True)
