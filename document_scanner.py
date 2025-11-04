"""
Document Scanner Module
Automatically detects and crops document boundaries from images using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class DocumentScanner:
    """
    A class to scan and process document images with automatic boundary detection
    """
    
    def __init__(self):
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.approx_epsilon = 0.02
        self.gaussian_kernel = (5, 5)
        
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: numpy array of 4 points with shape (4, 2)
            
        Returns:
            Ordered numpy array of points
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left point has the smallest sum
        # Bottom-right point has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has the smallest difference
        # Bottom-left point has the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def compute_output_dimensions(self, pts: np.ndarray) -> Tuple[int, int]:
        """
        Compute the width and height for the output warped image
        
        Args:
            pts: Ordered corner points
            
        Returns:
            Tuple of (width, height)
        """
        (tl, tr, br, bl) = pts
        
        # Compute width
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Compute height
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        return maxWidth, maxHeight
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the document contour in the image
        
        Args:
            image: Input BGR image
            
        Returns:
            Document contour as numpy array or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for the first contour that can be approximated to 4 points
        for contour in contours[:10]:  # Check top 10 largest contours
            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            
            # Approximate the contour
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
            
            # If our approximated contour has 4 points, we found the document
            if len(approx) == 4:
                return approx
        
        return None
    
    def apply_perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform to get a top-down view of the document
        
        Args:
            image: Input image
            pts: Corner points of the document
            
        Returns:
            Warped image
        """
        # Order the points
        rect = self.order_points(pts)
        
        # Compute output dimensions
        maxWidth, maxHeight = self.compute_output_dimensions(rect)
        
        # Construct the destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply the perspective transformation
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the image quality using various techniques
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Apply gamma correction
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(image, table)
        
        # Apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def scan(self, image: np.ndarray, enhance: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Scan a document image and return the cropped and corrected version
        
        Args:
            image: Input BGR image
            enhance: Whether to apply image enhancement
            
        Returns:
            Tuple of (processed image or None, status message)
        """
        if image is None or image.size == 0:
            return None, "Invalid image"
        
        # Store original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize for processing if image is too large
        max_dimension = 1500
        if max(orig_height, orig_width) > max_dimension:
            scale = max_dimension / max(orig_height, orig_width)
            work_image = cv2.resize(image, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
        else:
            work_image = image.copy()
            scale = 1.0
        
        # Find document contour
        document_contour = self.find_document_contour(work_image)
        
        if document_contour is None:
            # If no document found, return original image
            if enhance:
                return self.enhance_image(image), "No document boundary detected, returning enhanced original"
            return image, "No document boundary detected, returning original"
        
        # Scale contour back to original image size if we resized
        if scale != 1.0:
            document_contour = document_contour.astype("float32")
            document_contour /= scale
            document_contour = document_contour.astype("int32")
        
        # Reshape contour to (4, 2)
        pts = document_contour.reshape(4, 2)
        
        # Apply perspective transform on original image
        warped = self.apply_perspective_transform(image, pts)
        
        # Enhance if requested
        if enhance:
            warped = self.enhance_image(warped)
        
        return warped, "Document successfully scanned and processed"
