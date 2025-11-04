"""
Document Scanner Module
Automatically detects and crops document boundaries from images using OpenCV
Simplified and optimized for color preservation and accurate detection
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
    
    def preprocess_for_contour_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing for robust document boundary detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 11, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
        
        # Apply morphological gradient to enhance edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply Canny on the original blurred image
        canny = cv2.Canny(blurred, 40, 120)
        
        # Combine threshold and canny
        edges = cv2.bitwise_or(thresh, canny)
        
        # Dilate to connect edges
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, dilate_kernel, iterations=3)
        
        # Close gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        
        return edges
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the document contour with improved detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Document contour as numpy array or None if not found
        """
        # Get edges
        edges = self.preprocess_for_contour_detection(image)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Image dimensions
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Try to find document contour
        for contour in contours[:30]:
            area = cv2.contourArea(contour)
            area_ratio = area / img_area
            
            # Skip too small or too large contours
            if area_ratio < 0.1 or area_ratio > 0.99:
                continue
            
            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            
            # Try different epsilon values
            for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                
                if len(approx) == 4:
                    if self._is_valid_document_contour(approx, img_width, img_height):
                        return approx
        
        return None
    
    def _is_valid_document_contour(self, contour: np.ndarray, img_width: int, img_height: int) -> bool:
        """
        Validate document contour
        
        Args:
            contour: 4-point contour
            img_width: Image width
            img_height: Image height
            
        Returns:
            True if valid
        """
        if len(contour) != 4:
            return False
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False
        
        # Check solidity
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        if bbox_area > 0:
            solidity = float(contour_area) / bbox_area
            if solidity < 0.7:
                return False
        
        return True
    
    def apply_perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform
        
        Args:
            image: Input image
            pts: Corner points
            
        Returns:
            Warped image
        """
        # Order points
        rect = self.order_points(pts)
        
        # Compute dimensions
        maxWidth, maxHeight = self.compute_output_dimensions(rect)
        
        # Ensure minimum dimensions
        maxWidth = max(maxWidth, 100)
        maxHeight = max(maxHeight, 100)
        
        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Get transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply transform
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return warped
    
    def remove_shadow_simple(self, image: np.ndarray) -> np.ndarray:
        """
        Simple shadow removal that preserves color
        
        Args:
            image: Input BGR image
            
        Returns:
            Shadow-removed image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality while preserving color
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        result = image.copy()
        
        # Simple shadow removal
        result = self.remove_shadow_simple(result)
        
        # Gamma correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        result = cv2.LUT(result, table)
        
        # Contrast adjustment
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def scan(self, image: np.ndarray, enhance: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Scan a document image
        
        Args:
            image: Input BGR image
            enhance: Whether to enhance
            
        Returns:
            Tuple of (processed image, status message)
        """
        if image is None or image.size == 0:
            return None, "Invalid image"
        
        # Original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize if too large
        max_dimension = 1500
        if max(orig_height, orig_width) > max_dimension:
            scale = max_dimension / max(orig_height, orig_width)
            work_image = cv2.resize(image, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
        else:
            work_image = image.copy()
            scale = 1.0
        
        # Find contour
        document_contour = self.find_document_contour(work_image)
        
        if document_contour is None:
            # Return enhanced original if no contour found
            if enhance:
                return self.enhance_image(image), "No document boundary detected, returning enhanced original"
            return image, "No document boundary detected, returning original"
        
        # Scale contour back
        if scale != 1.0:
            document_contour = document_contour.astype("float32")
            document_contour /= scale
            document_contour = document_contour.astype("int32")
        
        # Reshape
        pts = document_contour.reshape(4, 2)
        
        # Transform
        warped = self.apply_perspective_transform(image, pts)
        
        # Enhance
        if enhance:
            warped = self.enhance_image(warped)
        
        return warped, "Document successfully scanned and processed"
