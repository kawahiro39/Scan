"""
Document Scanner Module
Automatically detects and crops document boundaries from images using OpenCV
Enhanced version with improved edge detection, rotation correction, and shadow removal
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
        Enhanced preprocessing for better contour detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Apply adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilate to connect nearby edges
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, dilate_kernel, iterations=1)
        
        # Apply Canny edge detection on multiple scales
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges3 = cv2.Canny(blurred, 70, 200)
        
        # Combine edges from different scales
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        # Dilate edges to connect broken segments
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        
        return edges
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the document contour in the image with improved detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Document contour as numpy array or None if not found
        """
        # Get preprocessed edges
        edges = self.preprocess_for_contour_detection(image)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Look for the first contour that can be approximated to 4 points
        for contour in contours[:15]:  # Check top 15 largest contours
            # Calculate area ratio to filter out too small contours
            area = cv2.contourArea(contour)
            area_ratio = area / img_area
            
            # Skip if contour is too small (less than 10% of image area)
            if area_ratio < 0.1:
                continue
            
            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            
            # Try multiple epsilon values for better approximation
            for epsilon_factor in [0.01, 0.015, 0.02, 0.025, 0.03]:
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                # If our approximated contour has 4 points, we found the document
                if len(approx) == 4:
                    # Verify that the contour is roughly rectangular
                    if self._is_valid_document_contour(approx, img_width, img_height):
                        return approx
        
        return None
    
    def _is_valid_document_contour(self, contour: np.ndarray, img_width: int, img_height: int) -> bool:
        """
        Validate if the contour is a valid document boundary
        
        Args:
            contour: 4-point contour
            img_width: Image width
            img_height: Image height
            
        Returns:
            True if valid, False otherwise
        """
        # Check if contour has 4 points
        if len(contour) != 4:
            return False
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio (should be reasonable for documents)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            return False
        
        # Check if the contour covers a reasonable portion of the image
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        if bbox_area > 0:
            solidity = float(contour_area) / bbox_area
            # Document should have high solidity (close to rectangle)
            if solidity < 0.7:
                return False
        
        return True
    
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
        
        # Ensure minimum dimensions
        if maxWidth < 100 or maxHeight < 100:
            maxWidth = max(maxWidth, 100)
            maxHeight = max(maxHeight, 100)
        
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
    
    def remove_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from scanned document
        
        Args:
            image: Input image
            
        Returns:
            Image with shadows removed
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply morphological operations to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # Calculate the difference to remove shadows
        diff = cv2.subtract(bg, l)
        
        # Normalize
        norm = cv2.normalize(diff, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Merge back
        result = cv2.merge([norm, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return result
    
    def remove_bleed_through(self, image: np.ndarray) -> np.ndarray:
        """
        Remove bleed-through (show-through) from scanned documents
        
        Args:
            image: Input image
            
        Returns:
            Image with reduced bleed-through
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to smooth while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding to separate text from background
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        
        # Convert back to BGR
        result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Blend with original for more natural look
        alpha = 0.7
        blended = cv2.addWeighted(image, alpha, result, 1 - alpha, 0)
        
        return blended
    
    def auto_rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-rotate image to correct orientation based on text lines
        
        Args:
            image: Input image
            
        Returns:
            Rotated image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        # Calculate the rotation angle
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only rotate if angle is significant (more than 0.5 degrees)
            if abs(angle) > 0.5:
                # Get image dimensions
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                
                # Perform rotation
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        
        return image
    
    def enhance_image(self, image: np.ndarray, remove_shadows: bool = True, 
                     remove_bleed: bool = True) -> np.ndarray:
        """
        Enhance the image quality using various techniques
        
        Args:
            image: Input image
            remove_shadows: Whether to apply shadow removal
            remove_bleed: Whether to apply bleed-through removal
            
        Returns:
            Enhanced image
        """
        result = image.copy()
        
        # Remove shadows first
        if remove_shadows:
            result = self.remove_shadow(result)
        
        # Apply gamma correction
        gamma = 1.3
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)
        
        # Apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Remove bleed-through
        if remove_bleed:
            result = self.remove_bleed_through(result)
        
        # Increase contrast slightly
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.5
        result = cv2.filter2D(result, -1, kernel)
        
        return result
    
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
        max_dimension = 2000  # Increased for better detection
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
            # If no document found, try to auto-rotate and enhance
            if enhance:
                rotated = self.auto_rotate(image)
                enhanced = self.enhance_image(rotated)
                return enhanced, "No document boundary detected, returning enhanced and rotated original"
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
        
        # Auto-rotate if needed
        warped = self.auto_rotate(warped)
        
        # Enhance if requested
        if enhance:
            warped = self.enhance_image(warped, remove_shadows=True, remove_bleed=True)
        
        return warped, "Document successfully scanned and processed"
