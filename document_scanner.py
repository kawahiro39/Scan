"""
Document Scanner Module
Automatically detects and crops document boundaries from images using OpenCV
Enhanced with techniques from:
- DocShadow-SD7K (ICCV 2023): Frequency-aware shadow removal
- DocScanner (IJCV 2025): Robust document localization
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
    
    def frequency_split(self, image: np.ndarray, sigma: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split image into low and high frequency components (DocShadow-SD7K technique)
        
        Args:
            image: Input image
            sigma: Gaussian blur sigma for low-pass filter
            
        Returns:
            Tuple of (low_freq, high_freq) images
        """
        # Low frequency: Gaussian blur
        low_freq = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        
        # High frequency: original - low frequency
        high_freq = cv2.subtract(image, low_freq)
        
        return low_freq, high_freq
    
    def estimate_illumination_map(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate illumination map using heavy Gaussian blur (DocShadow-SD7K technique)
        
        Args:
            image: Grayscale image
            
        Returns:
            Illumination map
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply heavy Gaussian blur to estimate background illumination
        illumination = cv2.GaussianBlur(img_float, (0, 0), sigmaX=50, sigmaY=50)
        
        return illumination
    
    def detect_shadow_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect shadow regions using illumination ratio (DocShadow-SD7K technique)
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary shadow mask
        """
        # Convert to LAB and extract L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Estimate background illumination
        illumination = self.estimate_illumination_map(l_channel)
        
        # Compute illumination ratio (avoid division by zero)
        ratio = (l_channel + 1.0) / (illumination + 1.0)
        
        # Shadows have lower ratio (< 0.92 threshold)
        shadow_mask = (ratio < 0.92).astype(np.uint8) * 255
        
        # Refine mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for soft edges
        shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
        
        return shadow_mask
    
    def remove_shadow_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced shadow removal using frequency-aware processing (DocShadow-SD7K technique)
        
        Args:
            image: Input BGR image
            
        Returns:
            Shadow-removed image
        """
        # Split into low and high frequency
        low_freq, high_freq = self.frequency_split(image, sigma=20.0)
        
        # Detect shadow mask
        shadow_mask = self.detect_shadow_mask(image)
        shadow_mask_norm = shadow_mask.astype(np.float32) / 255.0
        
        # Convert to LAB color space for better shadow correction
        lab = cv2.cvtColor(low_freq, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Estimate illumination on L channel
        l_float = l.astype(np.float32)
        illumination = self.estimate_illumination_map(l_float)
        
        # Compute correction factor for shadowed regions
        # Brighten dark regions more
        correction_factor = np.ones_like(l_float)
        mask_bool = shadow_mask > 128
        if np.any(mask_bool):
            # Compute median of non-shadow and shadow regions
            non_shadow_median = np.median(l_float[~mask_bool]) if np.any(~mask_bool) else 128
            shadow_median = np.median(l_float[mask_bool])
            
            # Calculate brightening factor
            if shadow_median > 0:
                factor = non_shadow_median / shadow_median
                factor = np.clip(factor, 1.0, 2.0)  # Limit factor
                correction_factor[mask_bool] = factor
        
        # Apply correction with smooth transition
        l_corrected = l_float * (1.0 + (correction_factor - 1.0) * shadow_mask_norm)
        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
        
        # Merge back to LAB
        lab_corrected = cv2.merge([l_corrected, a, b])
        low_freq_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # Combine corrected low frequency with original high frequency
        result = cv2.add(low_freq_corrected, high_freq)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def preprocess_for_contour_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for robust document localization (DocScanner technique)
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed edge image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply Canny edge detection on multiple scales
        edges1 = cv2.Canny(enhanced, 30, 100)
        edges2 = cv2.Canny(enhanced, 50, 150)
        edges3 = cv2.Canny(enhanced, 75, 200)
        
        # Combine edges from different scales
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        
        # Dilate edges to connect broken segments
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, dilate_kernel, iterations=2)
        
        # Close small gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        
        return edges
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the document contour with robust localization (DocScanner technique)
        
        Args:
            image: Input BGR image
            
        Returns:
            Document contour as numpy array or None if not found
        """
        # Get preprocessed edges
        edges = self.preprocess_for_contour_detection(image)
        
        # Find contours (external only to avoid nested contours)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Look for the first contour that can be approximated to 4 points
        for contour in contours[:25]:  # Check top 25 largest contours
            # Calculate area ratio to filter out too small contours
            area = cv2.contourArea(contour)
            area_ratio = area / img_area
            
            # Skip if contour is too small (less than 3% of image area)
            if area_ratio < 0.03:
                continue
            
            # Skip if contour is almost the entire image (likely image border)
            if area_ratio > 0.98:
                continue
            
            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            
            # Try multiple epsilon values for better approximation
            for epsilon_factor in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
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
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            return False
        
        # Check if the contour covers a reasonable portion of the image
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        if bbox_area > 0:
            solidity = float(contour_area) / bbox_area
            # Document should have reasonable solidity
            if solidity < 0.60:
                return False
        
        # Check if contour touches image borders (may indicate incomplete detection)
        margin = 10
        touches_border = (x < margin or y < margin or 
                         (x + w) > (img_width - margin) or 
                         (y + h) > (img_height - margin))
        
        # If touches border and is very large, it's likely the image border itself
        if touches_border and contour_area / (img_width * img_height) > 0.95:
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
        
        # Apply the perspective transformation with high-quality interpolation
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        
        return warped
    
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
        if len(coords) > 100:  # Need sufficient points
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only rotate if angle is significant (more than 0.3 degrees)
            if abs(angle) > 0.3:
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
    
    def enhance_image(self, image: np.ndarray, remove_shadows: bool = True) -> np.ndarray:
        """
        Enhance the image quality using frequency-aware processing
        
        Args:
            image: Input image
            remove_shadows: Whether to apply advanced shadow removal
            
        Returns:
            Enhanced image
        """
        result = image.copy()
        
        # Remove shadows with frequency-aware method
        if remove_shadows:
            result = self.remove_shadow_advanced(result)
        
        # Apply gentle gamma correction
        gamma = 1.15
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)
        
        # Apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gentle contrast enhancement
        result = cv2.convertScaleAbs(result, alpha=1.05, beta=2)
        
        # Apply gentle sharpening
        kernel = np.array([[0, -1, 0],
                          [-1, 5.5, -1],
                          [0, -1, 0]]) / 1.5
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
        max_dimension = 2048  # Increased for better detection
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
                enhanced = self.enhance_image(rotated, remove_shadows=True)
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
            warped = self.enhance_image(warped, remove_shadows=True)
        
        return warped, "Document successfully scanned and processed"
