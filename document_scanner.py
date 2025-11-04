"""
Document Scanner Module
High-precision document detection using:
- Unsharp masking + Local binarization (shadow-resistant)
- Otsu-based automatic Canny thresholds
- Multi-candidate rectangle approximation
- Comprehensive scoring (area, rectangularity, edge linearity, edge contrast)
- Clockwise 4-point ordering for perspective transform
- Hough line fallback for robust detection
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class DocumentScanner:
    """
    High-precision document scanner with comprehensive scoring and fallback mechanisms
    """
    
    def __init__(self):
        pass
        
    def unsharp_mask(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, 
                     amount: float = 1.0, threshold: int = 0) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges
        
        Args:
            image: Input image
            kernel_size: Gaussian kernel size
            sigma: Gaussian sigma
            amount: Enhancement amount
            threshold: Threshold for mask application
            
        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    
    def local_binarization(self, image: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
        """
        Apply local adaptive thresholding (shadow-resistant)
        
        Args:
            image: Grayscale image
            block_size: Block size for adaptive threshold
            C: Constant subtracted from mean
            
        Returns:
            Binary image
        """
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        return binary
    
    def auto_canny(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Automatic Canny edge detection using Otsu threshold
        
        Args:
            image: Grayscale image
            sigma: Multiplier for threshold calculation
            
        Returns:
            Edge image
        """
        # Compute Otsu threshold
        otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply Canny with automatic thresholds based on Otsu
        lower = int(max(0, (1.0 - sigma) * otsu_thresh))
        upper = int(min(255, (1.0 + sigma) * otsu_thresh))
        edges = cv2.Canny(image, lower, upper)
        
        return edges
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing: Unsharp mask + Local binarization
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (processed grayscale, edges)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply unsharp mask
        sharpened = self.unsharp_mask(gray, kernel_size=5, sigma=1.0, amount=1.5)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(sharpened, h=10)
        
        # Apply local binarization
        binary = self.local_binarization(denoised, block_size=51, C=10)
        
        # Auto Canny on denoised image
        edges = self.auto_canny(denoised, sigma=0.33)
        
        # Dilate edges to connect broken segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        return denoised, edges
    
    def find_contour_candidates(self, edges: np.ndarray, img_area: float) -> List[np.ndarray]:
        """
        Find multiple rectangle candidates from contours
        
        Args:
            edges: Edge image
            img_area: Total image area
            
        Returns:
            List of 4-point contour candidates
        """
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        candidates = []
        
        # Check top contours
        for contour in contours[:50]:
            area = cv2.contourArea(contour)
            area_ratio = area / img_area
            
            # Skip if too small or too large
            if area_ratio < 0.05 or area_ratio > 0.98:
                continue
            
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            
            # Try multiple epsilon values
            for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                if len(approx) == 4:
                    candidates.append(approx.reshape(4, 2))
                    break  # Found a 4-point approximation, move to next contour
        
        return candidates
    
    def compute_rectangularity(self, points: np.ndarray) -> float:
        """
        Compute how rectangular a 4-point polygon is
        
        Args:
            points: 4 corner points
            
        Returns:
            Rectangularity score (0-1, higher is better)
        """
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Compute areas
        poly_area = cv2.contourArea(points)
        rect_area = cv2.contourArea(box)
        
        if rect_area == 0:
            return 0.0
        
        # Rectangularity = polygon area / rectangle area
        return poly_area / rect_area
    
    def compute_edge_linearity(self, points: np.ndarray) -> float:
        """
        Compute how straight the edges are
        
        Args:
            points: 4 corner points (ordered)
            
        Returns:
            Linearity score (0-1, higher is better)
        """
        # For perfect rectangle, opposite sides should be parallel and equal length
        # Compute side lengths
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            sides.append(length)
        
        # Opposite sides should be similar
        ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2]) if max(sides[0], sides[2]) > 0 else 0
        ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3]) if max(sides[1], sides[3]) > 0 else 0
        
        return (ratio1 + ratio2) / 2.0
    
    def compute_edge_contrast(self, image: np.ndarray, points: np.ndarray) -> float:
        """
        Compute average edge contrast along the document boundary
        
        Args:
            image: Grayscale image
            points: 4 corner points (ordered)
            
        Returns:
            Contrast score (0-255)
        """
        contrasts = []
        
        for i in range(4):
            p1 = points[i].astype(int)
            p2 = points[(i + 1) % 4].astype(int)
            
            # Sample points along the edge
            num_samples = 20
            for t in np.linspace(0, 1, num_samples):
                px = int(p1[0] * (1 - t) + p2[0] * t)
                py = int(p1[1] * (1 - t) + p2[1] * t)
                
                # Check bounds
                if py < 1 or py >= image.shape[0] - 1 or px < 1 or px >= image.shape[1] - 1:
                    continue
                
                # Compute gradient magnitude at this point
                gx = int(image[py, px + 1]) - int(image[py, px - 1])
                gy = int(image[py + 1, px]) - int(image[py - 1, px])
                gradient = np.sqrt(gx ** 2 + gy ** 2)
                contrasts.append(gradient)
        
        return np.mean(contrasts) if contrasts else 0.0
    
    def score_candidate(self, points: np.ndarray, image: np.ndarray, img_area: float) -> float:
        """
        Comprehensive scoring of a candidate rectangle
        
        Args:
            points: 4 corner points
            image: Grayscale image
            img_area: Total image area
            
        Returns:
            Combined score (higher is better)
        """
        # Area score (prefer larger documents)
        area = cv2.contourArea(points)
        area_score = area / img_area
        
        # Rectangularity score
        rect_score = self.compute_rectangularity(points)
        
        # Order points for linearity check
        ordered_points = self.order_points_clockwise(points)
        
        # Edge linearity score
        linearity_score = self.compute_edge_linearity(ordered_points)
        
        # Edge contrast score (normalized)
        contrast_score = self.compute_edge_contrast(image, ordered_points) / 50.0
        contrast_score = min(contrast_score, 1.0)
        
        # Combined score (weighted)
        score = (
            area_score * 0.3 +
            rect_score * 0.3 +
            linearity_score * 0.2 +
            contrast_score * 0.2
        )
        
        return score
    
    def order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: 4 points
            
        Returns:
            Ordered points
        """
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
        tl, tr = top_pts[0], top_pts[1]
        
        # Bottom two points
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
        bl, br = bottom_pts[0], bottom_pts[1]
        
        return np.array([tl, tr, br, bl], dtype="float32")
    
    def hough_line_fallback(self, edges: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Fallback method: Use Hough lines to reconstruct rectangle
        
        Args:
            edges: Edge image
            image_shape: (height, width)
            
        Returns:
            4 corner points or None
        """
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return None
        
        # Separate into horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 20 or angle > 160:  # Horizontal
                horizontal_lines.append(line[0])
            elif 70 < angle < 110:  # Vertical
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Find top, bottom, left, right lines
        horizontal_lines = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
        vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
        
        # Compute intersections
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
            return (px, py)
        
        tl = line_intersection(top_line, left_line)
        tr = line_intersection(top_line, right_line)
        br = line_intersection(bottom_line, right_line)
        bl = line_intersection(bottom_line, left_line)
        
        if None in [tl, tr, br, bl]:
            return None
        
        points = np.array([tl, tr, br, bl], dtype="float32")
        
        # Validate points are within image
        h, w = image_shape
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            return None
        
        return points
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find document contour with comprehensive scoring
        
        Args:
            image: Input BGR image
            
        Returns:
            Best 4-point contour or None
        """
        # Preprocess
        gray, edges = self.preprocess_image(image)
        
        # Get image area
        img_area = image.shape[0] * image.shape[1]
        
        # Find candidates
        candidates = self.find_contour_candidates(edges, img_area)
        
        if not candidates:
            # Try Hough line fallback
            fallback_points = self.hough_line_fallback(edges, (image.shape[0], image.shape[1]))
            if fallback_points is not None:
                return fallback_points
            return None
        
        # Score all candidates
        scored_candidates = []
        for candidate in candidates:
            score = self.score_candidate(candidate, gray, img_area)
            scored_candidates.append((score, candidate))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return best candidate
        best_score, best_candidate = scored_candidates[0]
        
        # Require minimum score
        if best_score < 0.3:
            # Try Hough line fallback
            fallback_points = self.hough_line_fallback(edges, (image.shape[0], image.shape[1]))
            if fallback_points is not None:
                return fallback_points
            return None
        
        return best_candidate
    
    def apply_perspective_transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Apply perspective transform with clockwise ordered points
        
        Args:
            image: Input image
            points: 4 corner points
            
        Returns:
            Warped image
        """
        # Order points clockwise
        ordered = self.order_points_clockwise(points)
        (tl, tr, br, bl) = ordered
        
        # Compute output dimensions
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
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
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered, dst)
        
        # Apply transform
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        return warped
    
    def enhance_document(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance scanned document
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Gamma correction
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # Slight sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def scan(self, image: np.ndarray, enhance: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Scan document with high-precision detection
        
        Args:
            image: Input BGR image
            enhance: Whether to enhance
            
        Returns:
            Tuple of (processed image, status message)
        """
        if image is None or image.size == 0:
            return None, "Invalid image"
        
        # Store original
        original = image.copy()
        
        # Resize if too large
        max_dim = 1500
        height, width = image.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            work_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work_image = image.copy()
            scale = 1.0
        
        # Find document
        points = self.find_document_contour(work_image)
        
        if points is None:
            if enhance:
                return self.enhance_document(original), "No document detected, returning enhanced original"
            return original, "No document detected, returning original"
        
        # Scale points back to original size
        if scale != 1.0:
            points = points / scale
        
        # Apply perspective transform
        warped = self.apply_perspective_transform(original, points)
        
        # Enhance
        if enhance:
            warped = self.enhance_document(warped)
        
        return warped, "Document successfully scanned"
