import argparse
import cv2
import numpy as np
from main import (
    detect_document_with_vision_api,
    detect_document_with_ai,
    detect_document_contour,
    order_points,
)

def main():
    parser = argparse.ArgumentParser(description="Debug script for document detection.")
    parser.add_argument("image_path", help="Path to the input image.")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image at {args.image_path}")
        return

    detection_methods = {
        "vision_api": detect_document_with_vision_api,
        "ai_model": detect_document_with_ai,
        "opencv": detect_document_contour,
    }

    for name, method in detection_methods.items():
        print(f"Running detection method: {name}")
        # Create a fresh copy of the image for each method
        debug_image = image.copy()

        try:
            # We need to pass the raw image bytes to some detectors
            _, img_encoded = cv2.imencode(".jpg", image)
            image_bytes = img_encoded.tobytes()

            if name == "vision_api":
                corners = method(image_bytes)
            else:
                corners = method(image)

            if corners is not None and len(corners) == 4:
                print(f"  - Detected corners: {corners.tolist()}")
                # Ensure corners are in a consistent order for drawing
                ordered_corners = order_points(corners)

                # Draw the detected polygon
                pts = np.array(ordered_corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=5)

                # Draw circles on the corners
                for x, y in ordered_corners:
                    cv2.circle(debug_image, (int(x), int(y)), 10, (0, 0, 255), -1)

                output_path = f"debug_output_{name}.jpg"
                cv2.imwrite(output_path, debug_image)
                print(f"  - Saved debug image to {output_path}")
            else:
                print("  - No document found.")

        except Exception as e:
            print(f"  - An error occurred: {e}")

if __name__ == "__main__":
    main()
