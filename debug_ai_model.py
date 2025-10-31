import cv2
import numpy as np
import requests
from main import detect_document_with_vision_api
import os

# Create a directory for debug outputs if it doesn't exist to avoid errors
os.makedirs("debug_output", exist_ok=True)

def main():
    """
    Main function to run the debugging process for the Vision API.
    It downloads an image, runs the detection, and prints the results.
    """
    # This is the low-contrast image that was previously failing.
    image_url = "https://i.gyazo.com/fda9505817e79b926f1b142f18d7a162.png"
    print(f"--- Testing Low-Contrast Image ---")
    print(f"Downloading image from: {image_url}")

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        # Read the image content into bytes
        image_bytes = response.content

        # Also decode for getting dimensions
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not decode image.")
            return

        original_height, original_width = img.shape[:2]
        print(f"Image dimensions: {original_width}x{original_height}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return

    print("Running document detection with Google Cloud Vision API...")
    document_points = detect_document_with_vision_api(image_bytes)

    if document_points is not None:
        print("\n--- Document Detection Successful ---")
        print("Detected corner points (tl, tr, br, bl):")
        print(document_points)

        # Basic sanity checks on the coordinates
        valid = True
        for i, name in enumerate(["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]):
            px, py = document_points[i]
            if not (0 <= px <= original_width and 0 <= py <= original_height):
                print(f"  - WARNING: {name} point ({px}, {py}) is outside image bounds.")
                valid = False

        if valid:
            print("\nVerification: All points are within the image boundaries.")
        else:
            print("\nVerification: Some points are out of bounds, indicating a potential detection issue.")

    else:
        print("\n--- Document Detection Failed ---")
        print("The Vision API did not return any document points.")

    # --- Test the second, more complex image ---
    print("\n\n--- Testing High-Clutter Image ---")
    image_url_2 = "https://i.gyazo.com/1511f9324157509f75f3a7c442431f47.png"
    print(f"Downloading image from: {image_url_2}")

    try:
        response = requests.get(image_url_2, stream=True)
        response.raise_for_status()
        image_bytes_2 = response.content
        image_array_2 = np.asarray(bytearray(image_bytes_2), dtype=np.uint8)
        img_2 = cv2.imdecode(image_array_2, cv2.IMREAD_COLOR)
        if img_2 is None:
            print("Error: Could not decode image 2.")
            return
        h2, w2 = img_2.shape[:2]
        print(f"Image 2 dimensions: {w2}x{h2}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image 2: {e}")
        return

    print("Running detection on high-clutter image...")
    points_2 = detect_document_with_vision_api(image_bytes_2)

    if points_2 is not None:
        print("\n--- High-Clutter Detection Successful ---")
        print("Detected corner points (tl, tr, br, bl):")
        print(points_2)
    else:
        print("\n--- High-Clutter Detection Failed ---")


if __name__ == "__main__":
    main()
