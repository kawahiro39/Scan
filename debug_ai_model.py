import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import segmentation_models_pytorch as smp
import os

# Create a directory for debug outputs if it doesn't exist
os.makedirs("debug_output", exist_ok=True)

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders four points in the order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies a perspective transform using four source points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom), 1)
    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = max(int(height_right), int(height_left), 1)
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped

def main():
    """Main function to run the debugging process."""
    print("Loading AI model...")
    model_name = "resnet34"
    ai_model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    ai_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai_model.to(device)
    print("AI model loaded.")

    image_path = "sample_images/problematic_image_2.png"
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    print("Preprocessing image...")
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    print("Running model inference...")
    with torch.no_grad():
        output = ai_model(input_tensor)

    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    original_height, original_width = img.shape[:2]
    mask_resized = cv2.resize(mask, (original_width, original_height))

    # Save the raw mask for analysis
    raw_mask_visual = (mask_resized * 255).astype(np.uint8)
    cv2.imwrite("debug_output/raw_mask.png", raw_mask_visual)
    print("Saved raw model mask to debug_output/raw_mask.png")

    # --- Existing Thresholding (for comparison) ---
    _, binary_mask = cv2.threshold(
        raw_mask_visual, 127, 255, cv2.THRESH_BINARY
    )
    cv2.imwrite("debug_output/binary_mask_hardcoded.png", binary_mask)
    print("Saved binary mask (hardcoded threshold) to debug_output/binary_mask_hardcoded.png")

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contours found with hardcoded threshold.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    img_with_contour = img.copy()
    cv2.drawContours(img_with_contour, [largest_contour], -1, (0, 255, 0), 3)
    cv2.imwrite("debug_output/contour.png", img_with_contour)
    print("Saved contour image to debug_output/contour.png")

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    document_points = order_points(box.astype(np.float32))

    warped = four_point_transform(img, document_points)
    cv2.imwrite("debug_output/cropped.png", warped)
    print("Saved cropped image to debug_output/cropped.png")

if __name__ == "__main__":
    main()
