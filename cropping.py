import cv2
import os

def zoom_into_bbox(image_path, label_path, output_path):
    """
    Reads an image and a YOLO label file, extracts the bounding box, 
    zooms into it (crops), and saves the result.
    
    Parameters:
        image_path: Path to the input image.
        label_path: Path to the YOLO label file.
        output_path: Path to save the cropped output image.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image_height, image_width = img.shape[0], img.shape[1]

    # Read YOLO label file
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError("Label file must have: class x_center y_center width height")

    cls, x_center, y_center, w, h = parts
    x_center, y_center, w, h = map(float, [x_center, y_center, w, h])

    # Convert relative values to absolute pixel values
    x_c = x_center * image_width
    y_c = y_center * image_height
    w   = w * image_width
    h   = h * image_height

    # Top-left corner
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    # bottom-right corner
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)

    # Clip values to stay inside image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_width, x2), min(image_height, y2)

    # Crop image
    cropped = img[y1:y2, x1:x2]

    # Save with index if multiple boxes
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_name = f"{base_name}_cls{cls}_{i}_reingezoomt.jpg"
    save_path = os.path.join(output_path, save_name)

    # Save cropped image
    cv2.imwrite(save_path, cropped)
    print(f"Saved: {save_path}")
