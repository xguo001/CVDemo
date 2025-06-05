import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from PIL import Image

# === Settings ===
Image.MAX_IMAGE_PIXELS = None

pdf_path = "floorplan.pdf"             # PDF containing floorplan
template_path = "template.png"         # Your screenshot/cropped door image
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Convert PDF to images ===
images = convert_from_path(pdf_path, dpi=200)
print(f"Converted {len(images)} pages.")

# === Step 2: Load and process door template ===
template_raw = cv2.imread(template_path)
template_gray = cv2.cvtColor(template_raw, cv2.COLOR_BGR2GRAY)
template_edges = cv2.Canny(template_gray, 50, 150)

# === Step 3: Try different scales of the template ===
scales = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# === Step 4: Process each PDF page ===
for i, image in enumerate(images):
    page_path = os.path.join(output_dir, f"page_{i+1}_raw.png")
    image.save(page_path, "PNG")

    # Resize the image to 50% (to match
