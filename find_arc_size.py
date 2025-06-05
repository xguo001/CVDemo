import cv2
import numpy as np
import math
import os

def is_rough_arc(contour, min_angle=20, max_angle=220):
    if len(contour) < 5:
        return False

    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return False

    axes = ellipse[1]
    a = axes[0] / 2
    b = axes[1] / 2

    arc_length = cv2.arcLength(contour, closed=False)
    ellipse_circumference = math.pi * (3*(a+b) - math.sqrt((3*a + b)*(a + 3*b)))

    if ellipse_circumference == 0:
        return False

    coverage_deg = (arc_length / ellipse_circumference) * 360

    # Debug print for contours that are somewhat arc-like
    if min_angle <= coverage_deg <= max_angle:
        print(f"Contour with {len(contour)} pts, coverage angle: {coverage_deg:.1f} degrees")

    return min_angle <= coverage_deg <= max_angle

def main():
    input_path = "output_images/page_0.png"
    output_path = "output_images/contour_detection_loose.png"

    os.makedirs("output_images", exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error loading image at {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")

    detected_arcs = []
    for i, contour in enumerate(contours):
        if len(contour) < 30:
            continue  # skip very small contours (noise)

        if is_rough_arc(contour):
            detected_arcs.append(contour)

    print(f"Detected rough arcs (loose filter): {len(detected_arcs)}")

    contour_img = img.copy()
    cv2.drawContours(contour_img, detected_arcs, -1, (0, 0, 255), 2)

    cv2.imwrite(output_path, contour_img)
    print(f"Saved contour image: {output_path}")

if __name__ == "__main__":
    main()
