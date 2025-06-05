import cv2
import numpy as np
import math
import os

def angle_between(p1, p2, center):
    a = np.array(p1) - np.array(center)
    b = np.array(p2) - np.array(center)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def fit_arc(contour, min_radius=10, max_radius=100, min_angle=60, max_angle=120):
    if len(contour) < 5:
        return None

    (x, y), radius = cv2.minEnclosingCircle(contour)
    radius = int(radius)
    if radius < min_radius or radius > max_radius:
        return None

    center = (int(x), int(y))
    angles = []
    for i in range(0, len(contour), max(1, len(contour)//10)):
        pt = tuple(contour[i][0])
        angles.append(math.degrees(math.atan2(pt[1]-center[1], pt[0]-center[0])))

    if len(angles) < 2:
        return None

    angle_span = max(angles) - min(angles)
    if angle_span < 0:
        angle_span += 360

    if min_angle <= angle_span <= max_angle:
        return center, radius, angle_span
    return None

def detect_arcs(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: image not found")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arc_count = 0
    for i, contour in enumerate(contours):
        result = fit_arc(contour)
        if result:
            center, radius, angle_span = result
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            arc_count += 1

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "detected_arcs_debug.png")
    cv2.imwrite(output_file, img)
    print(f"Detected arcs: {arc_count}")
    print(f"Saved result to: {output_file}")

# Example usage
detect_arcs("output_images/page_0.png", "output_images")
