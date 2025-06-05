import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import math

pdf_path = "floorplan.pdf"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def resize_image(img, scale_percent=40):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def point_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle_between_vectors(v1, v2):
    # Returns angle in degrees between two vectors
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def is_quarter_arc(gray_img, center, radius):
    # Check if the arc is roughly a quarter circle by examining edge points around the circle
    # Sample points along the circle's perimeter and count how many edges exist
    edge_points = 0
    total_points = 0
    for angle_deg in range(0, 91, 5):  # 0 to 90 degrees in steps
        angle_rad = np.deg2rad(angle_deg)
        x = int(center[0] + radius * np.cos(angle_rad))
        y = int(center[1] + radius * np.sin(angle_rad))
        total_points += 1
        if y < gray_img.shape[0] and x < gray_img.shape[1]:
            if gray_img[y, x] > 0:  # edge pixel detected
                edge_points += 1
    # Consider it a quarter arc if at least 60% of sampled points have edges
    return edge_points / total_points > 0.6

def detect_doors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    detected_doors = []

    if lines is None or circles is None:
        return img  # nothing detected

    circles = np.uint16(np.around(circles[0]))

    for circle in circles:
        cx, cy, r = circle

        if not is_quarter_arc(edges, (cx, cy), r):
            continue  # skip if not quarter arc

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Distances of endpoints to circle center
            dist1 = abs(point_distance((x1, y1), (cx, cy)) - r)
            dist2 = abs(point_distance((x2, y2), (cx, cy)) - r)

            tolerance = 12  # pixels

            # Exactly one endpoint near circle circumference
            endpoint1_near = dist1 < tolerance
            endpoint2_near = dist2 < tolerance

            if endpoint1_near ^ endpoint2_near:  # XOR: only one endpoint connected
                # The other endpoint should be free: far from circle and other lines (simple check)
                free_endpoint = (x2, y2) if endpoint1_near else (x1, y1)
                dist_to_circle = point_distance(free_endpoint, (cx, cy))

                # Check if free_endpoint is not near circle circumference or other lines endpoints
                if dist_to_circle > r + 20:
                    detected_doors.append({
                        "circle": (cx, cy, r),
                        "line": (x1, y1, x2, y2)
                    })

    # Draw detected doors
    for door in detected_doors:
        cx, cy, r = door["circle"]
        x1, y1, x2, y2 = door["line"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # green line
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)       # red arc

    return img

def main():
    print("Converting PDF to images...")
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i}.png")
        page.save(image_path, 'PNG')
        print(f"Saved page {i} as image: {image_path}")

        img = cv2.imread(image_path)
        img = resize_image(img, 40)

        print("Detecting doors...")
        detected_img = detect_doors(img)

        result_path = os.path.join(output_folder, f"detected_{i}.png")
        cv2.imwrite(result_path, detected_img)
        print(f"Saved detection result: {result_path}")

if __name__ == "__main__":
    main()
