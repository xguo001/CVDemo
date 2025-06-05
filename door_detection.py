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

def detect_doors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=10)
    # Detect circles (door arcs)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=60)

    detected_doors = []

    if lines is not None and circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for circle in circles:
            cx, cy, r = circle
            # For simplicity, treat the circle as a potential arc if radius is reasonable
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if one endpoint of line is near circle perimeter (within some tolerance)
                dist1 = abs(point_distance((x1,y1), (cx,cy)) - r)
                dist2 = abs(point_distance((x2,y2), (cx,cy)) - r)

                tolerance = 15  # pixels
                if dist1 < tolerance or dist2 < tolerance:
                    # Potential door candidate found
                    detected_doors.append({
                        "circle": (cx, cy, r),
                        "line": (x1, y1, x2, y2)
                    })

    # Draw detected doors
    for door in detected_doors:
        cx, cy, r = door["circle"]
        x1, y1, x2, y2 = door["line"]
        # Draw line in green
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw circle arc in red
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 3)

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
