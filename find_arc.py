import cv2
import numpy as np
import math
import os

input_image_path = "output_images/page_0.png"  # Change to your image name
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def angle_between_points(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def fit_circle_to_points(points):
    # Algebraic circle fit
    x = points[:,0]
    y = points[:,1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suu = np.sum(u*u)
    Suv = np.sum(u*v)
    Svv = np.sum(v*v)
    Suuu = np.sum(u*u*u)
    Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v)
    Svuu = np.sum(v*u*u)

    A = np.array([[Suu, Suv],[Suv, Svv]])
    B = np.array([0.5*(Suuu + Suvv), 0.5*(Svvv + Svuu)])

    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None

    cx = x_m + uc
    cy = y_m + vc
    r = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
    return cx, cy, r

def is_quarter_arc(contour, cx, cy, r):
    points = contour[:,0,:]
    angles = []
    for p in points:
        angle = angle_between_points((cx, cy), p)
        angles.append(angle)
    angle_min = min(angles)
    angle_max = max(angles)
    angle_range = angle_max - angle_min
    if angle_range < 0:
        angle_range += 360
    return 20 <= angle_range <= 130  # relaxed quarter arc approx range

def mean_fit_error(contour, cx, cy, r):
    points = contour[:,0,:]
    dists = np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2)
    errors = np.abs(dists - r)
    return np.mean(errors)

def main():
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold black color (invert for black shapes)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Save threshold image for debug
    cv2.imwrite(os.path.join(output_folder, "thresh.png"), thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(f"Total contours found: {len(contours)}")

    # Draw all contours in blue for debug
    debug_img = img.copy()
    cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(output_folder, "all_contours.png"), debug_img)

    # Filter arcs
    arcs = []
    arc_img = img.copy()
    for cnt in contours:
        if len(cnt) < 5:
            continue

        circle_params = fit_circle_to_points(cnt[:,0,:])
        if circle_params is None:
            continue

        cx, cy, r = circle_params

        if not is_quarter_arc(cnt, cx, cy, r):
            continue

        err = mean_fit_error(cnt, cx, cy, r)
        if err > 5:  # relaxed error threshold
            continue

        arcs.append(cnt)
        # Draw detected arc in green
        cv2.drawContours(arc_img, [cnt], -1, (0,255,0), 2)

    print(f"Detected arcs: {len(arcs)}")

    cv2.imwrite(os.path.join(output_folder, "detected_arcs.png"), arc_img)

if __name__ == "__main__":
    main()
