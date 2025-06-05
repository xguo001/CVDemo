import cv2
import numpy as np

def rotate_templates(template):
    rotations = []
    for angle in [0, 90, 180, 270]:
        (h, w) = template.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        rotations.append(rotated)
    return rotations

def preprocess_patch(patch):
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    patch = cv2.equalizeHist(patch)
    return patch

# === Load image ===
image_path = "your_marked_floorplan.png"  # replace this!
img = cv2.imread(image_path)
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# === Click to mark doors ===
print("[INFO] Click real doors. ESC when done.")
coords = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Doors", img)

cv2.imshow("Click Doors", img)
cv2.setMouseCallback("Click Doors", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[INFO] Marked {len(coords)} doors")

# === Prepare templates ===
patch_size = 50
samples = []
for (x, y) in coords:
    patch = edges[y - patch_size//2:y + patch_size//2, x - patch_size//2:x + patch_size//2]
    if patch.shape == (patch_size, patch_size):
        samples.append(preprocess_patch(patch))

if not samples:
    print("[ERROR] No valid patches.")
    exit()

avg_template = np.mean(samples, axis=0).astype(np.uint8)
templates = rotate_templates(avg_template)

# === Match templates ===
match_count = 0
threshold = 0.65

for tmpl in templates:
    result = cv2.matchTemplate(edges, tmpl, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= threshold)
    for pt in zip(*locs[::-1]):
        cv2.rectangle(orig, pt, (pt[0]+patch_size, pt[1]+patch_size), (0, 255, 0), 2)
        match_count += 1

print(f"[INFO] Detected {match_count} doors.")
cv2.imwrite("detected_doors_improved.png", orig)
print("[INFO] Result saved to detected_doors_improved.png")
