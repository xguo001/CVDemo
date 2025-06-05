import cv2
import numpy as np
import matplotlib.pyplot as plt

def debug_black_line_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image:", image_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_images/debug_gray.png", gray)

    # Save histogram of grayscale
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title("Grayscale histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.savefig("output_images/debug_histogram.png")
    plt.close()

    # Threshold for dark pixels (adjust threshold value here)
    threshold_val = 120  # you can try increasing to 80 or 100
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("output_images/debug_threshold_mask.png", thresh)

    print(f"Saved grayscale image, histogram and threshold mask to output_images/")

if __name__ == "__main__":
    debug_black_line_detection("output_images/page_0.png")
