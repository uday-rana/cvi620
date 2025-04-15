import cv2
import numpy as np

image = cv2.imread("Lucy.jpg")

if image is None:
    print("ERROR! Image not available...!")

# Gaussian Noise
noise = np.random.normal(0, 25, image.shape).astype("float32")

noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 255).astype("uint8")

kernel = np.ones((3, 3), dtype=np.float32) / 9
denoised_image = cv2.filter2D(noisy_image, -1, kernel)

cv2.imshow("less noisy", denoised_image)
cv2.imshow("frame", noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
