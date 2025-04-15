import cv2
import numpy as np


img = cv2.imread("Lenna.png")
kernel1 = np.array([[-1, 1]])
kernel2 = np.array([[-1],
                    [1]])

out_image1 = cv2.filter2D(img, cv2.CV_8U, kernel1)
out_image2 = cv2.filter2D(img, cv2.CV_8U, kernel2)

cv2.imshow("left edge", out_image1)
cv2.imshow("bottom edge", out_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



