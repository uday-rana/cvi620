import cv2
import numpy as np

image = cv2.imread("Lucy.jpg")

(h, w) = image.shape[:2]

center_x = w//2
center_y = h//2

center = (center_x, center_y)

angel = 90
scale = 0.5

rotation_matrix = cv2.getRotationMatrix2D(center, angel, scale)
print(rotation_matrix)


rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h))


cv2.imshow('frame', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()