import cv2 as cv
import numpy as np


def empty(a):
    pass


cv.namedWindow("Trackbar")
cv.resizeWindow("Trackbar", 640, 250)
cv.createTrackbar("Min Hue", "Trackbar", 0, 179, empty)
cv.createTrackbar("Min Sat", "Trackbar", 0, 255, empty)
cv.createTrackbar("Min Val", "Trackbar", 0, 255, empty)
cv.createTrackbar("Max Hue", "Trackbar", 0, 179, empty)
cv.createTrackbar("Max Sat", "Trackbar", 0, 255, empty)
cv.createTrackbar("Max Val", "Trackbar", 0, 255, empty)

while True:
    image = cv.imread("images/S9_Lambo.png")
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    min_h = cv.getTrackbarPos("Min Hue", "Trackbar")
    min_s = cv.getTrackbarPos("Min Sat", "Trackbar")
    min_v = cv.getTrackbarPos("Min Val", "Trackbar")
    max_h = cv.getTrackbarPos("Max Hue", "Trackbar")
    max_s = cv.getTrackbarPos("Max Sat", "Trackbar")
    max_v = cv.getTrackbarPos("Max Val", "Trackbar")

    print(f"HSV Values: {min_h, min_s, min_v, max_h, max_s, max_v}")

    min = np.array([min_h, min_s, min_v])
    max = np.array([max_h, max_s, max_v])
    mask = cv.inRange(hsv_image, min, max)
    masked_image = cv.bitwise_and(image, image, mask=mask)

    cv.imshow("original", image)
    cv.imshow("mask", mask)
    cv.imshow("HSV image", hsv_image)
    cv.imshow("masked", masked_image)
    cv.waitKey(1)

cv.imshow("Trackbar", image)
cv.waitKey(0)
cv.destroyAllWindows()
