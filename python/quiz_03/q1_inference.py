import cv2 as cv
import numpy as np
import glob
from joblib import load

IMG_SIZE = (32, 32)

clf = load("knn_classifier.z")

for item in glob.glob("data/Cat_Dog_Dataset/test\\*\\*"):
    img = cv.imread(item)
    r_img = cv.resize(img, IMG_SIZE)
    r_img = r_img / 255
    r_img = r_img.flatten()
    r_img = np.array([r_img])

    pred = clf.predict(r_img)[0]

    cv.putText(img, pred, (32, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow("frame", img)
    cv.waitKey(0)

cv.destroyAllWindows()
