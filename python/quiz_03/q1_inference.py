import cv2 as cv
import numpy as np
import glob
from joblib import load

IMG_SIZE = (32, 32)

clf = load("knn_classifier.z")
lr = load("logistic_regression.z")

for item in glob.glob("data/Cat_Dog_Dataset/test\\*\\*"):
    img = cv.imread(item)
    r_img = cv.resize(img, IMG_SIZE)
    r_img = r_img / 255
    r_img = r_img.flatten()
    r_img = np.array([r_img])

    knn_pred = clf.predict(r_img)[0]
    lr_pred = lr.predict(r_img)[0]
    print(f"knn prediction: {knn_pred}, logistic regression prediction: {lr_pred}")

    cv.putText(
        img, f"KNN: {knn_pred}", (32, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
    )
    cv.putText(
        img, f"LR: {lr_pred}", (32, 64), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
    )
    cv.imshow("predictions", img)
    cv.waitKey(0)


cv.destroyAllWindows()
