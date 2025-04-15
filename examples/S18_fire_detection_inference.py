import cv2
import numpy as np
import glob
from joblib import load


clf = load('S18/classifier.z')


for item in glob.glob("S18/test_data\\*"):
    img = cv2.imread(item)
    r_img = cv2.resize(img, (32, 32))
    r_img = r_img/255
    r_img = r_img.flatten()
    r_img = np.array([r_img])


    # print(clf.predict(r_img))
    pred = clf.predict(r_img)[0]

    cv2.putText(img, pred, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 0), 2)
    cv2.imshow('frame', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


