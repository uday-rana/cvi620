import cv2
import numpy as np
from keras.models import load_model

net = load_model("S23/Kapcha_Decoder.h5")

img = cv2.imread("S23/digits.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
T, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))

cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

for i in range(len(cnts)):
    x, y, w, h = cv2.boundingRect(cnts[i])

    roi = img[y - 5 : y + h + 5, x - 5 : x + w + 5]
    roi = cv2.resize(roi, (32, 32))
    roi = roi / 255
    roi = np.array([roi])

    output = net.predict(roi)[0]
    max_index = np.argmax(output) + 1
    print(max_index)

    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
    cv2.putText(
        img, str(max_index), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2
    )
    cv2.imshow("img", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
