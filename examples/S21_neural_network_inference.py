import cv2
import numpy as np
from keras import models

IMG_SIZE = (32, 32)

labels = ["fire", "nonfire"]

img = cv2.imread("S21/test_image.jpg")
img = cv2.resize(img, IMG_SIZE)
img = img / 255
img = img.flatten()
img = np.array([img])

net = models.load_model("S21/net.h5")
pred = net.predict(img)
max_pred = np.argmax(pred)
out = labels[max_pred]

print(out)
