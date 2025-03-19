import glob
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


IMG_SIZE = (32, 32)


# Data
def preprocess_data():
    data_list = []
    label_list = []

    for i, address in enumerate(glob.glob("data/Cat_Dog_Dataset/train\\*\\*")):
        print(f"i: {i}, address:{address}")
        img = cv.imread(address)
        img = cv.resize(img, IMG_SIZE)
        img = img / 255
        img = img.flatten()

        label = address.split("\\")[1]

        data_list.append(img)
        label_list.append(label)

    data_list = np.array(data_list)

    return train_test_split(data_list, label_list, test_size=0.2)


x_train, x_test, y_train, y_test = preprocess_data()

# Models
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

# Evaluation
predictions = clf.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print(f"accuracy: {accuracy}")
