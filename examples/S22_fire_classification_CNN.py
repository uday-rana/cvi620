import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers

import warnings

warnings.filterwarnings("ignore")

IMG_SIZE = (32, 32)


# DATA
def preprocess_data():
    data_list = []
    label_list = []
    le = LabelEncoder()

    for i, address in enumerate(glob.glob("S18/fire_dataset\\*\\*")):
        img = cv2.imread(address)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255
        # img = img.flatten()

        data_list.append(img)

        label = address.split("\\")[-1].split(".")[0]
        label_list.append(label)

        if i % 100 == 0:
            print(f"[INFO]: {i}/1000 processed!")

    data_list = np.array(data_list)

    X_train, X_test, y_train, y_test = train_test_split(
        data_list, label_list, test_size=0.2
    )

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print(f"before one hot: {y_train}")

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(f"after one hot: {y_train}")
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_data()


# MODEL
net = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation="relu"),
    ]
)


net.compile(optimizer="adam", loss="MSE", metrics="accuracy")  # mse, mae

net.build()
print(net.summary())

H = net.fit(
    X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test)
)


# Evaluation
net.save("S21/classification.h5")
