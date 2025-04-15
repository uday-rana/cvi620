import glob
import cv2
import numpy as np

# from keras.preprocessing import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, save_model
import matplotlib.pyplot as plt


def data_preprocessing():
    images = []
    labels = []

    for i, item in enumerate(glob.glob("S23\kapcha\\*\\*")):
        img = cv2.imread(item)
        img = cv2.resize(img, (32, 32))
        img = img / 255
        images.append(img)

        label = item.split("\\")[2]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO] {i}/2000 data processed!")

    images = np.array(images)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, shuffle=True
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test


def classification_model(X_train, X_test, y_train, y_test):
    net = Sequential(
        [
            Conv2D(16, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPool2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(16, activation="relu"),
            Dense(9, activation="softmax"),
        ]
    )

    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    H = net.fit(
        X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10
    )

    save_model(net, "S23/Kapcha_model_decoder.h5")
    return H


def visualize_results(H):
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="validation loss")
    plt.plot(H.history["accuracy"], label="train acc")
    plt.plot(H.history["val_accuracy"], label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("metrics")
    plt.title("EVALUATIONS")
    plt.legend()
    plt.show()


X_train, X_test, y_train, y_test = data_preprocessing()
H = classification_model(X_train, X_test, y_train, y_test)
visualize_results(H)
