import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import tensorflow as tf

IMG_SIZE = (32, 32)


# Data
def preprocess_data():
    le = sklearn.preprocessing.LabelEncoder()
    data_list = []
    label_list = []

    for i, address in enumerate(glob.glob("data\kapcha\\*\\*")):
        print(f"i: {i}, address:{address}")
        img = cv.imread(address)
        img = cv.resize(img, IMG_SIZE)
        img = img / 255
        label = address.split("\\")[-2]

        data_list.append(img)
        label_list.append(label)

    data_list = np.array(data_list)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data_list, label_list, test_size=0.2
    )

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train, x_test), (y_train, y_test)


(x_train, x_test), (y_train, y_test) = preprocess_data()


# Model
def train_model():
    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(32, 32, 3)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(9, activation="softmax"),
        ]
    )

    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics="accuracy")

    H = model.fit(
        aug.flow(x_train, y_train, batch_size=32),
        validation_data=(x_test, y_test),
        batch_size=64,
        epochs=10,
    )

    return model, H


model, H = train_model()


# Evaluation
def evaluate_model():
    fig = plt.figure()
    ax_array = fig.subplots(ncols=2)
    ax_array[0].plot(H.history["accuracy"], label="Training accuracy")
    ax_array[0].plot(H.history["val_accuracy"], label="Testing accuracy")
    ax_array[0].legend()
    ax_array[0].set_xlabel("Epochs")
    ax_array[0].set_ylabel("Accuracy")

    ax_array[1].plot(H.history["loss"], label="Training loss")
    ax_array[1].plot(H.history["val_loss"], label="Testing loss")
    ax_array[1].legend()
    ax_array[1].set_xlabel("Epochs")
    ax_array[1].set_ylabel("Loss")

    plt.show()


evaluate_model()
