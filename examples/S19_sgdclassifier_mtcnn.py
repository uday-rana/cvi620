import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from mtcnn import MTCNN
import warnings

warnings.filterwarnings("ignore")

detector = MTCNN()


def face_detector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out["box"]

        return img[y : y + h, x : x + w]

    except:
        pass


def data_preprocess():
    data = []
    labels = []
    print("[INFO] Processing data!")
    for i, address in enumerate(glob.glob("S19\smile_dataset\\*\\*")):
        img = cv2.imread(address)
        face = face_detector(img)

        if face is None:
            continue

        face = cv2.resize(face, (32, 32))
        face = face / 255
        face = face.flatten()

        data.append(face)

        label = address.split("\\")[2]
        labels.append(label)

        # cv2.imshow('image', face)
        # cv2.waitKey(0)
        if i % 100 == 0:
            print(f"[INFO] {i}/3300 data processed!")

    data = np.array(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

    # cv2.destroyAllWindows()


print("[INFO] Start!")
X_train, X_test, y_train, y_test = data_preprocess()

clf = SGDClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_train, predictions))

# dump(clf, 'smile_detectio.z')
