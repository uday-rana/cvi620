import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Data
def preprocess_data():
    dataset = pd.read_csv("data/MNIST_Dataset/mnist_train.csv", header=None)
    data_list = dataset.iloc[:, 1:]
    label_list = dataset.iloc[:, 0]

    return train_test_split(data_list, label_list, test_size=0.2)


x_train, x_test, y_train, y_test = preprocess_data()

# Models
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

lr = LogisticRegression()
lr.fit(x_train, y_train)

# Evaluation
clf_predictions = clf.predict(x_test)
knn_accuracy = accuracy_score(y_test, clf_predictions)

lr_predictions = lr.predict(x_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"knn accuracy: {knn_accuracy}, logistic regression accuracy: {lr_accuracy}")

# Save models
dump(clf, "mnist_knn_classifier.z")
dump(lr, "mnist_logistic_regression.z")
