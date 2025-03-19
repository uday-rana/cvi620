import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

# Data
test_dataset = pd.read_csv("data/MNIST_Dataset/mnist_test.csv", header=None)
data_list = test_dataset.iloc[:, 1:].values
label_list = test_dataset.iloc[:, 0].values

# Models
clf = load("mnist_knn_classifier.z")
lr = load("mnist_logistic_regression.z")

# Evaluation
knn_predictions = clf.predict(data_list)
knn_accuracy = accuracy_score(label_list, knn_predictions)

lr_predictions = lr.predict(data_list)
lr_accuracy = accuracy_score(label_list, lr_predictions)

print(f"knn accuracy: {knn_accuracy}, logistic regression accuracy: {lr_accuracy}")
