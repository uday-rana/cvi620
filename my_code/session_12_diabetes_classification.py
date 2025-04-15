import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Preprocess
df = pd.read_csv('./data/diabetes.csv')

## Replace 0s with mean value
zero_not_accepted_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in zero_not_accepted_columns:
    # Replace 0 with NaN so they don't affect mean calculation
    df[column] = df[column].replace(0, np.nan)
    mean = int(df[column].mean(skipna=True))
    # Replace NaN with the mean
    df[column] = df[column].replace(np.nan, mean)

## Split features and target
x = df.drop(columns=['Outcome'])
y = df['Outcome']

## Split sample data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Train
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

print(f'Accuracy Score: {accuracy}')