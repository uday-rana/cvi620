import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# DATA
dataset = pd.read_csv("S16/petrol_consumption.csv")
X = dataset[
    ["Petrol_tax", "Average_income", "Paved_Highways", "Population_Driver_licence(%)"]
]
y = dataset["Petrol_Consumption"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# MODEL
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=["Coefficient"])
# print(coeff_df)
# print(regressor.intercept_)


sgdregressor = SGDRegressor(max_iter=1000)
sgdregressor.fit(X_train, y_train)
coeff_sgd = pd.DataFrame(sgdregressor.coef_, X.columns, columns=["Coefficient"])
print(coeff_sgd)
print(sgdregressor.intercept_)


# RESULTS AND EVALUATION
# y_pred = regressor.predict(X_test)
# print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


y_pred_sgd = sgdregressor.predict(X_test)
print("Mean Absolute Error sgd:", mean_absolute_error(y_test, y_pred_sgd))
print("Mean Squared Error: sgd", mean_squared_error(y_test, y_pred_sgd))
