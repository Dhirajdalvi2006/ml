import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([10, 20, 30, 40, 50, 60])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

m = LinearRegression().fit(X_train, y_train)

print("Prediction:", m.predict(X_test))
print("MSE:", mean_squared_error(y_test, m.predict(X_test)))
print("R2 Score:", m.score(X_test, y_test))