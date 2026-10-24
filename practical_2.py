# Practical 2: Linear Regression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('R2 Score:', r2_score(y_test, pred))
