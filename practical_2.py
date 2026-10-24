# Practical 2: Regression-Based Machine Learning Model
# Predicts a continuous target variable using Linear Regression.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load diabetes dataset (real-world medical dataset for regression)
data = load_diabetes()
X, y = data.data, data.target

# 2. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on test set
y_pred = model.predict(X_test)

# 5. Evaluate model performance
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score (Coefficient of Determination):", r2_score(y_test, y_pred))
