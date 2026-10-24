# Practical 3: Logistic Regression for Classification
# Solves a classification problem using Logistic Regression.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Iris dataset (classification problem)
data = load_iris()
X, y = data.data, data.target

# 2. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train Logistic Regression model
model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
