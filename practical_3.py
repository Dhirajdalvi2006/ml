# Practical 3: Logistic Regression Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=.2, random_state=42
)
model = LogisticRegression(solver='liblinear', max_iter=200).fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print('\nReport:\n', classification_report(y_test, pred, target_names=data.target_names))
