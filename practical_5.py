# Practical 5: Decision Tree Classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, pred))
print('\nReport:\n', classification_report(y_test, pred))
