from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
m = DecisionTreeClassifier(random_state=0).fit(X, y)

p = m.predict(X)
print("Accuracy:", accuracy_score(y, p))