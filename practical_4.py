from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [
    "win money now", "claim your prize", "free offer",
    "hello friend", "meeting tomorrow", "good morning",
    "win a free prize", "let us meet today",
    "claim free money", "see you tomorrow"
]
y = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]

v = CountVectorizer()
X = v.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

m = MultinomialNB().fit(X_train, y_train)
p = m.predict(X_test)

print("Prediction:", ["Spam" if i else "Not Spam" for i in p])
print("Accuracy:", accuracy_score(y_test, p))
print("New Message:", "Spam" if m.predict(v.transform(["win free money"]))[0] else "Not Spam")