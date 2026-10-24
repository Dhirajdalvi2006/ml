# Practical 4: Spam Email Classification using Naïve Bayes
# Classifies text messages/emails as Spam (1) or Not Spam (0).

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample email dataset
emails = [
    "Win a free iPhone now click here", "Hi Bob, meeting at 3pm today",
    "Congratulations you won lottery cash prize", "Project report attached review please",
    "Claim your free gift card urgently", "Can we reschedule our call tomorrow?",
    "Get cheap meds without prescription", "Don't forget to submit your assignment"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: Spam, 0: Not Spam

# 2. Convert text emails into numerical feature vectors (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# 4. Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
