# Practical 4: Naive Bayes Spam Classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

emails = [
    'Win a free iPhone now click here', 'Hi Bob, meeting at 3pm today',
    'Congratulations you won lottery cash prize', 'Project report attached review please',
    'Claim your free gift card urgently', 'Can we reschedule our call tomorrow?',
    'Get cheap meds without prescription', 'Do not forget to submit your assignment'
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]
X = CountVectorizer().fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=.25, random_state=42, stratify=labels
)
pred = MultinomialNB().fit(X_train, y_train).predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
print('\nReport:\n', classification_report(y_test, pred, zero_division=0))
