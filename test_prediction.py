
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def predict(email):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction


# Test some examples
test_emails = [
    ("Congratulations! You've won a free iPhone. Click here to claim your prize.", 1),
    ("Hey, are we still meeting for lunch tomorrow?", 0),
    ("Win a $1000 gift card now! Limited time offer.", 1),
    ("Reminder: Project deadline is next week. Please submit your work.", 0)
]

print("Testing predictions:\n")
all_passed = True
for email, expected in test_emails:
    result = predict(email)
    status = "✓" if result == expected else "✗"
    print(f"{status} Email: {email[:50]}... | Expected: {'Spam' if expected else 'Ham'} | Got: {'Spam' if result else 'Ham'}")
    if result != expected:
        all_passed = False

print("\nAll tests passed!" if all_passed else "\nSome tests failed!")
