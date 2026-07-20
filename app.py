import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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


# ---------------- UI ----------------

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")

st.write(
    "Paste an email below and click **Predict** to check whether it is Spam or Ham."
)

email = st.text_area(
    "Email Text",
    height=250
)

if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email.")

    else:

        result = predict(email)

        if result == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Ham Email")