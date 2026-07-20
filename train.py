
import sys
import pandas as pd
import joblib
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Redirect stdout to a log file
log_file = open("train_output.log", "w")
sys.stdout = log_file

try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

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

    # Download and load the SMS Spam Collection dataset
    import zipfile
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('SMSSpamCollection') as f:
            df = pd.read_csv(f, sep='\t', names=['label', 'message'])

    # Map labels to 0 (ham) and 1 (spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Clean the text
    df['cleaned'] = df['message'].apply(clean_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vectorized)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, "spam_classifier.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("\nModel and vectorizer saved successfully!")
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc()
finally:
    log_file.close()
    sys.stdout = sys.__stdout__
