
import joblib

with open("check_pickles_output.txt", "w") as f:
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        f.write("Vectorizer type: " + str(type(vectorizer)) + "\n")
        f.write("Vectorizer attributes: " + str(dir(vectorizer)) + "\n")
        try:
            f.write("Has idf_?: " + str(hasattr(vectorizer, 'idf_')) + "\n")
            if hasattr(vectorizer, 'idf_'):
                f.write("idf_ shape: " + str(vectorizer.idf_.shape) + "\n")
            f.write("Has vocabulary_?: " + str(hasattr(vectorizer, 'vocabulary_')) + "\n")
        except Exception as e:
            f.write("Error checking vectorizer: " + str(e) + "\n")

        model = joblib.load("spam_classifier.pkl")
        f.write("\nModel type: " + str(type(model)) + "\n")
    except Exception as e:
        f.write("Error: " + str(e) + "\n")
