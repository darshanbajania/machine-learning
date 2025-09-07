import pandas as pd
import re
from nltk.corpus import stopwords
import string
import pickle

# --- Save your trained vectorizer and model (Do this once after training) ---
# Assuming you have trained a vectorizer and a model named 'vectorizer' and 'model'
# with open('sentiment_vectorizer.pkl', 'wb') as file:
#     pickle.dump(vectorizer, file)
# with open('sentiment_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# --- Start of the code to use your saved model ---
# 1. Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('sentiment_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)


# 2. Define a new, single review to predict
new_review = "This movie was nice"

# 3. Apply the SAME text cleaning functions as before
# These functions must be defined or imported here as well
def remove_punctuation(text):
    return re.sub(r'[^a-z0-9\s]', '', text)

def remove_stopwords(text):
    negation_words = {'not', 'no'}
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = stop_words - negation_words
    return ' '.join([word for word in str(text).split() if word not in filtered_stopwords])

cleaned_review = new_review.lower()
cleaned_review = remove_punctuation(cleaned_review)
cleaned_review = remove_stopwords(cleaned_review)


# 4. Transform the cleaned text into numerical features
# Use the loaded vectorizer's .transform() method, not .fit_transform()
X_new = loaded_vectorizer.transform([cleaned_review])

# 5. Make a prediction on the new data
prediction = loaded_model.predict(X_new)

print(f"The review is: {prediction[0]}")