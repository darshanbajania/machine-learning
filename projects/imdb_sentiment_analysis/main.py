import pandas as pd
import re
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import pickle


# import nltk
# import ssl

# try:
#      _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#      pass
# else:
#      ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')

# Load the data
df = pd.read_csv("../../datasets/IMDB_dataset.csv")

# 1. Remove rows with missing values
# The .dropna() function is the easiest way to do this.
# It will remove any row that has a missing value (like NaN or None).
# The 'inplace=True' argument modifies the original DataFrame.
df.dropna(inplace=True)

# 2. Convert all text to lowercase
# We can do this by applying the .str.lower() method to the 'review' column.
df['review'] = df['review'].str.lower()

# 3. Remove punctuation
# We can use the 're' module for this. The code below replaces
# all punctuation marks with an empty string.
def remove_punctuation(text):
    return re.sub(r'[^a-z0-9\s]', '', text)

df['review'] = df['review'].apply(remove_punctuation)

# 4. Remove stop words
# We can use the stopwords list from the NLTK library.
# The code below removes the common words like "the", "a", "is", etc.

# stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    negation_words = {'not', 'no'}
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = stop_words - negation_words
    return ' '.join([word for word in str(text).split() if word not in filtered_stopwords])

df['review'] = df['review'].apply(remove_stopwords)

# Print the head of the cleaned DataFrame to see the changes
print(df.head())
print(df.shape)


# # Initialize the CountVectorizer
# # We can set parameters to filter out words
# # min_df=5 means ignore words that appear in less than 5 reviews
# # max_df=0.8 means ignore words that appear in more than 80% of reviews
# vectorizer = CountVectorizer(min_df=5, max_df=0.8)

# # Fit and transform the 'review' column
# # This step builds the vocabulary and converts the text into a sparse matrix
# X = vectorizer.fit_transform(df['review'])

# # Print the shape of the resulting matrix
# # The first number is the number of reviews (rows)
# # The second number is the size of the vocabulary (columns)
# print(X.shape)


# Initialize the TfidfVectorizer
# We can use the same parameters to filter out common/rare words
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))


# Fit and transform the 'review' column
# This step builds the vocabulary and converts the text into a sparse matrix
X = vectorizer.fit_transform(df['review'])


with open('sentiment_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# Print the shape of the resulting matrix
print(X.shape)

# Our labels (the sentiment)
y = df['sentiment']

# Perform the 60:40 train-test split
# The 'random_state' ensures we get the same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# 1. Make predictions on the test data
y_pred = model.predict(X_test)

# 2. Print the classification report
# This compares our model's predictions (y_pred) with the actual labels (y_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# # Define the hyperparameters to search
# param_grid = {'C': [0.1, 1, 10, 20, 100]}
# # Create the GridSearchCV object
# # It will test each 'C' value and use cross-validation to find the best one
# grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')

# # Fit the grid search to the training data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters and the best score
# print("Best C:", grid_search.best_params_['C'])
# print("Best cross-validation accuracy:", grid_search.best_score_)



# # Initialize the Naive Bayes model
# nb_model = MultinomialNB()

# # Train the model
# nb_model.fit(X_train, y_train)

# # Make predictions
# nb_pred = nb_model.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, nb_pred))