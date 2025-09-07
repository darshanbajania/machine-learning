import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib
# Load the dataset
# Make sure the file path is correct based on where you saved the dataset
df = pd.read_csv("../../datasets/creditcard.csv")

# Print the first 5 rows of the DataFrame
print(df.head())

# Print the number of rows and columns
print("\nDataFrame shape:", df.shape)

print(df['Class'].value_counts())


# --- Addressing the Class Imbalance with Undersampling ---
# 1. Separate the majority and minority classes
fraudulent_df = df[df['Class'] == 1]
non_fraudulent_df = df[df['Class'] == 0]

# 2. Randomly undersample the non-fraudulent transactions
# We'll match the number of fraudulent transactions (492)
# The `replace=False` argument ensures we don't pick the same row twice
undersampled_non_fraudulent = non_fraudulent_df.sample(n=len(fraudulent_df), random_state=42, replace=False)

# 3. Concatenate the two DataFrames to create the final balanced DataFrame
balanced_df = pd.concat([undersampled_non_fraudulent, fraudulent_df])

# 4. Shuffle the new DataFrame to mix the classes
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new class distribution
print("\nClass distribution after undersampling:")
print(balanced_df['Class'].value_counts())
print("\nNew DataFrame shape:", balanced_df.shape)


# --- Training and Evaluating the Random Forest Model ---
# 1. Define features (X) and target (y) from the balanced DataFrame
# We drop 'Time' and 'Amount' because they might not be useful for the model
X = balanced_df.drop(columns=['Time', 'Amount', 'Class'])
y = balanced_df['Class']

# 2. Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # 3. Initialize the Random Forest Classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # 4. Train the model on the training data
# rf_model.fit(X_train, y_train)

# # 5. Make predictions on the test data
# y_pred = rf_model.predict(X_test)

# # 6. Print the classification report
# print("\nRandom Forest Classifier Report:")
# print(classification_report(y_test, y_pred))



# # 3. Initialize the Support Vector Machine Classifier
# # Note: SVM can be very slow on larger datasets, so we'll use a linear kernel for faster training
# svm_model = SVC(kernel='linear', random_state=42)

# # 4. Train the model on the training data
# svm_model.fit(X_train, y_train)

# # 5. Make predictions on the test data
# y_pred = svm_model.predict(X_test)

# # 6. Print the classification report
# print("\nSupport Vector Machine Classifier Report:")
# print(classification_report(y_test, y_pred))


# 3. Initialize the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# 4. Train the model on the training data
gb_model.fit(X_train, y_train)

# 5. Make predictions on the test data
y_pred = gb_model.predict(X_test)

# 6. Print the classification report
print("\nGradient Boosting Classifier Report:")
print(classification_report(y_test, y_pred))

# --- Analyzing Feature Importance ---
# Get the feature importances from the trained model
importances = gb_model.feature_importances_

# Create a Series to easily view the feature names and their importance scores
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Print the top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))


# --- Saving the Best-Performing Model ---
# Save the trained Gradient Boosting model to a file
joblib.dump(gb_model, 'gradient_boosting_model.joblib')

print("\nGradient Boosting model saved as gradient_boosting_model.joblib")
