import numpy as np
from sklearn.linear_model import LogisticRegression

hours_studied = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Reshape for scikit-learn

# Generate a bit of random noise to add to the scores

exam_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

print("Hours Studied (X):\n", hours_studied)
print("\nExam Scores (y):\n", exam_result)
model = LogisticRegression()

model.fit(hours_studied, exam_result)


# Get the learned parameters (slope and intercept)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"Learned Slope (w1): {slope:.2f}")
# print(f"Learned Intercept (w0): {intercept:.2f}")

# Make a prediction for a new data point
hours_new = np.array([[7]])

predicted_score = model.predict_proba(hours_new)
# print(f"\nPredicted score for 7 hours of study: {predicted_score[0]:.2f}")