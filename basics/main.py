import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Our simple dataset
hours_studied = np.array([2, 3, 4, 5, 6]).reshape(-1, 1)  # Reshape for scikit-learn

# Generate a bit of random noise to add to the scores
noise = np.random.normal(loc=0, scale=5, size=5)
exam_scores = np.array([50, 60, 75, 80, 90]) + noise

# Display the data to verify it
print("Hours Studied (X):\n", hours_studied)
print("\nExam Scores (y):\n", exam_scores)

# plt.figure(figsize=(8, 6))
# plt.scatter(hours_studied, exam_scores, color='blue', label='Actual Data')
# plt.title('Exam Score vs. Hours Studied')
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.grid(True)
# plt.legend()
# plt.show()

steps = [
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
]
# Create a Linear Regression model instance
model = LinearRegression()
polynomial_model = Pipeline(steps)

# Train the model using the .fit() method
# This is where the model calculates the best-fit line (finds the optimal weights and bias)
model.fit(hours_studied, exam_scores)
polynomial_model.fit(hours_studied, exam_scores)

# Get the learned parameters (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Learned Slope (w1): {slope:.2f}")
print(f"Learned Intercept (w0): {intercept:.2f}")

# Make a prediction for a new data point
hours_new = np.array([[7]])
predicted_score = model.predict(hours_new)
pm_predicted_score = polynomial_model.predict(hours_new)
print(f"\nPredicted score for 7 hours of study: {predicted_score[0]:.2f}")
print(f"\nPredicted score for 7 hours of study PM: {pm_predicted_score[0]:.2f}")

# Plot the regression line on top of the original data
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, exam_scores, color='blue', label='Actual Data')
plt.plot(hours_studied, model.predict(hours_studied), color='red', linewidth=2, label='Regression Line')
plt.plot(hours_studied, polynomial_model.predict(hours_studied), color='green', linewidth=2, label='Regression Line')
plt.title('Linear Regression Model Fit')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.legend()
plt.show()

# Evaluate the model using the R-squared score
r_squared = model.score(hours_studied, exam_scores)
r_squared_pm = polynomial_model.score(hours_studied, exam_scores)
print(f"R-squared score: {r_squared:.4f}")
print(f"R-squared score PM: {r_squared_pm:.4f}")