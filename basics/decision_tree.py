from sklearn.tree import DecisionTreeClassifier
import numpy as np



height = np.array([155, 160, 165, 170, 175, 180, 185, 190, 195])
weight = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95])

x = np.column_stack((height, weight))


class_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0: Not Fit, 1: Fit



model  = DecisionTreeClassifier()

model.fit(x, class_labels)


person_1 = np.array([[172, 71]])

predicted_probability = model.predict_proba(person_1)
predicted_score = model.predict(person_1)
model_classes = model.classes_
print(f"\nPredicted probability: {predicted_probability}")
print(f"\nPredicted score: {predicted_score}")
print(f"\model classes: {model_classes}")