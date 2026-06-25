import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("student_data.csv")

print("First 5 rows:")
print(df.head())

# 2. Encode target column
label_encoder = LabelEncoder()
df["Result"] = label_encoder.fit_transform(df["Result"])   # Fail=0, Pass=1

print("\nEncoded Data:")
print(df.head())

# 3. Split features and target
X = df.drop("Result", axis=1)
y = df["Result"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Predict for a new student
new_student = pd.DataFrame([{
    "StudyHours": 4,
    "Attendance": 73,
    "PreviousScore": 58,
    "AssignmentsCompleted": 1
}])

prediction = model.predict(new_student)[0]
result = label_encoder.inverse_transform([prediction])[0]

print("\nNew Student Prediction:", result)

# 9. Visualize the decision tree
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True
)
plt.title("Decision Tree - Student Pass/Fail Prediction")
plt.show()