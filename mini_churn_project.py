import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("churn_small.csv")

print("First 5 rows:")
print(df.head())

# 2. Encode categorical columns
label_encoders = {}

categorical_cols = ["gender", "Contract", "PaymentMethod", "Churn"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoded Data:")
print(df.head())

# 3. Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Predict for a new customer
new_customer = pd.DataFrame([{
    "gender": label_encoders["gender"].transform(["Female"])[0],
    "SeniorCitizen": 0,
    "tenure": 4,
    "MonthlyCharges": 88.5,
    "Contract": label_encoders["Contract"].transform(["Month-to-month"])[0],
    "PaymentMethod": label_encoders["PaymentMethod"].transform(["Electronic check"])[0]
}])

prediction = model.predict(new_customer)[0]
result = label_encoders["Churn"].inverse_transform([prediction])[0]

print("\nNew Customer Prediction:", result)