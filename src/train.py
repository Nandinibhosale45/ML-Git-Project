import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Features and target
X = data[["study_hours", "attendance", "assignments", "previous_score"]]
y = data["result"].map({"Fail": 0, "Pass": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
with open("models/student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")