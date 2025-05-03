# fashion_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load the dataset
# Skip lines with errors and drop rows with missing values
df = pd.read_csv("styles.csv", on_bad_lines='skip')  # Ignores problematic rows
df.dropna(inplace=True)  # Drop rows with missing values

# Optional: Reset index
df.reset_index(drop=True, inplace=True)

# Step 2: Preprocess the data
# Features (independent variables)
X = df[["masterCategory", "subCategory", "articleType", "baseColour", "season", "year"]]

# Target variable (dependent variable)
y = df["gender"]

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Step 3: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, y_train)

# Step 5: Evaluate model (Optional: Check accuracy)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Step 6: Save the trained model using joblib
joblib.dump(model, "fashion_classifier_logistic.pkl")
print("Model saved successfully.")

