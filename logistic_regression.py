import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Load dataset
df = pd.read_csv("styles.csv", on_bad_lines='skip')



# Preprocessing: Encoding categorical variables using LabelEncoder
le = LabelEncoder()

# Encode the necessary columns
df['Gender'] = le.fit_transform(df['gender'])
df['Category'] = le.fit_transform(df['masterCategory'])
df['Subcategory'] = le.fit_transform(df['subCategory'])
df['Type'] = le.fit_transform(df['articleType'])
df['Color'] = le.fit_transform(df['baseColour'])
df['Season'] = le.fit_transform(df['season'])
df['Year'] = le.fit_transform(df['year'])
df['Usage'] = le.fit_transform(df['usage'])

# Selecting Features (X) and Target (y)
X = df[['Gender', 'Category', 'Subcategory', 'Type', 'Color', 'Season', 'Year']]
y = df['Usage']  # Assuming we are predicting 'usage' (Casual, Formal, etc.)

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Example prediction
y_pred = model.predict(X_test)

# If using probabilities, convert them to class labels
# y_pred = (model.predict_proba(X_test) >= 0.5).astype(int)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'logistic_model.pkl')
print("Model saved as logistic_model.pkl")

