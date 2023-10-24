import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data from Google Sheets (replace with your actual Google Sheets URL)
google_sheets_url = 'your_google_sheets_url_here'
data = pd.read_csv(google_sheets_url)

# Map the satisfaction levels to binary labels (Satisfied/Not Satisfied)
satisfaction_mapping = {
    'Very Satisfied': 1,
    'Satisfied': 1,
    'Neutral': 0,
    'Not Satisfied': 0,
    'Very Not Satisfied': 0
}
data['Satisfaction'] = data['How satisfied are you with our product?'].map(satisfaction_mapping)

# Select features and target variable
X = data[['How often do you use our product?', 'On a scale of 1 to 10, how likely are you to recommend our product?']]
y = data['Satisfaction']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['How often do you use our product?'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
