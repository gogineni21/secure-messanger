import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from datetime import datetime

# Load the dataset
df = pd.read_csv('access_data.csv')

# Preprocess the data
# Convert timestamps to datetime and extract hour
df['access_time'] = pd.to_datetime(df['access_time'])
df['last_login_time'] = pd.to_datetime(df['last_login_time'])
df['access_time_hour'] = df['access_time'].dt.hour
df['last_login_time_hour'] = df['last_login_time'].dt.hour

# Drop original timestamp columns (we only need the hour)
df = df.drop(['access_time', 'last_login_time'], axis=1)

# Separate features (X) and target (y)
X = df.drop('access_granted', axis=1)
y = df['access_granted']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['user_role', 'department'])

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Denied (0)', 'Granted (1)']))

# Save the trained model to a file
with open('rfc_access_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)
print("Trained model saved as 'rfc_access_model.pkl'")

# Feature importance (optional, for insight)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rfc.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)