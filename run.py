# run.py (Updated for the new dataset)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Load the new, richer dataset ---
print("Loading new dataset 'access_data.csv'...")
try:
    # The new CSV uses quotes, which pandas handles automatically
    df = pd.read_csv('access_data.csv')
except FileNotFoundError:
    print("Error: 'access_data.csv' not found. Please ensure the file exists and contains the new data.")
    exit()

# --- Feature Engineering and Selection ---
# We will train the model ONLY on features the live app can provide.
# This avoids a crash and makes the model robust.
print("Selecting features compatible with the live application...")
features_to_keep = [
    'user_role',
    'department',
    'access_time',
    'last_login_time',
    'access_granted'
]
df_selected = df[features_to_keep]

# --- Preprocess the data ---
# Convert timestamps and extract hour (same as before)
df_selected['access_time'] = pd.to_datetime(df_selected['access_time'])
df_selected['last_login_time'] = pd.to_datetime(df_selected['last_login_time'])
df_selected['access_time_hour'] = df_selected['access_time'].dt.hour
df_selected['last_login_time_hour'] = df_selected['last_login_time'].dt.hour

# Drop original timestamp columns
df_processed = df_selected.drop(['access_time', 'last_login_time'], axis=1)

# Separate features (X) and target (y)
X = df_processed.drop('access_granted', axis=1)
y = df_processed['access_granted']

# One-hot encode categorical variables
X_processed = pd.get_dummies(X, columns=['user_role', 'department'])

# --- Train the model ---
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("Training the Random Forest Classifier on the new dataset...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# --- Evaluate the model ---
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Random Forest Classifier: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Denied (0)', 'Granted (1)']))

# --- Save the trained model and the column order ---
model_data = {
    'model': rfc,
    'columns': X_train.columns
}
with open('rfc_access_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("\nTrained model and feature columns saved as 'rfc_access_model.pkl'")