# train_content_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading SMS spam dataset from 'spam.csv'...")
try:
    # Use encoding='latin-1' for compatibility
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("\nError: 'spam.csv' not found.")
    exit()

# This part is crucial: it selects only the two columns we need.
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
print("Dataset loaded and columns renamed.")

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("Labels converted to numerical format.")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

print("Training the content analysis model (TF-IDF + Logistic Regression)...")
content_classifier = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

content_classifier.fit(X_train, y_train)
print("Model training complete.")

y_pred = content_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Content Classifier: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Suspicious (1)']))

with open('content_model.pkl', 'wb') as f:
    pickle.dump(content_classifier, f)

print("\nTrained content analysis model saved as 'content_model.pkl'")