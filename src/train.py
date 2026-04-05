import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create model folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/resume.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns)

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

df["Resume_str"] = df["Resume_str"].apply(clean_text)

# Features
X = df["Resume_str"]
y = df["Category"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)

# Train
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("================================")
print("Model trained successfully")
print("Accuracy:", acc)
print("================================")