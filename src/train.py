import pandas as pd
import mlflow
import mlflow.sklearn
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# STEP 1: Create folders
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# STEP 2: Load dataset
# =========================
df = pd.read_csv("data/resume.csv")

print("Dataset loaded successfully")
print("Columns:", df.columns)

# =========================
# STEP 3: Clean text
# =========================
def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)   # remove HTML tags
    text = re.sub(r'[^a-zA-Z ]', '', text)  # remove special chars
    text = text.lower()
    return text

df["Resume_str"] = df["Resume_str"].apply(clean_text)

# =========================
# STEP 4: Define features
# =========================
X = df["Resume_str"]
y = df["Category"]

# =========================
# STEP 5: Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 6: Convert text → numbers
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# STEP 7: Model
# =========================
model = LogisticRegression(max_iter=200)

# =========================
# STEP 8: MLflow tracking
# =========================
import os

USE_MLFLOW = os.getenv("USE_MLFLOW", "1") == "1"

if USE_MLFLOW:
    mlflow.start_run()

# Train
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

if USE_MLFLOW:
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

print("================================")
print("Model trained successfully")
print("Accuracy:", acc)
print("================================")