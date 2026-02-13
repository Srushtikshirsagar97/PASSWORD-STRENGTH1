# ============================================================
# AI PROJECT: PASSWORD STRENGTH CLASSIFICATION USING ML
# ============================================================

import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================================================
# 1. DATASET GENERATION (10,000 PASSWORDS)
# ============================================================

def generate_password(strength):
    length = {
        "weak": random.randint(4, 7),
        "medium": random.randint(8, 10),
        "strong": random.randint(11, 16)
    }[strength]
  
    chars = ""
    if strength == "weak":
        chars = string.ascii_lowercase
    elif strength == "medium":
        chars = string.ascii_letters + string.digits
    else:
        chars = string.ascii_letters + string.digits + string.punctuation

    return ''.join(random.choice(chars) for _ in range(length))


data = []

for _ in range(3000):
    data.append([generate_password("weak"), "Weak"])

for _ in range(3500):
    data.append([generate_password("medium"), "Medium"])

for _ in range(3500):
    data.append([generate_password("strong"), "Strong"])

df = pd.DataFrame(data, columns=["password", "strength"])
df.to_csv("password_dataset.csv", index=False)
print("✅ Dataset created: password_dataset.csv")

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(pw):
    return {
        "length": len(pw),
        "digits": sum(c.isdigit() for c in pw),
        "upper": sum(c.isupper() for c in pw),
        "lower": sum(c.islower() for c in pw),
        "symbols": sum(c in string.punctuation for c in pw),
        "unique_chars": len(set(pw))
    }

# Create feature table
feature_rows = []
for pw in df["password"]:
    feature_rows.append(extract_features(pw))

features_df = pd.DataFrame(feature_rows)
labels = df["strength"]

print("✅ Features extracted!")

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels, test_size=0.2, random_state=42
)

# ============================================================
# 4. MACHINE LEARNING MODEL TRAINING
# ============================================================

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================================================
# 5. METRICS & RESULTS
# ============================================================

print("\n==========================")
print("🔍 MODEL PERFORMANCE")
print("==========================\n")

print("🔸 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔸 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔸 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================================
# 6. SAVE TRAINED MODEL
# ============================================================

joblib.dump(model, "password_strength_model.pkl")
print("\n✅ Model saved as password_strength_model.pkl")

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_strength(pw):
    feat = extract_features(pw)
    df_input = pd.DataFrame([feat])
    return model.predict(df_input)[0]

# Example Test
test_pw = input("\nEnter any password to test: ")
print("Predicted Strength:", predict_strength(test_pw))

