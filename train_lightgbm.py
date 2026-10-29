import pandas as pd
import lightgbm as lgb
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "oa_risk_training_data.csv"
MODEL_PATH = BASE_DIR / "lightgbm_oa_risk_model.pkl"


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("=" * 60)
print("LOADING OA RISK DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# =========================================================
# 3. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["risk_level"])
y = df["risk_level"]


# =========================================================
# 4. ENCODE RISK LEVEL
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nRisk classes:")
for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 6. CREATE LIGHTGBM MODEL
# =========================================================

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,

    n_estimators=300,
    learning_rate=0.05,

    num_leaves=31,
    max_depth=-1,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)


# =========================================================
# 7. TRAIN
# =========================================================

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("\nLightGBM training completed successfully!")


# =========================================================
# 8. PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance.to_string(index=False))


# =========================================================
# 11. SAVE MODEL
# =========================================================

model_package = {
    "model": model,
    "label_encoder": label_encoder,
    "features": list(X.columns)
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")

print("\nLightGBM OA Risk Model is ready!")