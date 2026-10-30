import joblib
import pandas as pd
from pathlib import Path


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "lightgbm_oa_risk_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading LightGBM OA Risk Model...")

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
label_encoder = model_package["label_encoder"]
FEATURES = model_package["features"]

print("LightGBM OA Risk Model loaded successfully!")


# =========================================================
# RISK PREDICTION FUNCTION
# =========================================================

def predict_risk(patient_data):

    # Create dataframe using the exact training feature order
    input_data = pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )

    # Predict class
    predicted_class = model.predict(input_data)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Convert numeric class back to name
    risk_level = label_encoder.inverse_transform(
        [int(predicted_class)]
    )[0]

    # Probability of predicted class
    confidence = float(
        probabilities[int(predicted_class)]
    )

    return {
        "risk_level": risk_level,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            label_encoder.inverse_transform([i])[0]:
            round(float(probabilities[i]) * 100, 2)
            for i in range(len(probabilities))
        }
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_patient = {

        "age": 60,

        "pain_score": 7,

        "morning_stiffness": 1,

        "walking_difficulty": 1,

        "stairs_difficulty": 1,

        "knee_swelling": 1,

        "previous_knee_injury": 1,

        "thigh_acceleration": 1.2,

        "shank_acceleration": 1.1,

        "thigh_angular_velocity": 1.3,

        "shank_angular_velocity": 1.2,

        "knee_range_of_motion": 90,

        "gait_speed": 0.6,

        "emg_activity": 0.7,

        "pressure_peak": 70,

        "pressure_asymmetry": 22
    }

    result = predict_risk(test_patient)

    print("\n" + "=" * 60)
    print("OA RISK PREDICTION TEST")
    print("=" * 60)

    print("Risk Level:", result["risk_level"])
    print("Confidence:", result["confidence"], "%")

    print("\nRisk Probabilities:")

    for risk, probability in result["probabilities"].items():
        print(f"{risk}: {probability}%")

    print("=" * 60)