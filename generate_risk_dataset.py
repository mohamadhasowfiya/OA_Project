import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible results
np.random.seed(42)

# Number of synthetic screening samples
N = 3000

data = {
    "age": np.random.randint(35, 81, N),

    "pain_score": np.random.randint(0, 11, N),

    "morning_stiffness": np.random.randint(0, 2, N),

    "walking_difficulty": np.random.randint(0, 2, N),

    "stairs_difficulty": np.random.randint(0, 2, N),

    "knee_swelling": np.random.randint(0, 2, N),

    "previous_knee_injury": np.random.randint(0, 2, N),

    # IMU features
    "thigh_acceleration": np.round(
        np.random.normal(1.5, 0.5, N), 3
    ),

    "shank_acceleration": np.round(
        np.random.normal(1.4, 0.5, N), 3
    ),

    "thigh_angular_velocity": np.round(
        np.random.normal(1.8, 0.6, N), 3
    ),

    "shank_angular_velocity": np.round(
        np.random.normal(1.7, 0.6, N), 3
    ),

    # Knee movement
    "knee_range_of_motion": np.round(
        np.random.normal(115, 15, N), 2
    ),

    "gait_speed": np.round(
        np.random.normal(1.0, 0.25, N), 3
    ),

    # EMG
    "emg_activity": np.round(
        np.random.normal(0.5, 0.2, N), 3
    ),

    # Pressure sensor
    "pressure_peak": np.round(
        np.random.normal(50, 15, N), 2
    ),

    "pressure_asymmetry": np.round(
        np.random.normal(10, 5, N), 2
    ),
}

df = pd.DataFrame(data)

# Keep physical values within reasonable prototype ranges
df["knee_range_of_motion"] = df["knee_range_of_motion"].clip(70, 140)
df["gait_speed"] = df["gait_speed"].clip(0.3, 1.8)
df["emg_activity"] = df["emg_activity"].clip(0.05, 1.0)
df["pressure_peak"] = df["pressure_peak"].clip(10, 100)
df["pressure_asymmetry"] = df["pressure_asymmetry"].clip(0, 30)

# ---------------------------------------------------------
# Create prototype risk score
# ---------------------------------------------------------
risk_score = (
    df["age"] * 0.08
    + df["pain_score"] * 2.5
    + df["morning_stiffness"] * 5
    + df["walking_difficulty"] * 7
    + df["stairs_difficulty"] * 6
    + df["knee_swelling"] * 5
    + df["previous_knee_injury"] * 4

    # Movement-related indicators
    + (120 - df["knee_range_of_motion"]).clip(lower=0) * 0.15
    + (1.2 - df["gait_speed"]).clip(lower=0) * 8

    # Sensor-related indicators
    + df["pressure_asymmetry"] * 0.5
    + df["pressure_peak"] * 0.05
)

# Add a small amount of randomness so the model
# does not learn a perfectly deterministic rule.
risk_score += np.random.normal(0, 3, N)

# ---------------------------------------------------------
# Convert score into 3 risk categories
# ---------------------------------------------------------
def assign_risk(score):
    if score < 35:
        return "Low"
    elif score < 55:
        return "Medium"
    else:
        return "High"


df["risk_level"] = risk_score.apply(assign_risk)

# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

output_path = BASE_DIR / "oa_risk_training_data.csv"

df.to_csv(output_path, index=False)

print("=" * 60)
print("OA RISK TRAINING DATASET CREATED")
print("=" * 60)

print(f"Total samples: {len(df)}")
print(f"Saved to: {output_path}")

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

print("\nDataset preview:")
print(df.head())

print("\nDataset columns:")
print(list(df.columns))

print("\nDataset creation completed successfully!")