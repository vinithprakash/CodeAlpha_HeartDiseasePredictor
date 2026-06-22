"""
CodeAlpha Internship - Task 4: Disease Prediction from Medical Data
generate_dataset.py — Synthetic Heart Disease dataset (Cleveland UCI structure)

Generates 1,000 realistic patient records with the same 13 features as the
UCI Cleveland Heart Disease dataset. Target: 1 = Heart Disease Present, 0 = Absent.
All correlations are medically grounded so the model learns real patterns.
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 99
N = 1000
np.random.seed(RNG_SEED)


def generate():
    rng = np.random.default_rng(RNG_SEED)

    # Demographics
    age  = rng.integers(29, 78, N)
    sex  = rng.choice([0, 1], N, p=[0.33, 0.67])   # 1=Male, 0=Female

    # Chest Pain Type: 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic
    cp   = rng.choice([0, 1, 2, 3], N, p=[0.08, 0.17, 0.28, 0.47])

    # Resting blood pressure (mmHg)
    trestbps = np.clip(rng.normal(130, 18, N), 90, 200).round(0).astype(int)

    # Serum cholesterol (mg/dl)
    chol = np.clip(rng.normal(245, 52, N), 120, 565).round(0).astype(int)

    # Fasting blood sugar > 120 mg/dl  (1=True)
    fbs  = rng.choice([0, 1], N, p=[0.85, 0.15])

    # Resting ECG: 0=normal,1=ST-T wave abnormality,2=LV hypertrophy
    restecg = rng.choice([0, 1, 2], N, p=[0.50, 0.48, 0.02])

    # Max heart rate achieved
    thalach = np.clip(rng.normal(150, 23, N), 70, 202).round(0).astype(int)

    # Exercise-induced angina (1=Yes)
    exang   = rng.choice([0, 1], N, p=[0.67, 0.33])

    # ST depression induced by exercise vs rest (oldpeak)
    oldpeak = np.clip(rng.exponential(1.1, N), 0, 6.2).round(1)

    # Slope of peak exercise ST segment: 0=upsloping,1=flat,2=downsloping
    slope   = rng.choice([0, 1, 2], N, p=[0.22, 0.50, 0.28])

    # Number of major vessels coloured by fluoroscopy (0-3)
    ca      = rng.choice([0, 1, 2, 3], N, p=[0.58, 0.22, 0.13, 0.07])

    # Thalassemia: 1=normal,2=fixed defect,3=reversible defect
    thal    = rng.choice([1, 2, 3], N, p=[0.55, 0.07, 0.38])

    # --- Disease probability (medically correlated) -------------------------
    risk  = (
        0.04 * (age - 40)
        + 0.30 * sex                     # male higher risk
        + 0.55 * (cp == 3).astype(int)   # asymptomatic cp most dangerous
        + 0.012 * (trestbps - 120)
        + 0.003 * (chol - 200)
        + 0.25 * fbs
        + 0.20 * (restecg > 0).astype(int)
        - 0.015 * (thalach - 100)
        + 0.45 * exang
        + 0.30 * oldpeak
        + 0.20 * (slope == 1).astype(int)
        + 0.40 * ca
        + 0.35 * (thal == 3).astype(int)
        + rng.normal(0, 0.6, N)
    )
    prob_disease = 1 / (1 + np.exp(-risk + 1.5))
    target = (rng.random(N) < prob_disease).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target,
    })
    return df


def main():
    df = generate()
    os.makedirs("dataset", exist_ok=True)
    path = "dataset/heart_data.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} records → {path}")
    print(df["target"].value_counts().rename({0:"No Disease", 1:"Disease"}))
    print(df.head())


if __name__ == "__main__":
    main()
