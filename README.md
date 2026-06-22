# 🫀 CardioScan AI — Heart Disease Risk Analyzer
### CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-71%25-brightgreen)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.752-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

**CardioScan AI** is a clinical-grade Heart Disease Risk Predictor wrapped in
a hospital-themed interactive Streamlit dashboard. It takes 13 standard
cardiac clinical indicators and returns:

- 🎯 **Risk tier** — High / Moderate / Low-Moderate / Low
- 📊 **Probability gauge** (0–100%) with animated needle
- 🔴🟢 **Personalised risk & protective factor pills** — plain-English
  summary of what's driving the score for *this* patient
- 📈 Full model comparison dashboard (4 algorithms head-to-head)
- 🗂️ Clinical dataset explorer with interactive charts

---

## 🖼️ UI Design Language

| Element | Choice |
|---|---|
| **Theme** | Deep navy surgical theatre (`#0d1b2a`) |
| **Accent colours** | Clinical red `#e63946` (risk) · Teal `#2ec4b6` (safe) |
| **Typography** | IBM Plex Mono (vitals & readouts) + Inter (body) |
| **Signature element** | Animated ECG heartbeat strip in the header |
| **Cards** | Glassmorphic vital-stat cards at the top of every session |

---

## 🗂️ Project Structure

```
CodeAlpha_HeartDiseasePredictor/
│
├── app.py                        # Streamlit dashboard (main entry point)
├── train.py                      # Trains 4 models, saves the best
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── .streamlit/
│   └── config.toml               # Dark hospital theme
│
├── dataset/
│   ├── generate_dataset.py       # Synthetic dataset generator
│   └── heart_data.csv            # 1,000 patient records (auto-generated)
│
├── model/
│   ├── heart_model.pkl           # Trained pipeline (best model)
│   ├── metrics.json              # Evaluation metrics for all 4 models
│   └── meta.json                 # Feature names, labels, ranges
│
└── utils/
    ├── model_comparison.png      # Bar chart comparing 4 models
    ├── roc_curve.png             # ROC curves for all 4 models
    ├── confusion_matrix.png      # Confusion matrix (best model)
    ├── feature_importance.png    # Feature importances
    └── age_risk.png              # Disease prevalence by age group
```

---

## 📊 Dataset

A medically-grounded synthetic dataset (1,000 patients) matching the
structure of the **UCI Cleveland Heart Disease** dataset. All 13 features are
correlated with the target through a realistic clinical scoring formula — no
random noise labels.

| Feature | Description |
|---|---|
| `age` | Patient age in years |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0=typical angina … 3=asymptomatic) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar >120 mg/dl (1=True) |
| `restecg` | Resting ECG result (0=normal, 1=ST-T abnormality, 2=LV hypertrophy) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1=Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (1=normal, 2=fixed defect, 3=reversible defect) |
| `target` | **Heart disease present** (1=Yes, 0=No) |

---

## 🤖 Models Compared

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.710 | 0.705 | 0.760 | 0.731 | **0.752** |
| Random Forest | 0.660 | 0.648 | 0.760 | 0.699 | 0.716 |
| Gradient Boosting | 0.625 | 0.633 | 0.663 | 0.648 | 0.649 |
| SVM (RBF) | 0.625 | 0.624 | 0.702 | 0.661 | 0.686 |

**Deployed model:** Logistic Regression — highest ROC-AUC and fully
interpretable via coefficient-based factor analysis.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python dataset/generate_dataset.py
```

### 3. Train Models
```bash
python train.py
```
> Trains all 4 algorithms, prints a comparison report, saves the best
> model + all evaluation plots (~30 seconds on CPU).

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
> Opens at **http://localhost:8501**

---

## 🎮 How to Use the App

1. Open the **🩺 Patient Assessment** tab
2. Enter the patient's clinical data across three sections:
   - *Patient Demographics* — age, sex, BP, cholesterol
   - *Cardiac Indicators* — chest pain type, max HR, exercise angina, ECG, ST readings
   - *Lab & Imaging Results* — blood sugar, vessels blocked, thalassemia
3. Click **🩺 Run Cardiac Assessment**
4. Review:
   - Risk tier banner (High / Moderate / Low-Moderate / Low)
   - Probability gauge (0–100%)
   - Red/green factor pills for personalised explanation

---

## 🔬 Key Medical Concepts Used

| Feature | Why it matters |
|---|---|
| Chest Pain Type | Asymptomatic CP is paradoxically highest-risk |
| Max Heart Rate | Lower max HR during stress often indicates disease |
| ST Depression (Oldpeak) | Key ECG marker for ischaemia |
| Major Vessels (CA) | Direct imaging of arterial blockage |
| Thalassemia | Blood disorder correlating with cardiac stress |

---

## 🔮 Future Improvements

- Integrate SHAP for model-agnostic explanations
- Add PDF report generation (patient summary download)
- Support batch CSV upload for clinical screening
- Deploy on Streamlit Community Cloud
- Tune with real UCI Cleveland dataset + cross-validation

---

## 👨‍💻 Author

**Vinith Prakash B**
Machine Learning Intern — CodeAlpha
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 🏢 About CodeAlpha

CodeAlpha is a leading software development company driving innovation
through AI and intelligent systems.
🌐 [www.codealpha.tech](https://www.codealpha.tech)

---

## ⚠️ Medical Disclaimer

This project is built **for educational purposes only** as part of a machine
learning internship. It is **not** a medical device and **must not** be used
for actual clinical decision-making. Always consult a qualified cardiologist.

---

## 📄 License

MIT License
