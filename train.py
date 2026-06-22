"""
CodeAlpha Internship - Task 4: Disease Prediction from Medical Data
train.py — Heart Disease Prediction Model Training

Trains & compares: Logistic Regression, Random Forest, SVM, XGBoost-style
Gradient Boosting. Saves best model + metadata + evaluation plots.
"""

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
DATA_PATH    = "dataset/heart_data.csv"
FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak","slope","ca","thal"
]
FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex (1=M, 0=F)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar >120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (0-3)",
    "thal":     "Thalassemia",
}

# ── Palette ────────────────────────────────────────────────────────────────
CLR_POS  = "#e63946"   # disease
CLR_NEG  = "#2ec4b6"   # no disease
CLR_GRID = "#1d3a4a"
BG       = "#0d1b2a"


def load():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_NAMES]
    y  = df["target"]
    return df, X, y


def evaluate(name, pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    m = dict(
        accuracy  = accuracy_score(y_test, y_pred),
        precision = precision_score(y_test, y_pred),
        recall    = recall_score(y_test, y_pred),
        f1        = f1_score(y_test, y_pred),
        roc_auc   = roc_auc_score(y_test, y_proba),
    )
    print(f"\n── {name} ──")
    for k, v in m.items(): print(f"  {k:10s}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Disease","Disease"]))
    return m, y_pred, y_proba


# ── Plots ──────────────────────────────────────────────────────────────────
def _dark_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig


def _dark_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#b0c4d8")
    ax.xaxis.label.set_color("#b0c4d8")
    ax.yaxis.label.set_color("#b0c4d8")
    ax.title.set_color("#e0eaf4")
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_GRID)
    ax.grid(color=CLR_GRID, linewidth=0.6, alpha=0.7)
    return ax


def plot_model_comparison(results):
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    models  = list(results.keys())
    x = np.arange(len(metrics)); w = 0.20
    colors = ["#e63946","#2ec4b6","#f4a261","#8ecae6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=model, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#e0eaf4")
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([m.replace("_"," ").title() for m in metrics], color="#b0c4d8")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score", color="#b0c4d8")
    ax.set_title("Model Comparison — Heart Disease Prediction", color="#e0eaf4", fontsize=14, pad=12)
    ax.tick_params(colors="#b0c4d8")
    ax.grid(axis="y", color=CLR_GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values(): s.set_edgecolor(CLR_GRID)
    ax.legend(facecolor="#112233", edgecolor=CLR_GRID, labelcolor="#e0eaf4")
    plt.tight_layout()
    plt.savefig("utils/model_comparison.png", dpi=150, facecolor=BG)
    plt.close(); print("Saved: utils/model_comparison.png")


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = ["#e63946","#2ec4b6","#f4a261","#8ecae6"]
    for (name, (fpr, tpr, auc)), col in zip(curves.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=col, linewidth=2)
    ax.plot([0,1],[0,1],"--", color="#445566", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#b0c4d8")
    ax.set_ylabel("True Positive Rate", color="#b0c4d8")
    ax.set_title("ROC Curves — All Models", color="#e0eaf4", fontsize=13)
    ax.tick_params(colors="#b0c4d8")
    ax.grid(color=CLR_GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values(): s.set_edgecolor(CLR_GRID)
    ax.legend(facecolor="#112233", edgecolor=CLR_GRID, labelcolor="#e0eaf4")
    plt.tight_layout()
    plt.savefig("utils/roc_curve.png", dpi=150, facecolor=BG)
    plt.close(); print("Saved: utils/roc_curve.png")


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.color_palette("light:#e63946", as_cmap=True),
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"],
                linewidths=0.5, linecolor=BG, ax=ax)
    ax.set_title("Confusion Matrix — Best Model", color="#e0eaf4", fontsize=13)
    ax.set_xlabel("Predicted", color="#b0c4d8")
    ax.set_ylabel("Actual", color="#b0c4d8")
    ax.tick_params(colors="#b0c4d8")
    plt.tight_layout()
    plt.savefig("utils/confusion_matrix.png", dpi=150, facecolor=BG)
    plt.close(); print("Saved: utils/confusion_matrix.png")


def plot_feature_importance(pipe):
    model = pipe.named_steps["classifier"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    order = np.argsort(importances)
    labels = [FEATURE_LABELS[FEATURE_NAMES[i]] for i in order]
    vals   = importances[order]
    colors = [CLR_POS if v > np.median(vals) else CLR_NEG for v in vals]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(range(len(vals)), vals, color=colors, alpha=0.88)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, color="#b0c4d8", fontsize=10)
    ax.set_xlabel("Feature Importance", color="#b0c4d8")
    ax.set_title("Feature Importance — Top Predictors", color="#e0eaf4", fontsize=13)
    ax.tick_params(colors="#b0c4d8")
    ax.grid(axis="x", color=CLR_GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values(): s.set_edgecolor(CLR_GRID)
    high_patch = mpatches.Patch(color=CLR_POS, label="High importance")
    low_patch  = mpatches.Patch(color=CLR_NEG, label="Lower importance")
    ax.legend(handles=[high_patch, low_patch],
              facecolor="#112233", edgecolor=CLR_GRID, labelcolor="#e0eaf4")
    plt.tight_layout()
    plt.savefig("utils/feature_importance.png", dpi=150, facecolor=BG)
    plt.close(); print("Saved: utils/feature_importance.png")


def plot_age_risk(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bins  = pd.cut(df["age"], bins=[25,35,45,55,65,80])
    grp   = df.groupby(bins, observed=True)["target"].mean() * 100

    ax.bar([str(b) for b in grp.index], grp.values,
           color=CLR_POS, alpha=0.85, edgecolor=BG, linewidth=0.5)
    ax.set_xlabel("Age Group", color="#b0c4d8")
    ax.set_ylabel("Disease Prevalence (%)", color="#b0c4d8")
    ax.set_title("Disease Rate by Age Group", color="#e0eaf4", fontsize=13)
    ax.tick_params(colors="#b0c4d8")
    ax.grid(axis="y", color=CLR_GRID, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values(): s.set_edgecolor(CLR_GRID)
    for i, v in enumerate(grp.values):
        ax.text(i, v+0.5, f"{v:.1f}%", ha="center", color="#e0eaf4", fontsize=9)
    plt.tight_layout()
    plt.savefig("utils/age_risk.png", dpi=150, facecolor=BG)
    plt.close(); print("Saved: utils/age_risk.png")


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("utils", exist_ok=True)

    print("="*55)
    print("  CodeAlpha — Heart Disease Prediction")
    print("="*55)

    df, X, y = load()
    print(f"Records: {len(df)} | Disease: {y.sum()} | No Disease: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=RANDOM_STATE),
        "SVM":                 SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
    }

    results, roc_data, pipelines = {}, {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        m, y_pred, y_proba = evaluate(name, pipe, X_test, y_test)
        results[name]   = m
        pipelines[name] = pipe
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, m["roc_auc"])

    # Best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipe = pipelines[best_name]
    print(f"\n✅ Best model: {best_name} (AUC={results[best_name]['roc_auc']:.4f})")

    # ── Plots
    plot_model_comparison(results)
    plot_roc(roc_data)
    best_pred = best_pipe.predict(X_test)
    plot_confusion(y_test, best_pred)
    plot_feature_importance(best_pipe)
    plot_age_risk(df)

    # ── Save
    joblib.dump(best_pipe, "model/heart_model.pkl")
    with open("model/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    meta = {
        "best_model":    best_name,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
            for col in FEATURE_NAMES
        }
    }
    with open("model/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    for n, m in results.items():
        mark = "✅" if n == best_name else "  "
        print(f"{mark} {n:22s} | Acc:{m['accuracy']:.3f} | F1:{m['f1']:.3f} | AUC:{m['roc_auc']:.3f}")
    print(f"\nModel saved → model/heart_model.pkl")


if __name__ == "__main__":
    main()
