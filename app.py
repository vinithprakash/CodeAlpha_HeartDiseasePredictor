"""
CodeAlpha Internship — Task 4: Disease Prediction from Medical Data
app.py  ·  CardioScan AI — Heart Disease Risk Analyzer

Design language: Deep-navy surgical theatre, red-teal clinical accent,
ECG-green heartbeat animation in the header, monospaced vitals readout,
glassmorphic stat cards. No generic dashboard template.
"""

import json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────
BG       = "#0d1b2a"
CARD_BG  = "#112233"
BORDER   = "#1d3a4a"
RED      = "#e63946"
TEAL     = "#2ec4b6"
BLUE     = "#8ecae6"
TEXT     = "#dde6f0"
MUTED    = "#7a9ab0"

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── ECG animated header ── */
.ecg-header {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
    padding: 1.4rem 2rem 1rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}}
.ecg-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: {RED};
    letter-spacing: -0.5px;
    line-height: 1;
}}
.ecg-subtitle {{
    font-size: 0.9rem;
    color: {MUTED};
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.ecg-line {{
    margin-top: 0.9rem;
    width: 100%;
    height: 36px;
    overflow: hidden;
    position: relative;
}}
.ecg-svg {{ width: 100%; height: 36px; }}

/* ── Vital cards ── */
.vital-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.2rem;
}}
.vital-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}}
.vital-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: {TEAL};
    line-height: 1;
}}
.vital-label {{
    font-size: 0.75rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 0.35rem;
}}

/* ── Section labels ── */
.sec-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {TEAL};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 0.25rem 0.7rem;
    border-left: 3px solid {RED};
    margin-bottom: 0.8rem;
    margin-top: 0.4rem;
}}

/* ── Risk result banner ── */
.risk-banner {{
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    border: 2px solid;
    display: flex;
    align-items: center;
    gap: 1rem;
}}
.risk-icon {{ font-size: 2.2rem; }}
.risk-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    line-height: 1;
}}
.risk-desc {{ font-size: 0.88rem; color: {MUTED}; margin-top: 0.3rem; }}

/* ── Prob bars ── */
.prob-row {{ margin-bottom: 0.5rem; }}
.prob-label {{ font-size: 0.82rem; color: {MUTED}; margin-bottom: 0.15rem; }}
.prob-bar-bg {{
    background: {BORDER};
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}}
.prob-bar-fill {{
    height: 10px;
    border-radius: 999px;
    transition: width 0.6s ease;
}}

/* ── Factor pills ── */
.factor-row {{ display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; }}
.factor-pill {{
    display:inline-block;
    padding:0.22rem 0.65rem;
    border-radius:999px;
    font-size:0.8rem;
    font-family:'IBM Plex Mono', monospace;
}}

/* ── Disclaimer ── */
.disclaimer {{
    background: rgba(230,57,70,0.08);
    border: 1px solid rgba(230,57,70,0.25);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: {MUTED};
    margin-top: 1rem;
}}
.footer {{ text-align:center; color:{MUTED}; font-size:0.8rem; margin-top:2rem; }}
</style>
""", unsafe_allow_html=True)


# ── Assets ─────────────────────────────────────────────────────────────────
ECG_PATH = """M0,18 L20,18 L25,18 L30,5 L35,30 L40,18 L50,18
              L55,18 L60,5 L65,30 L70,18 L80,18
              L85,18 L90,5 L95,30 L100,18 L110,18
              L115,18 L120,5 L125,30 L130,18 L140,18
              L145,18 L150,5 L155,30 L160,18 L170,18
              L175,18 L180,5 L185,30 L190,18 L200,18
              L205,18 L210,5 L215,30 L220,18 L230,18
              L235,18 L240,5 L245,30 L250,18 L260,18
              L265,18 L270,5 L275,30 L280,18 L290,18
              L295,18 L300,5 L305,30 L310,18 L320,18
              L325,18 L330,5 L335,30 L340,18 L350,18
              L355,18 L360,5 L365,30 L370,18 L380,18
              L385,18 L390,5 L395,30 L400,18 L410,18"""

FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar >120",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "ST Slope",
    "ca":       "Major Vessels (0-3)",
    "thal":     "Thalassemia Type",
}

FEATURE_NAMES = list(FEATURE_LABELS.keys())

CP_OPTIONS    = {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal Pain",3:"Asymptomatic"}
RESTECG_OPT   = {0:"Normal",1:"ST-T Wave Abnormality",2:"LV Hypertrophy"}
SLOPE_OPT     = {0:"Upsloping",1:"Flat",2:"Downsloping"}
THAL_OPT      = {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}


# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model/heart_model.pkl")
    with open("model/metrics.json") as f: metrics = json.load(f)
    with open("model/meta.json")    as f: meta    = json.load(f)
    return model, metrics, meta

try:
    model, metrics, meta = load_artifacts()
    LOADED = True
except FileNotFoundError:
    LOADED = False


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ecg-header">
  <div class="ecg-title">🫀 CardioScan AI</div>
  <div class="ecg-subtitle">Heart Disease Risk Prediction · CodeAlpha ML Internship — Task 4</div>
  <div class="ecg-line">
    <svg class="ecg-svg" viewBox="0 0 410 36" preserveAspectRatio="none">
      <polyline points="{ECG_PATH}"
        fill="none" stroke="{RED}" stroke-width="1.8" opacity="0.7"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

if not LOADED:
    st.error("⚠️ Model not found. Run `python train.py` first.")
    st.stop()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🫀 CardioScan AI")
    st.caption("AI-powered cardiac risk assessment")
    st.divider()
    st.markdown("**📋 About this tool**")
    st.markdown(
        "CardioScan AI uses a Machine Learning model trained on 1,000 patient "
        "records to assess the likelihood of heart disease from clinical indicators."
    )
    st.divider()
    st.markdown("**🧠 Model Performance**")
    best = meta["best_model"]
    m = metrics[best]
    cols = st.columns(2)
    cols[0].metric("Accuracy", f"{m['accuracy']*100:.1f}%")
    cols[1].metric("ROC-AUC",  f"{m['roc_auc']:.3f}")
    cols2 = st.columns(2)
    cols2[0].metric("F1-Score", f"{m['f1']:.3f}")
    cols2[1].metric("Recall",   f"{m['recall']:.3f}")
    st.caption(f"Best model: **{best}**")
    st.divider()
    st.markdown("**⚠️ Medical Disclaimer**")
    st.caption(
        "This tool is for educational purposes only. "
        "It is NOT a substitute for professional medical advice. "
        "Always consult a qualified cardiologist."
    )


# ── Vital stats top bar ────────────────────────────────────────────────────
df_data = pd.read_csv("dataset/heart_data.csv")
disease_rate = df_data["target"].mean() * 100
avg_age      = df_data["age"].mean()
avg_chol     = df_data["chol"].mean()
avg_hr       = df_data["thalach"].mean()

st.markdown(f"""
<div class="vital-grid">
  <div class="vital-card">
    <div class="vital-value">{disease_rate:.1f}%</div>
    <div class="vital-label">Dataset Disease Rate</div>
  </div>
  <div class="vital-card">
    <div class="vital-value">{avg_age:.0f} yrs</div>
    <div class="vital-label">Avg Patient Age</div>
  </div>
  <div class="vital-card">
    <div class="vital-value">{avg_chol:.0f}</div>
    <div class="vital-label">Avg Cholesterol (mg/dl)</div>
  </div>
  <div class="vital-card">
    <div class="vital-value">{avg_hr:.0f} bpm</div>
    <div class="vital-label">Avg Max Heart Rate</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩺  Patient Assessment", "📊  Model Analytics", "🗂️  Clinical Data"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown('<div class="sec-label">Patient Demographics</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age (years)", 20, 80, 52)
                sex = st.radio("Sex", ["Male","Female"], horizontal=True)
                sex_val = 1 if sex == "Male" else 0
            with c2:
                trestbps = st.slider("Resting BP (mmHg)", 90, 200, 130)
                chol     = st.slider("Cholesterol (mg/dl)", 120, 565, 245)

            st.markdown('<div class="sec-label">Cardiac Indicators</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                cp_label  = st.selectbox("Chest Pain Type", list(CP_OPTIONS.values()))
                cp_val    = [k for k,v in CP_OPTIONS.items() if v==cp_label][0]
                thalach   = st.slider("Max Heart Rate Achieved", 70, 205, 150)
                exang_str = st.radio("Exercise-Induced Angina", ["No","Yes"], horizontal=True)
                exang     = 1 if exang_str=="Yes" else 0
            with c4:
                ecg_label = st.selectbox("Resting ECG Result", list(RESTECG_OPT.values()))
                restecg   = [k for k,v in RESTECG_OPT.items() if v==ecg_label][0]
                oldpeak   = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, 0.1)
                slope_label = st.selectbox("ST Slope", list(SLOPE_OPT.values()))
                slope     = [k for k,v in SLOPE_OPT.items() if v==slope_label][0]

            st.markdown('<div class="sec-label">Lab & Imaging Results</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                fbs_str = st.radio("Fasting Blood Sugar >120 mg/dl", ["No","Yes"], horizontal=True)
                fbs     = 1 if fbs_str=="Yes" else 0
                ca      = st.select_slider("Major Vessels Coloured (0-3)", [0,1,2,3], 0)
            with c6:
                thal_label = st.selectbox("Thalassemia", list(THAL_OPT.values()), index=2)
                thal       = [k for k,v in THAL_OPT.items() if v==thal_label][0]

            st.markdown("")
            submitted = st.button("🩺 Run Cardiac Assessment", use_container_width=True, type="primary")

    with col_result:
        if submitted:
            input_df = pd.DataFrame([{
                "age":age,"sex":sex_val,"cp":cp_val,"trestbps":trestbps,
                "chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,
                "exang":exang,"oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal,
            }])

            proba     = model.predict_proba(input_df)[0]
            p_disease = proba[1]
            p_healthy = proba[0]

            # Risk tier
            if p_disease >= 0.75:
                tier, color, icon, desc = "HIGH RISK", RED, "🚨", "Strong indicators of heart disease detected. Immediate cardiology referral recommended."
            elif p_disease >= 0.50:
                tier, color, icon, desc = "MODERATE RISK", "#f4a261", "⚠️", "Elevated risk markers present. Detailed cardiac workup advised."
            elif p_disease >= 0.30:
                tier, color, icon, desc = "LOW-MODERATE RISK", BLUE, "💛", "Some risk factors noted. Lifestyle changes and monitoring recommended."
            else:
                tier, color, icon, desc = "LOW RISK", TEAL, "✅", "Indicators suggest low likelihood of heart disease. Maintain healthy habits."

            # Banner
            st.markdown(f"""
            <div class="risk-banner" style="background:{color}15; border-color:{color};">
              <div class="risk-icon">{icon}</div>
              <div>
                <div class="risk-title" style="color:{color};">{tier}</div>
                <div class="risk-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(p_disease * 100, 1),
                number={"suffix":"%","font":{"size":42,"color":TEXT}},
                delta={"reference":50,"increasing":{"color":RED},"decreasing":{"color":TEAL}},
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Disease Probability","font":{"size":16,"color":MUTED}},
                gauge={
                    "axis":{"range":[0,100],"tickwidth":1,"tickcolor":MUTED,"ticksuffix":"%"},
                    "bar":{"color":color,"thickness":0.25},
                    "bgcolor":"rgba(0,0,0,0)",
                    "borderwidth":0,
                    "steps":[
                        {"range":[0,30], "color":"rgba(46,196,182,0.18)"},
                        {"range":[30,50],"color":"rgba(142,202,230,0.15)"},
                        {"range":[50,75],"color":"rgba(244,162,97,0.18)"},
                        {"range":[75,100],"color":"rgba(230,57,70,0.22)"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},"thickness":0.85,"value":p_disease*100},
                },
            ))
            fig_gauge.update_layout(
                height=270, margin=dict(t=60,b=10,l=30,r=30),
                paper_bgcolor="rgba(0,0,0,0)", font={"color":TEXT},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability bars
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-label">Heart Disease: {p_disease*100:.1f}%</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p_disease*100:.1f}%; background:{RED};"></div>
              </div>
            </div>
            <div class="prob-row">
              <div class="prob-label">No Disease: {p_healthy*100:.1f}%</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p_healthy*100:.1f}%; background:{TEAL};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Key risk markers
            st.divider()
            st.markdown('<div class="sec-label">Key Risk Markers — This Patient</div>', unsafe_allow_html=True)

            risk_flags, protect_flags = [], []
            if age > 55:      risk_flags.append(f"Age {age} (>55)")
            if sex_val == 1:  risk_flags.append("Male sex")
            if cp_val == 3:   risk_flags.append("Asymptomatic chest pain")
            if exang == 1:    risk_flags.append("Exercise angina ✓")
            if oldpeak > 2:   risk_flags.append(f"High ST depression ({oldpeak})")
            if ca >= 2:       risk_flags.append(f"{ca} blocked vessels")
            if thal == 3:     risk_flags.append("Reversible thal defect")
            if chol > 240:    risk_flags.append(f"High cholesterol ({chol})")
            if thalach >= 150:protect_flags.append(f"Good max HR ({thalach} bpm)")
            if fbs == 0:      protect_flags.append("Normal blood sugar")
            if trestbps < 130:protect_flags.append(f"Normal BP ({trestbps} mmHg)")
            if cp_val != 3:   protect_flags.append("Symptomatic chest pain type")

            pills_risk = "".join([
                f'<span class="factor-pill" style="background:{RED}22;color:{RED};border:1px solid {RED}44;">{f}</span>'
                for f in risk_flags
            ]) or f'<span class="factor-pill" style="background:{TEAL}22;color:{TEAL};">None identified</span>'

            pills_prot = "".join([
                f'<span class="factor-pill" style="background:{TEAL}22;color:{TEAL};border:1px solid {TEAL}44;">{f}</span>'
                for f in protect_flags
            ]) or f'<span class="factor-pill" style="background:{MUTED}22;color:{MUTED};">None identified</span>'

            st.markdown(f"**🔴 Risk factors:**")
            st.markdown(f'<div class="factor-row">{pills_risk}</div>', unsafe_allow_html=True)
            st.markdown(f"**🟢 Protective factors:**")
            st.markdown(f'<div class="factor-row">{pills_prot}</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
              ⚠️ <strong>Medical Disclaimer:</strong> This prediction is generated by a machine
              learning model for educational purposes only. It is not a medical diagnosis.
              Please consult a licensed cardiologist for any health concerns.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:{CARD_BG}; border:1px solid {BORDER}; border-radius:14px;
                        padding:3rem 1.5rem; text-align:center; margin-top:1rem;">
              <div style="font-size:3.5rem;">🫀</div>
              <div style="font-size:1.1rem; color:{MUTED}; margin-top:1rem; line-height:1.6;">
                Complete the patient form on the left and click<br>
                <strong style="color:{RED};">Run Cardiac Assessment</strong><br>
                to receive an AI risk analysis, probability gauge,<br>
                and personalised risk/protective factor summary.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">Model Performance Comparison</div>', unsafe_allow_html=True)

    df_metrics = pd.DataFrame(metrics).T.rename(columns={
        "accuracy":"Accuracy","precision":"Precision","recall":"Recall",
        "f1":"F1-Score","roc_auc":"ROC-AUC",
    })
    st.dataframe(
        df_metrics.style.format("{:.3f}").highlight_max(axis=0, color="rgba(46,196,182,0.3)"),
        use_container_width=True,
    )
    st.info(f"✅ **{meta['best_model']}** is deployed — selected by best ROC-AUC on held-out test set.")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec-label">Model Comparison</div>', unsafe_allow_html=True)
        st.image("utils/model_comparison.png", use_container_width=True)
        st.markdown('<div class="sec-label">Confusion Matrix</div>', unsafe_allow_html=True)
        st.image("utils/confusion_matrix.png", use_container_width=True)
    with r2:
        st.markdown('<div class="sec-label">ROC Curves</div>', unsafe_allow_html=True)
        st.image("utils/roc_curve.png", use_container_width=True)
        st.markdown('<div class="sec-label">Feature Importance</div>', unsafe_allow_html=True)
        st.image("utils/feature_importance.png", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLINICAL DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">Dataset Overview</div>', unsafe_allow_html=True)
    st.dataframe(df_data.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df_data))
    c2.metric("Disease Positive", int(df_data["target"].sum()))
    c3.metric("Disease Negative", int((df_data["target"]==0).sum()))
    c4.metric("Features Used", "13")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="sec-label">Disease Rate by Age Group</div>', unsafe_allow_html=True)
        st.image("utils/age_risk.png", use_container_width=True)
    with r2:
        st.markdown('<div class="sec-label">Disease Rate by Chest Pain Type</div>', unsafe_allow_html=True)
        cp_map = {0:"Typical Angina",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}
        cp_risk = df_data.groupby("cp")["target"].mean().reset_index()
        cp_risk["cp_label"] = cp_risk["cp"].map(cp_map)
        fig = go.Figure(go.Bar(
            x=cp_risk["cp_label"], y=cp_risk["target"]*100,
            marker_color=[RED,BLUE,TEAL,"#f4a261"],
            text=[f"{v*100:.1f}%" for v in cp_risk["target"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=360, yaxis_title="Disease Rate (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color":TEXT}, yaxis=dict(gridcolor=BORDER),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-label">Clinical Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_data.describe().T.style.format("{:.2f}"), use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer">CardioScan AI · CodeAlpha ML Internship — Task 4 · '
    f'Built with Streamlit, Scikit-learn & Plotly</div>',
    unsafe_allow_html=True,
)
