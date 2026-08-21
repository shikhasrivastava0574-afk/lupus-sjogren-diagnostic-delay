"""
Streamlit Interactive Clinical Decision Support System & XAI Dashboard.

Provides Cohort Analytics, Real-Time Patient Diagnostic Delay Risk Calculator,
SHAP Explainability Waterfall Plots, and Counterfactual "What-If" Clinical Recommendations.
Uses native Matplotlib/Seaborn and Streamlit components for 100% offline & Safari compatibility.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Page Config
st.set_page_config(
    page_title="Explainable Longitudinal ML for Autoimmune Diagnostic Delay",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Data Loaders
@st.cache_data
def load_simulated_cohort():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pat_path = os.path.join(script_dir, "simulated_data", "patients.csv")
    enc_path = os.path.join(script_dir, "simulated_data", "encounters.csv")
    if os.path.exists(pat_path) and os.path.exists(enc_path):
        patients_df = pd.read_csv(pat_path)
        encounters_df = pd.read_csv(enc_path)
        return patients_df, encounters_df
    return None, None

@st.cache_resource
def load_trained_pipeline():
    from explainability import DelayExplainabilityEngine
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, "simulated_data", "trained_pipeline.pkl")
    
    if not os.path.exists(pkl_path):
        from train_and_save_pipeline import build_and_save_pipeline
        build_and_save_pipeline(os.path.join(script_dir, "simulated_data"))
        
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    predictor = data['predictor']
    features = data['features']
    dataset = data['dataset']
    X = data['X']
    y = data['y']
    
    clf = predictor.fitted_classifiers['RandomForest']
    explain_engine = DelayExplainabilityEngine(clf, features)
    
    return predictor, explain_engine, features, dataset, X, y

# Main App Layout
def main():
    st.title("🩸 Explainable Longitudinal Machine Learning Framework")
    st.subheader("Predicting Prolonged Diagnostic Delay in Systemic Lupus Erythematosus & Sjögren's Disease")
    st.markdown("---")
    
    patients_df, encounters_df = load_simulated_cohort()
    predictor, explain_engine, feature_cols, dataset_df, X_bg, y_bg = load_trained_pipeline()
    
    # Sidebar Navigation
    st.sidebar.header("📌 Navigation")
    app_mode = st.sidebar.radio("Select View", [
        "1. Cohort Analytics & Delay Dynamics",
        "2. Patient Risk Predictor",
        "3. XAI Explainability Hub",
        "4. Counterfactual 'What-If' Engine"
    ])
    
    # =========================================================================
    # TAB 1: COHORT ANALYTICS & DELAY DYNAMICS
    # =========================================================================
    if app_mode == "1. Cohort Analytics & Delay Dynamics":
        st.header("📊 Cohort Diagnostic Delay Overview")
        st.markdown("""
        Systemic Lupus Erythematosus (SLE) and Sjögren's Disease (SjD) present with non-specific,
        insidious symptoms, leading to average diagnostic delays of **3 to 7+ years**.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        total_pats = len(patients_df)
        sle_count = len(patients_df[patients_df['disease'] == 'SLE'])
        sjd_count = len(patients_df[patients_df['disease'] == 'SjD'])
        avg_delay = patients_df['diagnostic_delay_months'].dropna().mean()
        
        col1.metric("Total Cohort Patients", f"{total_pats:,}")
        col2.metric("SLE Patients", f"{sle_count:,}")
        col3.metric("Sjögren's Patients", f"{sjd_count:,}")
        col4.metric("Avg Diagnostic Delay", f"{avg_delay:.1f} Months")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Diagnostic Delay Distribution (Months)")
            auto_patients = patients_df[patients_df['disease'].isin(['SLE', 'SjD'])].dropna(subset=['diagnostic_delay_months'])
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=auto_patients, x='diagnostic_delay_months', hue='disease', kde=True, ax=ax, palette={'SLE': '#e74c3c', 'SjD': '#3498db'}, bins=25)
            ax.axvline(24, color='black', linestyle='--', label='Prolonged Delay Cutoff (24m)')
            ax.set_xlabel("Diagnostic Delay (Months)")
            ax.set_ylabel("Patient Count")
            ax.set_title("Diagnostic Delay Distribution by Disease")
            ax.legend()
            st.pyplot(fig)
            plt.close()
            
        with c2:
            st.subheader("Early Non-Specific Symptom Prevalence")
            sym_cols = ['arthralgia', 'chronic_fatigue', 'dry_eyes', 'dry_mouth', 'malar_rash', 'raynaud_phenomenon', 'photosensitivity']
            sym_prev = (encounters_df.groupby('patient_id')[sym_cols].max().mean() * 100).sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sym_prev.plot(kind='barh', color='#c0392b', ax=ax)
            ax.set_xlabel("Prevalence (%)")
            ax.set_ylabel("Symptom")
            ax.set_title("Early Symptom Prevalence across Cohort")
            st.pyplot(fig)
            plt.close()

    # =========================================================================
    # TAB 2: PATIENT RISK PREDICTOR
    # =========================================================================
    elif app_mode == "2. Patient Risk Predictor":
        st.header("⚡ Live Patient Diagnostic Delay Risk Calculator")
        st.markdown("Enter patient clinical trajectory details at observation window (12 Months post initial presentation).")
        
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            st.subheader("Clinical Parameters")
            age = st.slider("Age at Symptom Onset", 18, 75, 34)
            sex = st.selectbox("Sex", ["Female", "Male"])
            encounters = st.number_input("Total Encounters in 12m", 1, 15, 6)
            
            st.markdown("**Early Non-Specific Symptoms**")
            arthralgia = st.checkbox("Arthralgia (Joint Pain)", value=True)
            fatigue = st.checkbox("Chronic Fatigue", value=True)
            dry_eyes = st.checkbox("Dry Eyes (Sicca)", value=False)
            dry_mouth = st.checkbox("Dry Mouth (Sicca)", value=False)
            malar_rash = st.checkbox("Malar Rash (Butterfly)", value=True)
            raynaud = st.checkbox("Raynaud's Phenomenon", value=False)
            
            st.markdown("**Serology & Lab Findings**")
            ana_titer = st.selectbox("ANA Titer Level", [0, 40, 80, 160, 320, 640, 1280], index=4)
            anti_dsdna = st.number_input("Anti-dsDNA (IU/mL)", 0.0, 300.0, 45.0)
            anti_ssa = st.number_input("Anti-SSA/Ro (Units/mL)", 0.0, 300.0, 15.0)
            low_c3 = st.checkbox("Low Complement C3 (<80 mg/dL)", value=True)
            low_c4 = st.checkbox("Low Complement C4 (<12 mg/dL)", value=False)
            seen_rheum = st.checkbox("Already Seen Rheumatologist?", value=False)
            
        with c_right:
            feat_dict = {f: 0 for f in feature_cols}
            feat_dict['age_onset'] = age
            feat_dict['sex_female'] = 1 if sex == "Female" else 0
            feat_dict['race_caucasian'] = 1
            feat_dict['observation_window_months'] = 12
            feat_dict['total_encounters_in_window'] = encounters
            
            feat_dict['sym_arthralgia_count'] = 3 if arthralgia else 0
            feat_dict['sym_arthralgia_ever'] = 1 if arthralgia else 0
            feat_dict['sym_chronic_fatigue_count'] = 3 if fatigue else 0
            feat_dict['sym_chronic_fatigue_ever'] = 1 if fatigue else 0
            feat_dict['sym_dry_eyes_ever'] = 1 if dry_eyes else 0
            feat_dict['sym_dry_mouth_ever'] = 1 if dry_mouth else 0
            feat_dict['sym_malar_rash_ever'] = 1 if malar_rash else 0
            feat_dict['sym_raynaud_phenomenon_ever'] = 1 if raynaud else 0
            
            feat_dict['sym_total_cumulative'] = sum([arthralgia, fatigue, dry_eyes, dry_mouth, malar_rash, raynaud]) * 2
            feat_dict['sym_time_decayed_score'] = feat_dict['sym_total_cumulative'] * 0.8
            feat_dict['sym_acquisition_velocity'] = feat_dict['sym_total_cumulative'] / 12.0
            
            feat_dict['ana_latest_titer'] = ana_titer
            feat_dict['ana_max_titer'] = ana_titer
            feat_dict['ana_positive_160'] = 1 if ana_titer >= 160 else 0
            feat_dict['anti_dsdna_max'] = anti_dsdna
            feat_dict['anti_dsdna_positive'] = 1 if anti_dsdna >= 30.0 else 0
            feat_dict['anti_ssa_max'] = anti_ssa
            feat_dict['anti_ssa_positive'] = 1 if anti_ssa >= 20.0 else 0
            
            feat_dict['c3_low'] = 1 if low_c3 else 0
            feat_dict['c4_low'] = 1 if low_c4 else 0
            feat_dict['seen_rheumatologist'] = 1 if seen_rheum else 0
            
            sle_score = (2.0 if ana_titer>=160 else 0) + (4.0 if malar_rash else 0) + (3.0 if low_c3 or low_c4 else 0) + (6.0 if anti_dsdna>=30 else 0)
            feat_dict['sle_clinical_score'] = sle_score
            
            x_input = np.array([feat_dict[f] for f in feature_cols])
            
            model = predictor.fitted_classifiers['RandomForest']
            risk_prob = float(model.predict_proba(x_input.reshape(1, -1))[0, 1])
            
            reg_model = predictor.fitted_regressors['RandomForestRegressor']
            est_delay_months = float(reg_model.predict(x_input.reshape(1, -1))[0])
            
            st.subheader("Predictive Risk Assessment")
            
            # Matplotlib Progress Gauge
            fig, ax = plt.subplots(figsize=(6, 1.8))
            color = "#d9534f" if risk_prob > 0.5 else "#f0ad4e" if risk_prob > 0.3 else "#5cb85c"
            ax.barh([0], [risk_prob * 100], color=color, height=0.5)
            ax.barh([0], [100 - risk_prob * 100], left=[risk_prob * 100], color='#e0e0e0', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Prolonged Diagnostic Delay Risk (%)")
            ax.set_title(f"Risk Score: {risk_prob*100:.1f}%")
            st.pyplot(fig)
            plt.close()
            
            m1, m2 = st.columns(2)
            m1.metric("Risk Status", "🔴 HIGH RISK (>24m Delay)" if risk_prob > 0.5 else "🟡 MODERATE RISK" if risk_prob > 0.3 else "🟢 LOW RISK")
            m2.metric("Estimated Total Delay", f"{est_delay_months:.1f} Months")

    # =========================================================================
    # TAB 3: XAI EXPLAINABILITY HUB
    # =========================================================================
    elif app_mode == "3. XAI Explainability Hub":
        st.header("🧠 SHAP Model Interpretability Hub")
        st.markdown("Understand global population drivers and local individual patient risk factors.")
        
        ex_tab1, ex_tab2 = st.tabs(["Global Population Drivers", "Local Patient Waterfall Explanation"])
        
        with ex_tab1:
            st.subheader("Global Population Feature Importance (SHAP)")
            glob_df = explain_engine.compute_global_importance(X_bg[:300], y_bg[:300]).head(12)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=glob_df, y='feature', x='importance_mean', palette='Blues_r', ax=ax)
            ax.set_xlabel("Mean |SHAP Value| (Feature Impact)")
            ax.set_ylabel("Clinical Feature")
            ax.set_title("Top 12 Features Driving Prolonged Diagnostic Delay Risk")
            st.pyplot(fig)
            plt.close()
            
        with ex_tab2:
            st.subheader("Local Patient Waterfall Risk Attribution")
            p_index = st.slider("Select Sample Patient Index", 0, len(X_bg)-1, 10)
            local_exp = explain_engine.compute_local_shap_waterfall(X_bg[p_index], X_bg[:200])
            
            st.info(f"**Base Population Risk**: {local_exp['base_value']*100:.1f}% | **Patient Risk**: {local_exp['patient_risk']*100:.1f}%")
            
            wf_df = local_exp['waterfall'].head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in wf_df['shap_value']]
            ax.barh(wf_df['feature'], wf_df['shap_value'], color=colors)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel("SHAP Impact (+ Increases Risk / - Decreases Risk)")
            ax.set_ylabel("Clinical Feature")
            ax.set_title(f"Local SHAP Feature Attributions for Patient #{p_index}")
            st.pyplot(fig)
            plt.close()

    # =========================================================================
    # TAB 4: COUNTERFACTUAL "WHAT-IF" ENGINE
    # =========================================================================
    elif app_mode == "4. Counterfactual 'What-If' Engine":
        st.header("🔮 Counterfactual Clinical Recommendations")
        st.markdown("Simulate clinical interventions (e.g. early serology panel, early Rheumatology referral) to quantify diagnostic delay reduction.")
        
        p_idx = st.slider("Select High-Risk Delayed Patient", 0, len(X_bg)-1, 5)
        
        x_orig = X_bg[p_idx].copy()
        model = predictor.fitted_classifiers['RandomForest']
        reg_model = predictor.fitted_regressors['RandomForestRegressor']
        
        orig_risk = float(model.predict_proba(x_orig.reshape(1, -1))[0, 1])
        orig_delay = float(reg_model.predict(x_orig.reshape(1, -1))[0])
        
        st.subheader(f"Current Patient Baseline (Patient #{p_idx})")
        c1, c2 = st.columns(2)
        c1.metric("Baseline Delay Risk", f"{orig_risk*100:.1f}%")
        c2.metric("Estimated Diagnostic Delay", f"{orig_delay:.1f} Months")
        
        st.markdown("---")
        st.subheader("⚡ Simulate Clinical Actions (What-If Scenarios)")
        
        c_act1, c_act2 = st.columns(2)
        act_serology = c_act1.checkbox("Order Early Autoimmune Serology Panel (ANA + Anti-SSA/dsDNA)", value=True)
        act_rheum = c_act2.checkbox("Immediate Rheumatology Referral", value=True)
        
        x_cf = x_orig.copy()
        ana_idx = feature_cols.index('ana_positive_160')
        rheum_idx = feature_cols.index('seen_rheumatologist')
        
        if act_serology:
            x_cf[ana_idx] = 1.0
        if act_rheum:
            x_cf[rheum_idx] = 1.0
            
        cf_risk = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
        cf_delay = float(reg_model.predict(x_cf.reshape(1, -1))[0])
        
        risk_reduction = (orig_risk - cf_risk) * 100
        months_saved = orig_delay - cf_delay
        
        st.subheader("🎯 Counterfactual Outcome Comparison")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("New Projected Risk", f"{cf_risk*100:.1f}%", delta=f"-{risk_reduction:.1f}% Risk")
        col_res2.metric("New Diagnostic Delay", f"{cf_delay:.1f} Months", delta=f"-{months_saved:.1f} Months Saved")
        col_res3.metric("Action Status", "✅ CLINICALLY RECOMMENDED" if risk_reduction > 10 else "ℹ️ MINIMAL IMPACT")

if __name__ == '__main__':
    main()
