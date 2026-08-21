"""
Flask REST API and Web Application for Explainable Autoimmune Diagnostic Delay Framework.

Serves a modern Tailwind CSS + Chart.js web dashboard with real-time risk predictions,
SHAP model interpretability, cohort analytics, and counterfactual clinical simulations.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Any

app = Flask(__name__, template_folder='templates')

# Global cached pipeline
DATA_CACHE = None

def get_pipeline():
    global DATA_CACHE
    if DATA_CACHE is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(script_dir, "simulated_data", "trained_pipeline.pkl")
        
        if not os.path.exists(pkl_path):
            from train_and_save_pipeline import build_and_save_pipeline
            build_and_save_pipeline(os.path.join(script_dir, "simulated_data"))
            
        with open(pkl_path, "rb") as f:
            DATA_CACHE = pickle.load(f)
            
    return DATA_CACHE

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cohort_overview', methods=['GET'])
def get_cohort_overview():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pat_path = os.path.join(script_dir, "simulated_data", "patients.csv")
    enc_path = os.path.join(script_dir, "simulated_data", "encounters.csv")
    
    patients_df = pd.read_csv(pat_path)
    encounters_df = pd.read_csv(enc_path)
    
    auto_df = patients_df[patients_df['disease'].isin(['SLE', 'SjD'])].dropna(subset=['diagnostic_delay_months'])
    
    sle_delays = list(auto_df[auto_df['disease'] == 'SLE']['diagnostic_delay_months'])
    sjd_delays = list(auto_df[auto_df['disease'] == 'SjD']['diagnostic_delay_months'])
    
    # Symptom prevalence
    sym_cols = ['arthralgia', 'chronic_fatigue', 'dry_eyes', 'dry_mouth', 'malar_rash', 'raynaud_phenomenon', 'photosensitivity']
    sym_prev = (encounters_df.groupby('patient_id')[sym_cols].max().mean() * 100).to_dict()
    
    return jsonify({
        'total_patients': int(len(patients_df)),
        'sle_count': int((patients_df['disease'] == 'SLE').sum()),
        'sjd_count': int((patients_df['disease'] == 'SjD').sum()),
        'control_count': int((patients_df['disease'] == 'Control').sum()),
        'avg_delay_months': float(patients_df['diagnostic_delay_months'].dropna().mean()),
        'prolonged_delay_rate_pct': float((patients_df['prolonged_delay'] == 1).mean() * 100),
        'sle_delays': sle_delays,
        'sjd_delays': sjd_delays,
        'symptom_prevalence': sym_prev
    })

@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    cache = get_pipeline()
    predictor = cache['predictor']
    feature_cols = cache['features']
    
    req_data = request.json or {}
    
    feat_dict = {f: 0 for f in feature_cols}
    feat_dict['age_onset'] = req_data.get('age', 34)
    feat_dict['sex_female'] = 1 if req_data.get('sex', 'Female') == 'Female' else 0
    feat_dict['race_caucasian'] = 1
    feat_dict['observation_window_months'] = 12
    feat_dict['total_encounters_in_window'] = req_data.get('encounters', 6)
    
    arthralgia = req_data.get('arthralgia', True)
    fatigue = req_data.get('fatigue', True)
    dry_eyes = req_data.get('dry_eyes', False)
    dry_mouth = req_data.get('dry_mouth', False)
    malar_rash = req_data.get('malar_rash', True)
    raynaud = req_data.get('raynaud', False)
    
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
    
    ana_titer = req_data.get('ana_titer', 320)
    anti_dsdna = req_data.get('anti_dsdna', 45.0)
    anti_ssa = req_data.get('anti_ssa', 15.0)
    low_c3 = req_data.get('low_c3', True)
    low_c4 = req_data.get('low_c4', False)
    seen_rheum = req_data.get('seen_rheumatologist', False)
    
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
    
    x_input = np.array([feat_dict[f] for f in feature_cols]).reshape(1, -1)
    
    model_name = req_data.get('model_name', 'XGBoost')
    clf = predictor.fitted_classifiers.get(model_name, predictor.fitted_classifiers['XGBoost'])
    reg = predictor.fitted_regressors.get(f"{model_name}Regressor", predictor.fitted_regressors['XGBoostRegressor'])
    
    risk_prob = float(clf.predict_proba(x_input)[0, 1])
    est_delay = float(reg.predict(x_input)[0])
    
    risk_status = "HIGH RISK" if risk_prob > 0.5 else "MODERATE RISK" if risk_prob > 0.3 else "LOW RISK"
    
    return jsonify({
        'model_used': model_name,
        'risk_probability': round(risk_prob * 100, 1),
        'risk_status': risk_status,
        'estimated_delay_months': round(est_delay, 1),
        'sle_clinical_score': sle_score
    })

@app.route('/api/shap_explain', methods=['GET'])
def get_shap_explain():
    from explainability import DelayExplainabilityEngine
    
    cache = get_pipeline()
    predictor = cache['predictor']
    features = cache['features']
    X_bg = cache['X']
    y_bg = cache['y']
    
    model_name = request.args.get('model_name', 'XGBoost')
    clf = predictor.fitted_classifiers.get(model_name, predictor.fitted_classifiers['XGBoost'])
    explain_engine = DelayExplainabilityEngine(clf, features)
    
    patient_idx = int(request.args.get('patient_index', 10))
    patient_idx = min(len(X_bg) - 1, max(0, patient_idx))
    
    # Global feature importance
    glob_df = explain_engine.compute_global_importance(X_bg[:300], y_bg[:300]).head(10)
    global_importance = glob_df.to_dict(orient='records')
    
    # Local patient SHAP waterfall
    local_exp = explain_engine.compute_local_shap_waterfall(X_bg[patient_idx], X_bg[:200])
    wf_df = local_exp['waterfall'].head(10)
    local_waterfall = wf_df.to_dict(orient='records')
    
    return jsonify({
        'model_used': model_name,
        'patient_index': patient_idx,
        'base_value': round(local_exp['base_value'] * 100, 1),
        'patient_risk': round(local_exp['patient_risk'] * 100, 1),
        'global_importance': global_importance,
        'local_waterfall': local_waterfall
    })

@app.route('/api/counterfactual', methods=['POST'])
def run_counterfactual():
    cache = get_pipeline()
    predictor = cache['predictor']
    feature_cols = cache['features']
    X_bg = cache['X']
    
    req_data = request.json or {}
    p_idx = int(req_data.get('patient_index', 5))
    p_idx = min(len(X_bg) - 1, max(0, p_idx))
    
    x_orig = X_bg[p_idx].copy()
    clf = predictor.fitted_classifiers['RandomForest']
    reg = predictor.fitted_regressors['RandomForestRegressor']
    
    orig_risk = float(clf.predict_proba(x_orig.reshape(1, -1))[0, 1])
    orig_delay = float(reg.predict(x_orig.reshape(1, -1))[0])
    
    act_serology = req_data.get('order_serology', True)
    act_rheum = req_data.get('rheumatologist_referral', True)
    
    x_cf = x_orig.copy()
    ana_idx = feature_cols.index('ana_positive_160')
    rheum_idx = feature_cols.index('seen_rheumatologist')
    
    if act_serology:
        x_cf[ana_idx] = 1.0
    if act_rheum:
        x_cf[rheum_idx] = 1.0
        
    cf_risk = float(clf.predict_proba(x_cf.reshape(1, -1))[0, 1])
    cf_delay = float(reg.predict(x_cf.reshape(1, -1))[0])
    
    risk_reduction = (orig_risk - cf_risk) * 100
    months_saved = orig_delay - cf_delay
    
    return jsonify({
        'patient_index': p_idx,
        'orig_risk': round(orig_risk * 100, 1),
        'orig_delay': round(orig_delay, 1),
        'cf_risk': round(cf_risk * 100, 1),
        'cf_delay': round(cf_delay, 1),
        'risk_reduction_pct': round(risk_reduction, 1),
        'months_saved': round(months_saved, 1),
        'recommendation': "CLINICALLY RECOMMENDED" if risk_reduction > 10 else "MINIMAL IMPACT"
    })

if __name__ == '__main__':
    get_pipeline() # Pre-cache pipeline
    app.run(host='0.0.0.0', port=5050, debug=False)
