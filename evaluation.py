"""
Comprehensive Benchmark & Diagnostic Delay Reduction Evaluation Engine.

Evaluates performance across multi-observation windows (6, 12, 24 months) and
simulates clinical diagnostic delay reduction impact (months saved).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from preprocessing_feature_engineering import prepare_dataset
from models import DiagnosticDelayPredictor

class EvaluationRunner:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def run_multi_window_benchmark(self, windows: List[int] = [6, 12, 24]) -> Dict[str, Any]:
        """
        Runs cross-validation model benchmarking across 6, 12, and 24-month observation windows.
        """
        benchmark_results = {}
        
        for t_obs in windows:
            print(f"\n=======================================================")
            print(f"--- Running Benchmark for Observation Window T_obs = {t_obs} Months ---")
            print(f"=======================================================")
            dataset, features = prepare_dataset(self.data_dir, observation_window_months=t_obs)
            autoimmune_df = dataset[dataset['target_autoimmune'] == 1].reset_index(drop=True)
            
            X = autoimmune_df[features].values
            y_cls = autoimmune_df['target_prolonged_delay'].values
            y_reg = autoimmune_df['target_delay_months'].values
            
            predictor = DiagnosticDelayPredictor(random_state=42)
            clf_metrics = predictor.train_evaluate_classifiers(X, y_cls, features)
            reg_metrics = predictor.train_evaluate_regressors(X, y_reg)
            
            # Clinical Impact Simulation: Diagnostic Delay Reduction
            # Best model: PyTorch_LSTM or RandomForest
            best_model_name = 'PyTorch_LSTM' if 'PyTorch_LSTM' in clf_metrics else 'RandomForest'
            best_roc = clf_metrics[best_model_name]['ROC_AUC']
            
            # Simulate high-risk threshold (prob >= 0.5)
            best_fitted = predictor.fitted_classifiers[best_model_name]
            if best_model_name == 'LogisticRegression':
                m_obj, scaler = best_fitted
                probs = m_obj.predict_proba(scaler.transform(X))[:, 1]
            else:
                probs = best_fitted.predict_proba(X)[:, 1]
                
            high_risk_mask = (probs >= 0.5) & (y_cls == 1)
            
            # For correctly identified prolonged delay patients, early referral at T_obs cuts remaining delay in half
            actual_delay = y_reg[high_risk_mask]
            new_estimated_delay = np.maximum(t_obs + 2, actual_delay * 0.5)
            months_saved_per_patient = float(np.mean(actual_delay - new_estimated_delay)) if len(actual_delay) > 0 else 0.0
            total_months_saved_cohort = float(np.sum(actual_delay - new_estimated_delay)) if len(actual_delay) > 0 else 0.0
            
            benchmark_results[f'T_obs_{t_obs}_months'] = {
                'observation_window_months': t_obs,
                'classification_metrics': clf_metrics,
                'regression_metrics': reg_metrics,
                'impact_simulation': {
                    'prolonged_delay_cases_detected': int(np.sum(high_risk_mask)),
                    'detection_rate_pct': float(np.sum(high_risk_mask) / np.sum(y_cls) * 100),
                    'avg_months_saved_per_flagged_patient': months_saved_per_patient,
                    'total_cohort_delay_months_saved': total_months_saved_cohort
                }
            }
            
        return benchmark_results

def run_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    
    runner = EvaluationRunner(data_dir)
    results = runner.run_multi_window_benchmark(windows=[6, 12, 24])
    
    out_json = os.path.join(script_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nFull benchmark results saved to {out_json}")
    
    # Format Summary Table
    print("\n" + "="*80)
    print(f"{'Obs Window':12s} | {'Model':22s} | {'ROC-AUC':10s} | {'PR-AUC':10s} | {'Sens':8s} | {'Spec':8s}")
    print("="*80)
    for win_key, win_data in results.items():
        w_name = f"{win_data['observation_window_months']} Months"
        for m_name, m_val in win_data['classification_metrics'].items():
            print(f"{w_name:12s} | {m_name:22s} | {m_val['ROC_AUC']:.4f}     | {m_val['PR_AUC']:.4f}     | {m_val['Sensitivity']:.4f}   | {m_val['Specificity']:.4f}")
        print("-"*80)
        
if __name__ == '__main__':
    run_evaluation()
