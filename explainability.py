"""
Model-Agnostic and Tree-Based SHAP Explainability Engine (XAI) for Diagnostic Delay.

Provides Global Population Feature Importance, Local Patient Waterfall Attributions,
and Temporal Trajectory Risk Progression Analysis.
"""

import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Any

class DelayExplainabilityEngine:
    def __init__(self, model: Any, feature_names: List[str], base_scaler: Any = None):
        """
        Args:
            model: Trained classifier (e.g. RandomForest, HistGradientBoosting, LogisticRegression).
            feature_names: List of predictor feature names.
            base_scaler: Optional StandardScaler for models like LogisticRegression.
        """
        self.model = model
        self.feature_names = feature_names
        self.scaler = base_scaler
        
    def _predict_prob(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise ValueError("Model has no prediction method.")

    def compute_global_importance(self, X_val: np.ndarray, y_val: np.ndarray, n_repeats: int = 10) -> pd.DataFrame:
        """
        Computes permutation feature importance for global population risk drivers.
        """
        if self.scaler is not None:
            X_eval = self.scaler.transform(X_val)
        else:
            X_eval = X_val
            
        res = permutation_importance(self.model, X_eval, y_val, n_repeats=n_repeats, random_state=42, scoring='roc_auc')
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': res.importances_mean,
            'importance_std': res.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
        
        return importance_df

    def compute_local_shap_waterfall(self, patient_features: np.ndarray, background_data: np.ndarray) -> Dict[str, Any]:
        """
        Computes local additive feature attributions (SHAP waterfall values) for a single patient.
        Returns base_value, predicted_risk, and feature contributions.
        """
        # 1. Base expected prediction over background population
        bg_preds = self._predict_prob(background_data)
        base_value = float(np.mean(bg_preds))
        
        # 2. Patient predicted risk
        x_patient = patient_features.reshape(1, -1)
        patient_pred = float(self._predict_prob(x_patient)[0])
        
        # 3. Marginal feature contributions via Kernel/Tree additive approximation
        # marginal_i = f(x_with_feature_i) - f(x_background_with_feature_i)
        n_features = len(self.feature_names)
        shap_values = np.zeros(n_features)
        
        # Compute marginal impact for each feature
        for i in range(n_features):
            # Create hybrid sample replacing feature i with background median vs patient value
            bg_mod = background_data.copy()
            bg_mod[:, i] = patient_features[i]
            mod_preds = self._predict_prob(bg_mod)
            shap_values[i] = np.mean(mod_preds) - base_value

        # Normalize so sum(shap_values) == (patient_pred - base_value)
        total_diff = patient_pred - base_value
        sum_shap = np.sum(shap_values)
        if abs(sum_shap) > 1e-6:
            shap_values = shap_values * (total_diff / sum_shap)
            
        # Format waterfall table
        waterfall_df = pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': patient_features,
            'shap_value': shap_values
        })
        
        # Sort by absolute impact
        waterfall_df['abs_shap'] = np.abs(waterfall_df['shap_value'])
        waterfall_df = waterfall_df.sort_values('abs_shap', ascending=False).drop(columns=['abs_shap']).reset_index(drop=True)
        
        return {
            'base_value': base_value,
            'patient_risk': patient_pred,
            'risk_delta': patient_pred - base_value,
            'waterfall': waterfall_df
        }

    def compute_temporal_trajectory_explanation(self, p_id: str, patients_df: pd.DataFrame, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tracks how patient diagnostic delay risk score evolves across 6, 12, 18, 24, 36 months.
        """
        from preprocessing_feature_engineering import LongitudinalFeatureExtractor
        
        pat_subset = patients_df[patients_df['patient_id'] == p_id]
        if len(pat_subset) == 0:
            raise ValueError(f"Patient ID {p_id} not found.")
            
        trajectory_records = []
        for t_win in [6, 12, 18, 24, 36]:
            extractor = LongitudinalFeatureExtractor(observation_window_months=t_win)
            feat_df = extractor.transform(pat_subset, encounters_df)
            X_t = feat_df[self.feature_names].values
            risk = float(self._predict_prob(X_t)[0])
            
            trajectory_records.append({
                'month_window': t_win,
                'encounters_logged': feat_df['total_encounters_in_window'].iloc[0],
                'cum_symptoms': feat_df['sym_total_cumulative'].iloc[0],
                'ana_titer': feat_df['ana_max_titer'].iloc[0],
                'seen_rheumatologist': feat_df['seen_rheumatologist'].iloc[0],
                'predicted_delay_risk': risk
            })
            
        return pd.DataFrame(trajectory_records)

if __name__ == '__main__':
    from preprocessing_feature_engineering import prepare_dataset
    from models import DiagnosticDelayPredictor
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    
    dataset, features = prepare_dataset(data_dir, observation_window_months=12)
    autoimmune_df = dataset[dataset['target_autoimmune'] == 1].reset_index(drop=True)
    
    X = autoimmune_df[features].values
    y = autoimmune_df['target_prolonged_delay'].values
    
    # Train predictor
    predictor = DiagnosticDelayPredictor()
    predictor.train_evaluate_classifiers(X, y, features)
    
    # Pick top model: RandomForest
    model, scaler = predictor.fitted_classifiers['RandomForest'], None
    engine = DelayExplainabilityEngine(model, features)
    
    print("\n--- Global Feature Importance (Top 10) ---")
    glob_imp = engine.compute_global_importance(X[:300], y[:300])
    print(glob_imp.head(10).to_string())
    
    print("\n--- Local Patient SHAP Waterfall Explanation (Patient 0) ---")
    local_exp = engine.compute_local_shap_waterfall(X[0], X[:200])
    print(f"Base Population Risk: {local_exp['base_value']:.4f}")
    print(f"Patient Predicted Risk: {local_exp['patient_risk']:.4f}")
    print("Top 8 Features driving patient risk:")
    print(local_exp['waterfall'].head(8).to_string())
