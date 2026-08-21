"""
Pre-trains model pipeline and persists trained artifacts to disk.
Prevents background thread re-training crashes during Streamlit app loading.
"""

import os
import pickle
import numpy as np
import pandas as pd
from preprocessing_feature_engineering import prepare_dataset
from models import DiagnosticDelayPredictor
from explainability import DelayExplainabilityEngine

def build_and_save_pipeline(output_dir: str):
    print("Preparing dataset...")
    dataset, features = prepare_dataset(output_dir, observation_window_months=12)
    autoimmune_df = dataset[dataset['target_autoimmune'] == 1].reset_index(drop=True)
    
    X = autoimmune_df[features].values
    y_cls = autoimmune_df['target_prolonged_delay'].values
    y_reg = autoimmune_df['target_delay_months'].values
    
    print("Training ML Classifiers and Regressors...")
    predictor = DiagnosticDelayPredictor(random_state=42)
    clf_res = predictor.train_evaluate_classifiers(X, y_cls, features, n_splits=5)
    reg_res = predictor.train_evaluate_regressors(X, y_reg, n_splits=5)
    
    clf = predictor.fitted_classifiers['RandomForest']
    explain_engine = DelayExplainabilityEngine(clf, features)
    
    pipeline_data = {
        'predictor': predictor,
        'features': features,
        'dataset': dataset,
        'X': X,
        'y': y_cls,
        'clf_res': clf_res,
        'reg_res': reg_res
    }
    
    save_path = os.path.join(output_dir, "trained_pipeline.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(pipeline_data, f)
        
    print(f"Pipeline successfully trained and saved to {save_path}")
    return save_path

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    build_and_save_pipeline(data_dir)
