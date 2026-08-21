"""
Automated PyTest Test Suite for Explainable Longitudinal Machine Learning Framework.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_simulator import EHRDataSimulator, generate_and_save_data
from preprocessing_feature_engineering import LongitudinalFeatureExtractor, prepare_dataset
from models import DiagnosticDelayPredictor, PyTorchModelWrapper
from explainability import DelayExplainabilityEngine

def setup_data_dir(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("test_sim_data"))
    pat_path, enc_path = generate_and_save_data(tmp_dir, n_patients=60)
    return tmp_dir, pat_path, enc_path

def test_data_simulator(setup_data_dir):
    tmp_dir, pat_path, enc_path = setup_data_dir
    
    assert os.path.exists(pat_path)
    assert os.path.exists(enc_path)
    
    patients_df = pd.read_csv(pat_path)
    encounters_df = pd.read_csv(enc_path)
    
    assert len(patients_df) == 60
    assert len(encounters_df) > 100
    assert set(['patient_id', 'disease', 'age_onset', 'sex', 'prolonged_delay']).issubset(patients_df.columns)
    assert set(['patient_id', 'encounter_id', 'month', 'provider_type', 'ana_titer']).issubset(encounters_df.columns)

def test_preprocessing_feature_engineering(setup_data_dir):
    tmp_dir, _, _ = setup_data_dir
    dataset, features = prepare_dataset(tmp_dir, observation_window_months=12)
    
    assert len(dataset) == 60
    assert len(features) > 30
    assert not dataset[features].isnull().values.any()
    assert 'target_prolonged_delay' in dataset.columns
    assert 'target_autoimmune' in dataset.columns

def test_models_classification_and_regression(setup_data_dir):
    tmp_dir, _, _ = setup_data_dir
    dataset, features = prepare_dataset(tmp_dir, observation_window_months=12)
    
    X = dataset[features].values
    y_cls = dataset['target_prolonged_delay'].values
    y_reg = dataset['target_delay_months'].values
    
    predictor = DiagnosticDelayPredictor(random_state=42)
    clf_res = predictor.train_evaluate_classifiers(X, y_cls, features, n_splits=2)
    reg_res = predictor.train_evaluate_regressors(X, y_reg, n_splits=2)
    
    assert 'HistGradientBoosting' in clf_res
    assert 'PyTorch_LSTM' in clf_res
    assert clf_res['RandomForest']['ROC_AUC'] >= 0.5
    assert reg_res['RandomForestRegressor']['MAE_Months'] > 0

def test_explainability_engine(setup_data_dir):
    tmp_dir, _, _ = setup_data_dir
    dataset, features = prepare_dataset(tmp_dir, observation_window_months=12)
    
    X = dataset[features].values
    y = dataset['target_prolonged_delay'].values
    
    predictor = DiagnosticDelayPredictor(random_state=42)
    predictor.train_evaluate_classifiers(X, y, features, n_splits=2)
    
    rf_model = predictor.fitted_classifiers['RandomForest']
    engine = DelayExplainabilityEngine(rf_model, features)
    
    glob_imp = engine.compute_global_importance(X[:20], y[:20], n_repeats=2)
    assert len(glob_imp) == len(features)
    
    local_exp = engine.compute_local_shap_waterfall(X[0], X[:20])
    assert 'base_value' in local_exp
    assert 'patient_risk' in local_exp
    assert len(local_exp['waterfall']) == len(features)
    print("All unit tests passed successfully!")

if __name__ == '__main__':
    import tempfile
    import shutil
    tmp_dir = tempfile.mkdtemp()
    try:
        pat_path, enc_path = generate_and_save_data(tmp_dir, n_patients=60)
        setup_tuple = (tmp_dir, pat_path, enc_path)
        print("Running test_data_simulator...")
        test_data_simulator(setup_tuple)
        print("Running test_preprocessing_feature_engineering...")
        test_preprocessing_feature_engineering(setup_tuple)
        print("Running test_models_classification_and_regression...")
        test_models_classification_and_regression(setup_tuple)
        print("Running test_explainability_engine...")
        test_explainability_engine(setup_tuple)
        print("--- ALL 4 UNIT TESTS PASSED CLEANLY ---")
    finally:
        shutil.rmtree(tmp_dir)

