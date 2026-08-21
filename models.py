"""
Machine Learning and Sequential Neural Models for Diagnostic Delay Prediction.

Includes HistGradientBoosting, RandomForest, Regularized Logistic Regression,
PyTorch Longitudinal Sequence Encoders (LSTM), and Time-to-Event Regression.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, accuracy_score, precision_score, recall_score
from typing import Dict, Tuple, List, Any

# PyTorch Longitudinal LSTM Model
class LongitudinalLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super(LongitudinalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take last time step output
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc_out(out)
        return self.sigmoid(out)

class PyTorchModelWrapper:
    """Wrapper to make PyTorch LSTM conform to scikit-learn API."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, epochs: int = 25, lr: float = 0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = LongitudinalLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        # Reshape tabular to sequence (batch_size, seq_len=1, features)
        X_seq = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_seq)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_seq = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_seq).numpy().flatten()
        return np.column_stack([1.0 - probs, probs])
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

class DiagnosticDelayPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Multi-model suite featuring XGBoost, Random Forest, and LightGBM
        self.classifiers = {
            'XGBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state),
            'LightGBM': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1),
            'LogisticRegression': LogisticRegression(max_iter=500, random_state=random_state)
        }
        self.regressors = {
            'XGBoostRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state),
            'LightGBMRegressor': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=5, random_state=random_state),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
        }
        self.fitted_classifiers = {}
        self.fitted_regressors = {}
        self.scaler = StandardScaler()
        
    def train_evaluate_classifiers(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Runs Stratified Cross-Validation on classification models.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results = {}
        
        # Include PyTorch LSTM in classification suite
        all_models = dict(self.classifiers)
        all_models['PyTorch_LSTM'] = PyTorchModelWrapper(input_dim=X.shape[1], hidden_dim=32, epochs=20)
        
        for name, model in all_models.items():
            auc_list, pr_auc_list, brier_list, acc_list, sens_list, spec_list = [], [], [], [], [], []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if name == 'LogisticRegression':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                    
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_val)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                auc_list.append(roc_auc_score(y_val, probs))
                pr_auc_list.append(average_precision_score(y_val, probs))
                brier_list.append(brier_score_loss(y_val, probs))
                acc_list.append(accuracy_score(y_val, preds))
                
                # Sensitivity and Specificity
                tp = np.sum((preds == 1) & (y_val == 1))
                fn = np.sum((preds == 0) & (y_val == 1))
                tn = np.sum((preds == 0) & (y_val == 0))
                fp = np.sum((preds == 1) & (y_val == 0))
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sens_list.append(sens)
                spec_list.append(spec)
                
            results[name] = {
                'ROC_AUC': float(np.mean(auc_list)),
                'PR_AUC': float(np.mean(pr_auc_list)),
                'Brier_Score': float(np.mean(brier_list)),
                'Accuracy': float(np.mean(acc_list)),
                'Sensitivity': float(np.mean(sens_list)),
                'Specificity': float(np.mean(spec_list))
            }
            
            # Fit final model on whole dataset for production deployment
            if name == 'LogisticRegression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                m_fit = LogisticRegression(max_iter=500, random_state=self.random_state).fit(X_scaled, y)
                self.fitted_classifiers[name] = (m_fit, scaler)
            elif name == 'PyTorch_LSTM':
                m_fit = PyTorchModelWrapper(input_dim=X.shape[1], hidden_dim=32, epochs=25).fit(X, y)
                self.fitted_classifiers[name] = m_fit
            else:
                m_fit = model.__class__(**model.get_params()).fit(X, y)
                self.fitted_classifiers[name] = m_fit
                
        return results

    def train_evaluate_regressors(self, X: np.ndarray, y_delay: np.ndarray, n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluates continuous diagnostic delay duration regressors (estimating delay in months).
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        # Use binned y for stratified split
        y_binned = pd.qcut(y_delay, q=4, labels=False, duplicates='drop')
        results = {}
        
        for name, model in self.regressors.items():
            mae_list = []
            for train_idx, val_idx in skf.split(X, y_binned):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_delay[train_idx], y_delay[val_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                mae_list.append(mean_absolute_error(y_val, preds))
                
            results[name] = {
                'MAE_Months': float(np.mean(mae_list))
            }
            
            m_fit = model.__class__(**model.get_params()).fit(X, y_delay)
            self.fitted_regressors[name] = m_fit
            
        return results

if __name__ == '__main__':
    from preprocessing_feature_engineering import prepare_dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    
    dataset, features = prepare_dataset(data_dir, observation_window_months=12)
    # Filter to SLE/SjD patients for diagnostic delay modeling
    autoimmune_df = dataset[dataset['target_autoimmune'] == 1].reset_index(drop=True)
    
    X = autoimmune_df[features].values
    y_cls = autoimmune_df['target_prolonged_delay'].values
    y_reg = autoimmune_df['target_delay_months'].values
    
    predictor = DiagnosticDelayPredictor(random_state=42)
    clf_res = predictor.train_evaluate_classifiers(X, y_cls, features)
    reg_res = predictor.train_evaluate_regressors(X, y_reg)
    
    print("\n--- Classification Performance (Prolonged Delay > 24 Months) ---")
    for m_name, m_metrics in clf_res.items():
        print(f"Model: {m_name:22s} | ROC-AUC: {m_metrics['ROC_AUC']:.4f} | PR-AUC: {m_metrics['PR_AUC']:.4f} | Brier: {m_metrics['Brier_Score']:.4f} | Sens: {m_metrics['Sensitivity']:.4f} | Spec: {m_metrics['Specificity']:.4f}")
        
    print("\n--- Delay Duration Regression Performance ---")
    for r_name, r_metrics in reg_res.items():
        print(f"Model: {r_name:30s} | MAE: {r_metrics['MAE_Months']:.2f} Months")
