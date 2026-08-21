"""
Preprocessing and Temporal Feature Engineering Pipeline for Longitudinal EHR Data.

Extracts static demographics, temporal symptom trajectories, lab dynamics,
care interaction velocity, and simplified EULAR/ACR risk indices across specified
observation windows (T_obs).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

class LongitudinalFeatureExtractor:
    def __init__(self, observation_window_months: int = 12, time_decay_lambda: float = 0.1):
        """
        Args:
            observation_window_months: Cutoff month T_obs post initial symptom presentation.
            time_decay_lambda: Decay coefficient for exponential time-decayed symptom weighting.
        """
        self.t_obs = observation_window_months
        self.lambda_decay = time_decay_lambda
        self.symptom_cols = [
            'arthralgia', 'chronic_fatigue', 'dry_eyes', 'dry_mouth',
            'malar_rash', 'raynaud_phenomenon', 'photosensitivity',
            'oral_ulcers', 'unexplained_fever'
        ]
        
    def transform(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes patients and longitudinal encounter logs into a feature matrix X and target y.
        """
        features_list = []
        
        for _, pat in patients_df.iterrows():
            p_id = pat['patient_id']
            t_first_sym = pat['first_symptom_month']
            t_cutoff = t_first_sym + self.t_obs
            
            # Filter encounters up to t_cutoff
            p_encs = encounters_df[
                (encounters_df['patient_id'] == p_id) & 
                (encounters_df['month'] <= t_cutoff)
            ].sort_values('month')
            
            # Baseline Demographics
            feat = {
                'patient_id': p_id,
                'age_onset': pat['age_onset'],
                'sex_female': 1 if pat['sex'] == 'Female' else 0,
                'race_caucasian': 1 if pat['race'] == 'Caucasian' else 0,
                'race_african_american': 1 if pat['race'] == 'African American' else 0,
                'race_hispanic': 1 if pat['race'] == 'Hispanic' else 0,
                'race_asian': 1 if pat['race'] == 'Asian' else 0,
                'observation_window_months': self.t_obs,
                'total_encounters_in_window': len(p_encs)
            }
            
            if len(p_encs) == 0:
                # Fallback for empty encounters
                for sym in self.symptom_cols:
                    feat[f'sym_{sym}_count'] = 0
                    feat[f'sym_{sym}_ever'] = 0
                feat['sym_total_cumulative'] = 0
                feat['sym_time_decayed_score'] = 0.0
                feat['sym_acquisition_velocity'] = 0.0
                
                feat['ana_latest_titer'] = 0
                feat['ana_positive_160'] = 0
                feat['anti_dsdna_max'] = 0.0
                feat['anti_ssa_max'] = 0.0
                feat['anti_ssb_max'] = 0.0
                feat['c3_min'] = 110.0
                feat['c4_min'] = 25.0
                feat['c3_low'] = 0
                feat['c4_low'] = 0
                feat['wbc_min'] = 6.5
                feat['leukopenia'] = 0
                feat['esr_max'] = 15.0
                feat['crp_max'] = 2.0
                feat['lab_panels_ordered'] = 0
                
                feat['pcp_visits'] = 0
                feat['specialist_visits'] = 0
                feat['seen_rheumatologist'] = 0
                
                feat['sle_clinical_score'] = 0.0
                feat['sjd_clinical_score'] = 0.0
            else:
                # 1. Symptom Features
                symptom_sums = p_encs[self.symptom_cols].sum()
                for sym in self.symptom_cols:
                    feat[f'sym_{sym}_count'] = int(symptom_sums[sym])
                    feat[f'sym_{sym}_ever'] = 1 if symptom_sums[sym] > 0 else 0
                
                feat['sym_total_cumulative'] = int(p_encs[self.symptom_cols].values.sum())
                
                # Exponential time-decayed symptom score
                months_ago = t_cutoff - p_encs['month'].values
                decay_weights = np.exp(-self.lambda_decay * months_ago)
                encounter_symptom_counts = p_encs[self.symptom_cols].sum(axis=1).values
                feat['sym_time_decayed_score'] = float(np.sum(decay_weights * encounter_symptom_counts))
                
                # Symptom acquisition velocity (symptoms added per month)
                span_months = max(1.0, float(p_encs['month'].max() - p_encs['month'].min()))
                feat['sym_acquisition_velocity'] = float(feat['sym_total_cumulative'] / span_months)
                
                # 2. Serology Labs & Biomarkers
                ana_vals = p_encs['ana_titer'].dropna()
                feat['ana_latest_titer'] = float(ana_vals.iloc[-1]) if len(ana_vals) > 0 else 0.0
                feat['ana_max_titer'] = float(ana_vals.max()) if len(ana_vals) > 0 else 0.0
                feat['ana_positive_160'] = 1 if feat['ana_max_titer'] >= 160 else 0
                
                dsdna_vals = p_encs['anti_dsdna'].dropna()
                feat['anti_dsdna_max'] = float(dsdna_vals.max()) if len(dsdna_vals) > 0 else 0.0
                feat['anti_dsdna_positive'] = 1 if feat['anti_dsdna_max'] >= 30.0 else 0
                
                ssa_vals = p_encs['anti_ssa_ro'].dropna()
                feat['anti_ssa_max'] = float(ssa_vals.max()) if len(ssa_vals) > 0 else 0.0
                feat['anti_ssa_positive'] = 1 if feat['anti_ssa_max'] >= 20.0 else 0
                
                ssb_vals = p_encs['anti_ssb_la'].dropna()
                feat['anti_ssb_max'] = float(ssb_vals.max()) if len(ssb_vals) > 0 else 0.0
                feat['anti_ssb_positive'] = 1 if feat['anti_ssb_max'] >= 20.0 else 0
                
                c3_vals = p_encs['complement_c3'].dropna()
                feat['c3_min'] = float(c3_vals.min()) if len(c3_vals) > 0 else 110.0
                feat['c3_low'] = 1 if feat['c3_min'] < 80.0 else 0
                
                c4_vals = p_encs['complement_c4'].dropna()
                feat['c4_min'] = float(c4_vals.min()) if len(c4_vals) > 0 else 25.0
                feat['c4_low'] = 1 if feat['c4_min'] < 12.0 else 0
                
                wbc_vals = p_encs['wbc_count'].dropna()
                feat['wbc_min'] = float(wbc_vals.min()) if len(wbc_vals) > 0 else 6.5
                feat['leukopenia'] = 1 if feat['wbc_min'] < 4.0 else 0
                
                esr_vals = p_encs['esr'].dropna()
                feat['esr_max'] = float(esr_vals.max()) if len(esr_vals) > 0 else 15.0
                
                crp_vals = p_encs['crp'].dropna()
                feat['crp_max'] = float(crp_vals.max()) if len(crp_vals) > 0 else 2.0
                
                feat['lab_panels_ordered'] = int(p_encs['ana_titer'].notna().sum())
                
                # 3. Provider Dynamics
                feat['pcp_visits'] = int((p_encs['provider_type'] == 'PCP').sum())
                feat['specialist_visits'] = int((p_encs['provider_type'].isin(['Dermatology', 'Ophthalmology', 'Rheumatology'])).sum())
                feat['seen_rheumatologist'] = 1 if (p_encs['provider_type'] == 'Rheumatology').any() else 0
                
                # 4. Clinical Score Heuristics (EULAR/ACR simplified score)
                # SLE Score: ANA (2) + Malar Rash (4) + Low C3/C4 (3) + Anti-dsDNA (6) + Leukopenia (2) + Arthralgia (2)
                sle_score = 0.0
                if feat['ana_positive_160']: sle_score += 2.0
                if feat['sym_malar_rash_ever']: sle_score += 4.0
                if feat['c3_low'] or feat['c4_low']: sle_score += 3.0
                if feat['anti_dsdna_positive']: sle_score += 6.0
                if feat['leukopenia']: sle_score += 2.0
                if feat['sym_arthralgia_ever']: sle_score += 2.0
                feat['sle_clinical_score'] = sle_score
                
                # SjD Score: Dry eyes (3) + Dry mouth (3) + Anti-SSA (6) + Anti-SSB (4) + ANA (2)
                sjd_score = 0.0
                if feat['sym_dry_eyes_ever']: sjd_score += 3.0
                if feat['sym_dry_mouth_ever']: sjd_score += 3.0
                if feat['anti_ssa_positive']: sjd_score += 6.0
                if feat['anti_ssb_positive']: sjd_score += 4.0
                if feat['ana_positive_160']: sjd_score += 2.0
                feat['sjd_clinical_score'] = sjd_score
                
            # Ground-truth targets
            feat['target_prolonged_delay'] = int(pat['prolonged_delay'])
            feat['target_delay_months'] = float(pat['diagnostic_delay_months']) if not np.isnan(pat['diagnostic_delay_months']) else 0.0
            feat['target_disease'] = pat['disease']
            feat['target_autoimmune'] = 1 if pat['disease'] in ['SLE', 'SjD'] else 0
            
            features_list.append(feat)
            
        df_out = pd.DataFrame(features_list)
        return df_out.fillna(0.0)

def prepare_dataset(data_dir: str, observation_window_months: int = 12) -> Tuple[pd.DataFrame, List[str]]:
    pat_path = os.path.join(data_dir, "patients.csv")
    enc_path = os.path.join(data_dir, "encounters.csv")
    
    patients_df = pd.read_csv(pat_path)
    encounters_df = pd.read_csv(enc_path)
    
    extractor = LongitudinalFeatureExtractor(observation_window_months=observation_window_months)
    dataset = extractor.transform(patients_df, encounters_df)
    
    # Extract list of predictor feature names
    non_feature_cols = [
        'patient_id', 'target_prolonged_delay', 'target_delay_months',
        'target_disease', 'target_autoimmune'
    ]
    feature_cols = [col for col in dataset.columns if col not in non_feature_cols]
    
    return dataset, feature_cols

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    dataset, features = prepare_dataset(data_dir, observation_window_months=12)
    print(f"Dataset extracted: {dataset.shape[0]} rows, {len(features)} predictor features.")
    print("Sample features:", features[:10])
