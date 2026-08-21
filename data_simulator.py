"""
Longitudinal EHR Data Simulator for SLE and Sjögren's Diagnostic Delay Study.

Generates clinically realistic synthetic longitudinal Electronic Health Records (EHR) data
containing patient encounters, symptoms, serological biomarker trajectories, provider visits,
and diagnostic timelines for Systemic Lupus Erythematosus (SLE), Sjögren's Disease (SjD),
and Non-Autoimmune Control cohorts.
"""

import numpy as np
import pandas as pd
import os
import random
from typing import Tuple, Dict, List

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

class EHRDataSimulator:
    def __init__(self, n_patients: int = 1500, random_state: int = 42):
        self.n_patients = n_patients
        self.random_state = random_state
        set_seed(random_state)
        
    def generate_cohort(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates patient static demographics and longitudinal encounter timeline.
        Returns:
            patients_df: Static patient demographics and ground-truth diagnosis data.
            encounters_df: Longitudinal visit logs with temporal symptoms & lab markers.
        """
        n_sle = self.n_patients // 3
        n_sjd = self.n_patients // 3
        n_ctrl = self.n_patients - n_sle - n_sjd
        
        disease_labels = ['SLE'] * n_sle + ['SjD'] * n_sjd + ['Control'] * n_ctrl
        random.shuffle(disease_labels)
        
        patient_records = []
        encounter_records = []
        
        for p_idx in range(self.n_patients):
            patient_id = f"PAT_{p_idx+1:04d}"
            disease = disease_labels[p_idx]
            
            # Demographics (Autoimmune conditions skew female, ~85-90%)
            female_prob = 0.88 if disease in ['SLE', 'SjD'] else 0.70
            sex = 'Female' if np.random.rand() < female_prob else 'Male'
            
            if disease == 'SLE':
                age_onset = int(np.random.normal(loc=32, scale=8))
            elif disease == 'SjD':
                age_onset = int(np.random.normal(loc=48, scale=10))
            else:
                age_onset = int(np.random.normal(loc=45, scale=12))
            age_onset = max(18, min(75, age_onset))
            
            race = np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], 
                                    p=[0.55, 0.20, 0.15, 0.07, 0.03])
            
            # Ground truth diagnostic delay dynamics
            # SLE & SjD often experience long diagnostic delay if early serology isn't ordered
            first_symptom_month = int(np.random.uniform(0, 6))
            
            if disease == 'Control':
                diagnosis_month = None
                diagnostic_delay = np.nan
                prolonged_delay = 0
                total_encounters = np.random.randint(4, 10)
            else:
                # 50-60% of SLE/SjD patients experience prolonged delay (> 24 months)
                is_prolonged = np.random.rand() < (0.55 if disease == 'SLE' else 0.60)
                if is_prolonged:
                    diagnostic_delay = int(np.random.uniform(25, 54)) # >24 months
                    prolonged_delay = 1
                else:
                    diagnostic_delay = int(np.random.uniform(3, 24)) # <= 24 months
                    prolonged_delay = 0
                
                diagnosis_month = first_symptom_month + diagnostic_delay
                total_encounters = np.random.randint(6, 16)
            
            patient_records.append({
                'patient_id': patient_id,
                'disease': disease,
                'age_onset': age_onset,
                'sex': sex,
                'race': race,
                'first_symptom_month': first_symptom_month,
                'diagnosis_month': diagnosis_month,
                'diagnostic_delay_months': diagnostic_delay,
                'prolonged_delay': prolonged_delay
            })
            
            # Generate longitudinal encounters
            # Time horizon up to 60 months
            if disease == 'Control':
                total_encounters = min(total_encounters, 60)
                encounter_months = sorted(list(np.random.choice(range(0, 60), total_encounters, replace=False)))
            else:
                # Include encounters before and after symptom onset / diagnosis
                max_month = min(60, int(diagnosis_month) + 6) if (diagnosis_month is not None and not np.isnan(diagnosis_month)) else 60
                max_month = max(max_month, total_encounters)
                encounter_months = sorted(list(np.random.choice(range(0, max_month + 1), min(total_encounters, max_month + 1), replace=False)))
                if first_symptom_month not in encounter_months:
                    encounter_months[0] = first_symptom_month
                    encounter_months.sort()

            # Serology trajectories progression over time
            # Early stage: mild symptoms, low/negative titers
            # Delayed diagnosis stage: escalating symptoms, missing rheumatology referral
            has_seen_rheumatologist = False
            
            for e_idx, month in enumerate(encounter_months):
                is_post_onset = (month >= first_symptom_month)
                
                # Disease severity factor increases with time post symptom onset
                progression_factor = min(1.0, max(0.0, (month - first_symptom_month) / 36.0)) if is_post_onset else 0.0
                
                # Symptoms generation
                if is_post_onset:
                    if disease == 'SLE':
                        arthralgia = 1 if np.random.rand() < (0.6 + 0.3 * progression_factor) else 0
                        fatigue = 1 if np.random.rand() < (0.7 + 0.2 * progression_factor) else 0
                        dry_eyes = 1 if np.random.rand() < (0.2 + 0.2 * progression_factor) else 0
                        dry_mouth = 1 if np.random.rand() < (0.2 + 0.2 * progression_factor) else 0
                        malar_rash = 1 if np.random.rand() < (0.3 + 0.45 * progression_factor) else 0
                        raynaud = 1 if np.random.rand() < (0.25 + 0.35 * progression_factor) else 0
                        photosensitivity = 1 if np.random.rand() < (0.3 + 0.4 * progression_factor) else 0
                        oral_ulcers = 1 if np.random.rand() < (0.25 + 0.35 * progression_factor) else 0
                        unexplained_fever = 1 if np.random.rand() < (0.2 + 0.3 * progression_factor) else 0
                    elif disease == 'SjD':
                        arthralgia = 1 if np.random.rand() < (0.5 + 0.3 * progression_factor) else 0
                        fatigue = 1 if np.random.rand() < (0.75 + 0.2 * progression_factor) else 0
                        dry_eyes = 1 if np.random.rand() < (0.8 + 0.18 * progression_factor) else 0
                        dry_mouth = 1 if np.random.rand() < (0.82 + 0.16 * progression_factor) else 0
                        malar_rash = 1 if np.random.rand() < 0.05 else 0
                        raynaud = 1 if np.random.rand() < (0.3 + 0.3 * progression_factor) else 0
                        photosensitivity = 1 if np.random.rand() < 0.1 else 0
                        oral_ulcers = 1 if np.random.rand() < 0.15 else 0
                        unexplained_fever = 1 if np.random.rand() < 0.1 else 0
                    else: # Control (Fibromyalgia / Osteoarthritis)
                        arthralgia = 1 if np.random.rand() < 0.65 else 0
                        fatigue = 1 if np.random.rand() < 0.70 else 0
                        dry_eyes = 1 if np.random.rand() < 0.15 else 0
                        dry_mouth = 1 if np.random.rand() < 0.12 else 0
                        malar_rash = 0
                        raynaud = 1 if np.random.rand() < 0.08 else 0
                        photosensitivity = 0
                        oral_ulcers = 0
                        unexplained_fever = 0
                else:
                    arthralgia = fatigue = dry_eyes = dry_mouth = malar_rash = 0
                    raynaud = photosensitivity = oral_ulcers = unexplained_fever = 0
                
                # Serology Labs
                # If lab is ordered (more likely in later visits or specialist visits)
                provider = 'PCP'
                if month == diagnosis_month and disease != 'Control':
                    provider = 'Rheumatology'
                    has_seen_rheumatologist = True
                elif is_post_onset and np.random.rand() < (0.1 + 0.4 * progression_factor):
                    provider = np.random.choice(['Dermatology', 'Ophthalmology', 'Rheumatology'], p=[0.4, 0.4, 0.2])
                    if provider == 'Rheumatology':
                        has_seen_rheumatologist = True
                
                lab_ordered = (provider in ['Rheumatology', 'Dermatology']) or (is_post_onset and np.random.rand() < 0.35)
                
                if lab_ordered:
                    if disease == 'SLE':
                        ana_titer = np.random.choice([0, 40, 80, 160, 320, 640, 1280], p=[0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1])
                        anti_dsdna = float(np.random.exponential(scale=45) + 20 * progression_factor)
                        anti_ssa = float(np.random.exponential(scale=35))
                        anti_ssb = float(np.random.exponential(scale=20))
                        c3 = float(max(40.0, np.random.normal(loc=75 - 25 * progression_factor, scale=15))) # Low C3
                        c4 = float(max(5.0, np.random.normal(loc=12 - 5 * progression_factor, scale=4)))   # Low C4
                        wbc = float(max(1.8, np.random.normal(loc=4.2 - 1.2 * progression_factor, scale=0.8))) # Leukopenia
                    elif disease == 'SjD':
                        ana_titer = np.random.choice([0, 40, 80, 160, 320, 640], p=[0.1, 0.1, 0.15, 0.25, 0.25, 0.15])
                        anti_dsdna = float(np.random.exponential(scale=10))
                        anti_ssa = float(np.random.exponential(scale=90) + 40 * progression_factor) # High Anti-SSA/Ro
                        anti_ssb = float(np.random.exponential(scale=60) + 25 * progression_factor) # High Anti-SSB/La
                        c3 = float(np.random.normal(loc=110, scale=15))
                        c4 = float(np.random.normal(loc=24, scale=6))
                        wbc = float(np.random.normal(loc=5.8, scale=1.0))
                    else: # Control
                        ana_titer = np.random.choice([0, 40, 80], p=[0.8, 0.15, 0.05])
                        anti_dsdna = float(np.random.exponential(scale=5))
                        anti_ssa = float(np.random.exponential(scale=8))
                        anti_ssb = float(np.random.exponential(scale=5))
                        c3 = float(np.random.normal(loc=115, scale=12))
                        c4 = float(np.random.normal(loc=26, scale=5))
                        wbc = float(np.random.normal(loc=6.5, scale=1.1))
                else:
                    ana_titer = np.nan
                    anti_dsdna = np.nan
                    anti_ssa = np.nan
                    anti_ssb = np.nan
                    c3 = np.nan
                    c4 = np.nan
                    wbc = np.nan
                
                esr = float(np.random.normal(loc=25 + (15 * progression_factor if disease != 'Control' else 0), scale=8))
                crp = float(np.random.exponential(scale=4.0))
                
                encounter_records.append({
                    'patient_id': patient_id,
                    'encounter_id': f"ENC_{p_idx+1:04d}_{e_idx+1:02d}",
                    'month': month,
                    'provider_type': provider,
                    'arthralgia': arthralgia,
                    'chronic_fatigue': fatigue,
                    'dry_eyes': dry_eyes,
                    'dry_mouth': dry_mouth,
                    'malar_rash': malar_rash,
                    'raynaud_phenomenon': raynaud,
                    'photosensitivity': photosensitivity,
                    'oral_ulcers': oral_ulcers,
                    'unexplained_fever': unexplained_fever,
                    'ana_titer': ana_titer,
                    'anti_dsdna': anti_dsdna,
                    'anti_ssa_ro': anti_ssa,
                    'anti_ssb_la': anti_ssb,
                    'complement_c3': c3,
                    'complement_c4': c4,
                    'wbc_count': wbc,
                    'esr': esr,
                    'crp': crp,
                    'is_diagnosis_visit': 1 if (disease != 'Control' and month == diagnosis_month) else 0
                })
                
        patients_df = pd.DataFrame(patient_records)
        encounters_df = pd.DataFrame(encounter_records)
        return patients_df, encounters_df

def generate_and_save_data(output_dir: str, n_patients: int = 1500) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    simulator = EHRDataSimulator(n_patients=n_patients, random_state=42)
    patients_df, encounters_df = simulator.generate_cohort()
    
    pat_path = os.path.join(output_dir, "patients.csv")
    enc_path = os.path.join(output_dir, "encounters.csv")
    
    patients_df.to_csv(pat_path, index=False)
    encounters_df.to_csv(enc_path, index=False)
    print(f"Data generated successfully:")
    print(f" Patients: {len(patients_df)} rows saved to {pat_path}")
    print(f" Encounters: {len(encounters_df)} rows saved to {enc_path}")
    return pat_path, enc_path

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "simulated_data")
    generate_and_save_data(data_dir, n_patients=1500)
