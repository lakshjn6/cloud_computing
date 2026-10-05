"""
Master Pipeline Orchestrator
=============================
Runs all four project phases sequentially using your real 1.csv dataset:

  Phase 1 — Data preprocessing  (data_prep.py)
  Phase 2 — Model training       (train_models.py)
  Phase 3 — Cloud simulation     (cloud_sim.py)
  Phase 4 — Launch dashboard     (streamlit run dashboard.py)

Usage:
  python run_pipeline.py
"""

import sys
import os


def run_phase_1():
    print("\n" + "="*60)
    print("PHASE 1: DATA PIPELINE AND PREPROCESSING")
    print("="*60)
    from data_prep import WorkloadDataPreprocessor
    prep = WorkloadDataPreprocessor(filepath="1.csv")
    prep.process_and_save("processed_data.joblib")
    print("[OK] Phase 1 complete.\n")


def run_phase_2():
    print("\n" + "="*60)
    print("PHASE 2: MODEL TRAINING (ARIMA, XGBOOST, LSTM)")
    print("="*60)
    from train_models import run_training_pipeline
    run_training_pipeline("processed_data.joblib")
    print("[OK] Phase 2 complete.\n")


def run_phase_3():
    print("\n" + "="*60)
    print("PHASE 3: CLOUD SIMULATION ENVIRONMENT (SIMPY)")
    print("="*60)
    from cloud_sim import compare_predictive_vs_reactive
    compare_predictive_vs_reactive()
    print("[OK] Phase 3 complete.\n")


def run_phase_4():
    print("\n" + "="*60)
    print("PHASE 4: STREAMLIT DASHBOARD")
    print("="*60)
    print("To launch the dashboard, run this command:")
    print()
    print("    streamlit run dashboard.py")
    print()


if __name__ == "__main__":
    # Check that the dataset exists before starting
    if not os.path.exists("1.csv"):
        print(
            "[ERROR] Dataset '1.csv' not found in the current directory.\n"
            "Please place your Bitbrains workload trace CSV file in:\n"
            f"  {os.path.abspath('1.csv')}"
        )
        sys.exit(1)

    run_phase_1()
    run_phase_2()
    run_phase_3()
    run_phase_4()
