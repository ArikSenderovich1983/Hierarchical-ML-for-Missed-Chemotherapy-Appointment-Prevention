"""
Run full HCMS pipeline: step1 -> step2 -> step3 -> step4 -> step5 -> step6
All steps use local paths. Steps 3-6 run in the same process so variables are shared.
"""
import os
import sys
import subprocess

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

def run_step(step_name, script_path):
    """Run a Python script as subprocess."""
    print(f"\n{'='*60}\n>>> Running {step_name}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=_script_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n*** {step_name} FAILED (exit code {result.returncode}) ***")
        sys.exit(result.returncode)
    print(f"\n*** {step_name} completed ***")

def main():
    # Step 1: Data ingestion (creates df_with_valid_indices_0717.csv)
    run_step("Step 1: Data Ingestion", "step1_data_ingestion.py")

    # Step 2: Feature engineering (creates df_variables_0717.csv)
    run_step("Step 2: Feature Engineering", "step2_feature_engineering.py")

    # Steps 3-6: Run in same process so step4/5/6 see step3's variables
    print(f"\n{'='*60}\n>>> Running Steps 3, 4, 5, 6 (shared namespace)\n{'='*60}")

    step3_path = os.path.join(_script_dir, "step3_Hierarchical pipeline.py")
    step4_path = os.path.join(_script_dir, "step4_semi_supervised.py")
    step5_path = os.path.join(_script_dir, "step5_explainability.py")
    step6_path = os.path.join(_script_dir, "step6_counterfactual.py")

    # Execute step3 (creates df, X_train, X_test, final_features, etc.)
    with open(step3_path, "r", encoding="utf-8") as f:
        exec(compile(f.read(), step3_path, "exec"), globals())

    # Execute step4 (uses X_noshow_benchmark, y_noshow_benchmark from step3)
    print("\n>>> Running Step 4: Semi-Supervised Learning")
    try:
        with open(step4_path, "r", encoding="utf-8") as f:
            exec(compile(f.read(), step4_path, "exec"), globals())
        print("*** Step 4 completed ***")
    except Exception as e:
        print(f"*** Step 4 failed: {e} ***")

    # Execute step5 (uses trained_stage2_model, X_test_sub_shap from step3's patient-level model)
    print("\n>>> Running Step 5: Explainability")
    try:
        # Create vars step5 expects (from step3's final model)
        import numpy as np
        stage1_pred = globals().get("stage1_predictions_patient")
        X_test_p = globals().get("X_test_p")
        if stage1_pred is not None and X_test_p is not None:
            idx_s2 = np.where(stage1_pred == 0)[0]
            globals()["X_test_sub_shap"] = np.array(X_test_p)[idx_s2]
        if "trained_stage2_model" not in globals() and "stage2_model" in globals():
            globals()["trained_stage2_model"] = globals()["stage2_model"]
        with open(step5_path, "r", encoding="utf-8") as f:
            exec(compile(f.read(), step5_path, "exec"), globals())
        print("*** Step 5 completed ***")
    except Exception as e:
        print(f"*** Step 5 failed (may need shapiq): {e} ***")

    # Execute step6 (uses X_train_primary_benchmark, X_test_primary_benchmark, final_features)
    print("\n>>> Running Step 6: Counterfactual Analysis (corrected)")
    with open(step6_path, "r", encoding="utf-8") as f:
        exec(compile(f.read(), step6_path, "exec"), globals())
    print("*** Step 6 completed ***")

    # Export tables to LaTeX
    print("\n>>> Exporting tables")
    try:
        import export_tables
        export_tables.export(globals())
    except Exception as e:
        print(f"*** Export failed: {e} ***")

    print(f"\n{'='*60}\n>>> All steps finished\n{'='*60}")

if __name__ == "__main__":
    main()
