# ==============================
#  TABLE OF CONTENTS 
# ==============================

# 0. Imports & Utility
# 1. Stage 1: Cancellation Detection
# 2. Stage 2: Show Prediction
# 3. Predict-only Helpers
# 4. Data Setup & Thresholds
# 5. Train 10 Seeds (Decision Tree + XGB)
# 6. Counterfactuals
#    - CSS-BP Intervention
#    - Extreme Values
#    - Literature-Based Interventions
#    - Confirmation (doubled)
#    - Clinic/Provider Consistency
#    - Scheduling Interventions
# 11. Variable Ranges (Table S10 helper)
# 12. End of Script


# =========================================
# 10 MODELS PER STAGE + COUNTERFACTUALS (NO SAVE/LOAD)
# =========================================

# -------- Imports --------
import numpy as np
import pandas as pd
import random, os
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# =========================================
# Utility
# =========================================
def apply_threshold(y_scores, threshold):
    return (y_scores >= threshold).astype(int)

# =========================================
# Stage 1: Cancellation detection (TRAINING)
# =========================================
def stage_one_fixed_threshold(model, X_train, X_test, y_train, y_test, threshold=0.43):
    y_train_bin = (y_train == 2).astype(int)
    y_test_bin = (y_test == 2).astype(int)

    model.fit(X_train, y_train_bin)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = apply_threshold(y_test_proba, threshold)

    print("\n=== Stage 1 Evaluation (Row-wise Split) ===")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(
        y_test_bin, y_test_pred,
        target_names=["Not Cancellation", "Cancellation"],
        zero_division=0
    ))

    return y_test_pred, model

# =========================================
# Stage 2: Show prediction (TRAINING)
# =========================================
def stage_two_fixed_threshold(model, X_train, X_test, y_train, y_test, stage1_pred, threshold=0.83):
    idx_train = np.where(y_train != 2)[0]
    idx_test = np.where(stage1_pred == 0)[0]

    X_train_sub = X_train[idx_train]
    y_train_sub = y_train[idx_train]
    X_test_sub = X_test[idx_test]
    y_test_sub = y_test[idx_test]

    y_train_bin = (y_train_sub == 0).astype(int)
    y_test_bin = (y_test_sub == 0).astype(int)

    model.fit(X_train_sub, y_train_bin)
    y_test_proba = model.predict_proba(X_test_sub)[:, 1]
    y_test_pred = apply_threshold(y_test_proba, threshold)

    print("\n=== Stage 2 Evaluation (Row-wise Split) ===")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(
        y_test_bin, y_test_pred,
        target_names=["Not Show", "Show"],
        zero_division=0
    ))

    return y_test_pred, model, X_test_sub, y_test_sub

# =========================================
# Stage 1: Predict only
# =========================================
def stage_one_predict_only(trained_model, X_test, threshold=0.43):
    y_test_proba = trained_model.predict_proba(X_test)[:, 1]
    return apply_threshold(y_test_proba, threshold)

# =========================================
# Stage 2: Predict only
# =========================================
def stage_two_predict_only(trained_model, X_test, stage1_pred, threshold=0.83):
    idx_test = np.where(stage1_pred == 0)[0]
    X_test_sub = X_test[idx_test]
    y_test_proba = trained_model.predict_proba(X_test_sub)[:, 1]
    return apply_threshold(y_test_proba, threshold), X_test_sub

# =========================================
# Data (expects these to already exist in your session)
#   - X_train_primary_benchmark, X_test_primary_benchmark
#   - y_train_primary_benchmark, y_test_primary_benchmark
#   - final_features (list of feature names for DataFrame columns)
# =========================================

# Convert to arrays for training
X_train_df = X_train_primary_benchmark.copy()
X_test_df  = X_test_primary_benchmark.copy()
X_train = np.array(X_train_df)
X_test  = np.array(X_test_df)
y_train = np.array(y_train_primary_benchmark)
y_test  = np.array(y_test_primary_benchmark)

# Thresholds
stage1_threshold = 0.43
stage2_threshold = 0.83

# =========================================
# Train 10 seeds (in-memory, no save/load)
# =========================================
models = {}
for seed in range(10):
    print(f"\n=== Training models with seed {seed} ===")
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Stage 1
    stage1_model = DecisionTreeClassifier(random_state=seed)
    stage1_preds, trained_stage1_model = stage_one_fixed_threshold(
        stage1_model, X_train, X_test, y_train, y_test, threshold=stage1_threshold
    )

    # Stage 2
    stage2_model = XGBClassifier(
        eval_metric='logloss',
        random_state=seed,
        nthread=1,
        use_label_encoder=False,
        tree_method='hist',
        enable_categorical=False,
        verbosity=0
    )
    stage2_preds, trained_stage2_model, _, _ = stage_two_fixed_threshold(
        stage2_model, X_train, X_test, y_train, y_test, stage1_preds, threshold=stage2_threshold
    )

    models[seed] = {
        "stage1": trained_stage1_model,
        "stage2": trained_stage2_model
    }

print("\n✅ All 10 models trained and stored in-memory (models[seed]['stage1'/'stage2']).")

# Keep a global baseline copy for later sections
X_test_original_df = X_test_df.copy()

# =========================================
# Commitment slot, 10 models (CSS-BP)
# =========================================
print("\n=== Counterfactual Analysis: Commitment Slot Scheduling Intervention (CSS-BP) ===")

all_results_cssbp = []

percent_changes_cssbp = {
    "No_Show_Left_Count_reduce": 20,
    "Average_Arrival_Lag_Hours_reduce": 20,
    "Socioeconomic_reduce": 20
}

for seed in range(10):
    print(f"\n================ SEED {seed} =================")

    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    X_test_original_df = X_test_df.copy()
    orig_X_test = X_test_original_df[final_features].values

    # --- Apply Counterfactual Intervention ---
    X_test_cf_cssbp = X_test_original_df.copy()

    X_test_cf_cssbp["No_Show_Left_Count"] = (
        X_test_cf_cssbp["No_Show_Left_Count"] * (1 - percent_changes_cssbp["No_Show_Left_Count_reduce"] / 100)
    ).round().astype(int)

    X_test_cf_cssbp["Average_Arrival_Lag_Hours"] = (
        X_test_cf_cssbp["Average_Arrival_Lag_Hours"] * (1 - percent_changes_cssbp["Average_Arrival_Lag_Hours_reduce"] / 100)
    )

    X_test_cf_cssbp["Socioeconomic"] = (
        X_test_cf_cssbp["Socioeconomic"] * (1 - percent_changes_cssbp["Socioeconomic_reduce"] / 100)
    ).round().astype(int)

    cf_X_test_cssbp = X_test_cf_cssbp[final_features].values

    # Predictions for cancellations
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf   = stage_one_predict_only(stage1_model, cf_X_test_cssbp, threshold=stage1_threshold)

    orig_cancel_rate = np.mean(stage1_preds_orig == 1)
    cf_cancel_rate   = np.mean(stage1_preds_cf == 1)
    cancel_reduction = (orig_cancel_rate - cf_cancel_rate) * 100

    # Predictions for no-shows
    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)
    stage2_preds_cf, _   = stage_two_predict_only(stage2_model, cf_X_test_cssbp, stage1_preds_cf, threshold=stage2_threshold)

    orig_noshow_rate = np.mean(stage2_preds_orig == 0)
    cf_noshow_rate   = np.mean(stage2_preds_cf == 0)
    noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

    print("\n========== CSS-BP COUNTERFACTUAL SUMMARY ==========")
    print(f"Original cancellation rate:         {orig_cancel_rate*100:.2f}%")
    print(f"Counterfactual cancellation rate:   {cf_cancel_rate*100:.2f}%")
    print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points\n")
    print(f"Original no-show rate:              {orig_noshow_rate*100:.2f}%")
    print(f"Counterfactual no-show rate:        {cf_noshow_rate*100:.2f}%")
    print(f"Absolute reduction in no-shows:     {noshow_reduction:.2f} percentage points")
    print("====================================================")

    all_results_cssbp.append([seed, "Stage 1", cancel_reduction])
    all_results_cssbp.append([seed, "Stage 2", noshow_reduction])

# Summary
df_results_cssbp = pd.DataFrame(all_results_cssbp, columns=["Seed", "Stage", "Δ (%)"])
summary_cssbp = df_results_cssbp.groupby("Stage")["Δ (%)"].agg(["mean", "std"])

print("\n=============== FINAL SUMMARY: CSS-BP INTERVENTION ===============")
print(summary_cssbp)

# =========================================
# Extreme for 10 models
# =========================================
import numpy as np
import pandas as pd

# Store results for all seeds and all interventions
all_results = []

stage1_threshold = 0.43
stage2_threshold = 0.83

# Define once for reuse (also used later in "range of variables")
extreme_values = {
    "Cancellation_Count": 0,
    "No_Show_Left_Count": 0,
    "Rescheduled_Appointments_Count": 0,
    "Average_Arrival_Lag_Hours": 0,
    "Avg_Schd_Encounter_Diff_Hours": 0,
    "Patient Choice": 0,
    "Clinic Management": 0,
    "Provider Choice": 0,
    "Socioeconomic": 0,
    "Provider_Change_Count": 0,
    "Department_Change_Count": 0,
    "Provider_Changed_Last_Visit": 0,
    "Department_Changed_Last_Visit": 0
}

# Loop over 10 models
for seed in range(10):
    print(f"\n================ SEED {seed} =================\n")
    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    # Keep a fresh baseline copy of the test data
    X_test_original_df = X_test_df.copy()

    # Original test matrix
    orig_X_test = X_test_original_df[final_features].values
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)

    # ------------------------------------
    # === Counterfactual Analysis: Extreme Values (Numerical Variables) ===
    # ------------------------------------
    for var, extreme_value in extreme_values.items():
        X_test_cf_df = X_test_original_df.copy()
        if var in X_test_cf_df.columns:
            X_test_cf_df[var] = extreme_value
        cf_X_test = X_test_cf_df[final_features].values

        # Counterfactual predictions
        stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test, threshold=stage1_threshold)
        stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test, stage1_preds_cf, threshold=stage2_threshold)

        # Use fixed baseline denominator for comparability
        orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
        cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
        orig_noshow_rate = np.mean(stage2_preds_orig == 0)
        cf_noshow_rate = np.mean(stage2_preds_cf == 0)

        cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100
        noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

        # Print
        print(f"\n--- Counterfactual Analysis: {var} → {extreme_value} ---")
        print("\n========== COUNTERFACTUAL SUMMARY ==========")
        print(f"Original cancellation rate:        {orig_cancellation_rate*100:.2f}%")
        print(f"Counterfactual cancellation rate:  {cf_cancellation_rate*100:.2f}%")
        print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
        print(f"Original no-show rate:             {orig_noshow_rate*100:.2f}%")
        print(f"Counterfactual no-show rate:       {cf_noshow_rate*100:.2f}%")
        print(f"Absolute reduction in no-shows:    {noshow_reduction:.2f} percentage points")
        print("=============================================")

        all_results.append([seed, var, cancel_reduction, noshow_reduction])

    # ------------------------------------
    # === Encounter Hour (categorical all-to-one) ===
    # ------------------------------------
    encounter_hour_scenarios = {
        "Encounter Hour: before 12pm": {"Encounter_Hour_before 12pm": 1, "Encounter_Hour_between 12-18pm": 0, "Encounter_Hour_after 18pm": 0},
        "Encounter Hour: 12–18pm": {"Encounter_Hour_before 12pm": 0, "Encounter_Hour_between 12-18pm": 1, "Encounter_Hour_after 18pm": 0},
        "Encounter Hour: after 18pm": {"Encounter_Hour_before 12pm": 0, "Encounter_Hour_between 12-18pm": 0, "Encounter_Hour_after 18pm": 1},
    }

    for scenario_name, changes in encounter_hour_scenarios.items():
        X_test_cf_df = X_test_original_df.copy()
        for col, value in changes.items():
            if col in X_test_cf_df.columns:
                X_test_cf_df[col] = value
        cf_X_test = X_test_cf_df[final_features].values

        stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test, threshold=stage1_threshold)
        stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test, stage1_preds_cf, threshold=stage2_threshold)

        orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
        cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
        orig_noshow_rate = np.mean(stage2_preds_orig == 0)
        cf_noshow_rate = np.mean(stage2_preds_cf == 0)

        cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100
        noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

        print(f"\n--- {scenario_name} ---")
        print("\n========== COUNTERFACTUAL SUMMARY ==========")
        print(f"Original cancellation rate:        {orig_cancellation_rate*100:.2f}%")
        print(f"Counterfactual cancellation rate:  {cf_cancellation_rate*100:.2f}%")
        print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
        print(f"Original no-show rate:             {orig_noshow_rate*100:.2f}%")
        print(f"Counterfactual no-show rate:       {cf_noshow_rate*100:.2f}%")
        print(f"Absolute reduction in no-shows:    {noshow_reduction:.2f} percentage points")
        print("=============================================")

        all_results.append([seed, scenario_name, cancel_reduction, noshow_reduction])

    # ------------------------------------
    # === Weekend Indicator (categorical all-to-one) ===
    # ------------------------------------
    weekend_scenarios = {
        "Weekend Indicator: Weekday": {"Weekend_Indicator_0": 1, "Weekend_Indicator_1": 0},
        "Weekend Indicator: Weekend": {"Weekend_Indicator_0": 0, "Weekend_Indicator_1": 1},
    }

    for scenario_name, changes in weekend_scenarios.items():
        X_test_cf_df = X_test_original_df.copy()
        for col, value in changes.items():
            if col in X_test_cf_df.columns:
                X_test_cf_df[col] = value
        cf_X_test = X_test_cf_df[final_features].values

        stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test, threshold=stage1_threshold)
        stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test, stage1_preds_cf, threshold=stage2_threshold)

        orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
        cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
        orig_noshow_rate = np.mean(stage2_preds_orig == 0)
        cf_noshow_rate = np.mean(stage2_preds_cf == 0)

        cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100
        noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

        print(f"\n--- {scenario_name} ---")
        print("\n========== COUNTERFACTUAL SUMMARY ==========")
        print(f"Original cancellation rate:        {orig_cancellation_rate*100:.2f}%")
        print(f"Counterfactual cancellation rate:  {cf_cancellation_rate*100:.2f}%")
        print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
        print(f"Original no-show rate:             {orig_noshow_rate*100:.2f}%")
        print(f"Counterfactual no-show rate:       {cf_noshow_rate*100:.2f}%")
        print(f"Absolute reduction in no-shows:    {noshow_reduction:.2f} percentage points")
        print("=============================================")

        all_results.append([seed, scenario_name, cancel_reduction, noshow_reduction])

# ------------------------------------
# === Final Summary Across Seeds ===
# ------------------------------------
df_results = pd.DataFrame(all_results, columns=["Seed", "Intervention", "Δ Cancel (%)", "Δ No-show (%)"])
summary = df_results.groupby("Intervention")[["Δ Cancel (%)", "Δ No-show (%)"]].agg(["mean", "std"])

print("\n================ FINAL SUMMARY (Mean ± SD across 10 models) =================")
print(summary)

# =========================================
# Reminder based on literature (cap amount), 10 seeds
# =========================================
import numpy as np
import pandas as pd

# Store results for all seeds and both stages
all_results_lit = []

stage1_threshold = 0.43
stage2_threshold = 0.83

print("\n=== Counterfactual Analysis: Literature Split by Stage ===")

# Loop over 10 models
for seed in range(10):
    print(f"\n================ SEED {seed} =================")

    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    X_test_original_df = X_test_df.copy()
    orig_X_test = X_test_original_df[final_features].values

    # =====================================================
    # --- Stage 1 Counterfactual (Cancellations Only) ---
    # =====================================================
    print("\n--- Counterfactual Analysis: Stage 1 (Cancellations Only) ---")

    X_test_cf_stage1 = X_test_original_df.copy()

    percent_changes_stage1 = {
        "Cancellation_Count_increase": 12,
        "No_Show_Left_Count_reduce": 9,
        "Rescheduled_Appointments_Count_increase": 15,
        "Average_Arrival_Lag_Hours_reduce": 10,
        "Patient_Choice_reduce": 7,
        "Last_Appointment_Status_change": 11
    }

    # Apply changes
    X_test_cf_stage1["Cancellation_Count"] = (
        X_test_cf_stage1["Cancellation_Count"] * (1 + percent_changes_stage1["Cancellation_Count_increase"]/100)
    ).round().astype(int)

    X_test_cf_stage1["No_Show_Left_Count"] = (
        X_test_cf_stage1["No_Show_Left_Count"] * (1 - percent_changes_stage1["No_Show_Left_Count_reduce"]/100)
    ).round().astype(int)

    X_test_cf_stage1["Rescheduled_Appointments_Count"] = (
        X_test_cf_stage1["Rescheduled_Appointments_Count"] * (1 + percent_changes_stage1["Rescheduled_Appointments_Count_increase"]/100)
    ).round().astype(int)

    X_test_cf_stage1["Average_Arrival_Lag_Hours"] = (
        X_test_cf_stage1["Average_Arrival_Lag_Hours"] * (1 - percent_changes_stage1["Average_Arrival_Lag_Hours_reduce"]/100)
    )

    X_test_cf_stage1["Patient Choice"] = (
        X_test_cf_stage1["Patient Choice"] * (1 - percent_changes_stage1["Patient_Choice_reduce"]/100)
    ).round().astype(int).clip(lower=0)

    # Handle Last_Appointment_Status changes
    status_change_pct = percent_changes_stage1["Last_Appointment_Status_change"] / 100
    np.random.seed(seed)
    idx_noshow_1 = X_test_cf_stage1.index[X_test_cf_stage1["Last_Appointment_Status_No Show"] == 1].tolist()
    n_noshow_change = int(len(idx_noshow_1) * status_change_pct)
    if n_noshow_change > 0:
        idx_noshow_change = np.random.choice(idx_noshow_1, size=n_noshow_change, replace=False)
        X_test_cf_stage1.loc[idx_noshow_change, "Last_Appointment_Status_No Show"] = 0

    idx_completed_0 = X_test_cf_stage1.index[X_test_cf_stage1["Last_Appointment_Status_Completed"] == 0].tolist()
    n_completed_change = int(len(idx_completed_0) * status_change_pct)
    if n_completed_change > 0:
        idx_completed_change = np.random.choice(idx_completed_0, size=n_completed_change, replace=False)
        X_test_cf_stage1.loc[idx_completed_change, "Last_Appointment_Status_Completed"] = 1

    cf_X_test_stage1 = X_test_cf_stage1[final_features].values

    # Predictions
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf   = stage_one_predict_only(stage1_model, cf_X_test_stage1, threshold=stage1_threshold)

    orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
    cf_cancellation_rate   = np.mean(stage1_preds_cf == 1)

    cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100

    print("\n========== STAGE 1 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original cancellation rate:         {orig_cancellation_rate*100:.2f}%")
    print(f"Counterfactual cancellation rate:   {cf_cancellation_rate*100:.2f}%")
    print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
    print("======================================================")

    all_results_lit.append([seed, "Stage 1", cancel_reduction])

    # =====================================================
    # --- Stage 2 Counterfactual (No-shows Only) ---
    # =====================================================
    print("\n--- Counterfactual Analysis: Stage 2 (No-shows Only) ---")

    X_test_cf_stage2 = X_test_original_df.copy()

    percent_changes_stage2 = {
        "Cancellation_Count_increase": 12,
        "No_Show_Left_Count_reduce": 9,
        "Rescheduled_Appointments_Count_increase": 15,
        "Average_Arrival_Lag_Hours_reduce": 10,
        "Patient_Choice_reduce": 7,
        "Last_Appointment_Status_change": 11
    }

    # Apply changes
    X_test_cf_stage2["Cancellation_Count"] = (
        X_test_cf_stage2["Cancellation_Count"] * (1 + percent_changes_stage2["Cancellation_Count_increase"]/100)
    ).round().astype(int)

    X_test_cf_stage2["No_Show_Left_Count"] = (
        X_test_cf_stage2["No_Show_Left_Count"] * (1 - percent_changes_stage2["No_Show_Left_Count_reduce"]/100)
    ).round().astype(int)

    X_test_cf_stage2["Rescheduled_Appointments_Count"] = (
        X_test_cf_stage2["Rescheduled_Appointments_Count"] * (1 + percent_changes_stage2["Rescheduled_Appointments_Count_increase"]/100)
    ).round().astype(int)

    X_test_cf_stage2["Average_Arrival_Lag_Hours"] = (
        X_test_cf_stage2["Average_Arrival_Lag_Hours"] * (1 - percent_changes_stage2["Average_Arrival_Lag_Hours_reduce"]/100)
    )

    X_test_cf_stage2["Patient Choice"] = (
        X_test_cf_stage2["Patient Choice"] * (1 - percent_changes_stage2["Patient_Choice_reduce"]/100)
    ).round().astype(int).clip(lower=0)

    # Handle Last_Appointment_Status changes
    status_change_pct_2 = percent_changes_stage2["Last_Appointment_Status_change"] / 100
    np.random.seed(seed)
    idx_noshow_1 = X_test_cf_stage2.index[X_test_cf_stage2["Last_Appointment_Status_No Show"] == 1].tolist()
    n_noshow_change = int(len(idx_noshow_1) * status_change_pct_2)
    if n_noshow_change > 0:
        idx_noshow_change = np.random.choice(idx_noshow_1, size=n_noshow_change, replace=False)
        X_test_cf_stage2.loc[idx_noshow_change, "Last_Appointment_Status_No Show"] = 0

    idx_completed_0 = X_test_cf_stage2.index[X_test_cf_stage2["Last_Appointment_Status_Completed"] == 0].tolist()
    n_completed_change = int(len(idx_completed_0) * status_change_pct_2)
    if n_completed_change > 0:
        idx_completed_change = np.random.choice(idx_completed_0, size=n_completed_change, replace=False)
        X_test_cf_stage2.loc[idx_completed_change, "Last_Appointment_Status_Completed"] = 1

    cf_X_test_stage2 = X_test_cf_stage2[final_features].values

    # Predictions
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf   = stage_one_predict_only(stage1_model, cf_X_test_stage2, threshold=stage1_threshold)

    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)
    stage2_preds_cf, _   = stage_two_predict_only(stage2_model, cf_X_test_stage2, stage1_preds_cf, threshold=stage2_threshold)

    orig_noshow_rate = np.mean(stage2_preds_orig == 0)
    cf_noshow_rate   = np.mean(stage2_preds_cf == 0)

    noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

    print("\n========== STAGE 2 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original no-show rate:              {orig_noshow_rate*100:.2f}%")
    print(f"Counterfactual no-show rate:        {cf_noshow_rate*100:.2f}%")
    print(f"Absolute reduction in no-shows:     {noshow_reduction:.2f} percentage points")
    print("======================================================")

    all_results_lit.append([seed, "Stage 2", noshow_reduction])

# =====================================================
# --- Final Summary Across Seeds ---
# =====================================================
df_results_lit = pd.DataFrame(all_results_lit, columns=["Seed", "Stage", "Δ (%)"])
summary_lit = df_results_lit.groupby("Stage")["Δ (%)"].agg(["mean", "std"])

print("\n================ FINAL SUMMARY (Mean ± SD across 10 models) =================")
print(summary_lit)

# =========================================
# Patient confirmation (doubled) — 10 seeds
# =========================================
import numpy as np
import pandas as pd

stage1_threshold = 0.43
stage2_threshold = 0.83

all_results_stage1_conf = []
all_results_stage2_conf = []

print("\n=== Counterfactual Analysis: Confirmation (doubled) Split by Stage ===")

# Loop over all 10 trained models
for seed in range(10):
    print(f"\n================ SEED {seed} =================")
    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    # Baseline test set
    X_test_original_df = X_test_df.copy()
    orig_X_test = X_test_original_df[final_features].values

    # ======================================================
    # Stage 1 Counterfactual (Cancellations Only)
    # ======================================================
    print("\n--- Counterfactual Analysis: Stage 1 (Cancellations Only) ---")

    X_test_cf_stage1 = X_test_original_df.copy()
    percent_changes_stage1 = {
        "Cancellation_Count_increase": 24,
        "No_Show_Left_Count_reduce": 18,
        "Rescheduled_Appointments_Count_increase": 30,
        "Average_Arrival_Lag_Hours_reduce": 20,
        "Patient_Choice_reduce": 14,
        "Last_Appointment_Status_change": 22
    }

    # Apply changes
    X_test_cf_stage1["Cancellation_Count"] *= (1 + percent_changes_stage1["Cancellation_Count_increase"]/100)
    X_test_cf_stage1["No_Show_Left_Count"] *= (1 - percent_changes_stage1["No_Show_Left_Count_reduce"]/100)
    X_test_cf_stage1["Rescheduled_Appointments_Count"] *= (1 + percent_changes_stage1["Rescheduled_Appointments_Count_increase"]/100)
    X_test_cf_stage1["Average_Arrival_Lag_Hours"] *= (1 - percent_changes_stage1["Average_Arrival_Lag_Hours_reduce"]/100)
    X_test_cf_stage1["Patient Choice"] *= (1 - percent_changes_stage1["Patient_Choice_reduce"]/100)

    # Round + clip
    X_test_cf_stage1["Cancellation_Count"] = X_test_cf_stage1["Cancellation_Count"].round().astype(int)
    X_test_cf_stage1["No_Show_Left_Count"] = X_test_cf_stage1["No_Show_Left_Count"].round().astype(int)
    X_test_cf_stage1["Rescheduled_Appointments_Count"] = X_test_cf_stage1["Rescheduled_Appointments_Count"].round().astype(int)
    X_test_cf_stage1["Patient Choice"] = X_test_cf_stage1["Patient Choice"].round().astype(int).clip(lower=0)

    # Last appointment status
    status_change_pct = percent_changes_stage1["Last_Appointment_Status_change"] / 100
    np.random.seed(seed)
    idx_noshow_1 = X_test_cf_stage1.index[X_test_cf_stage1["Last_Appointment_Status_No Show"] == 1].tolist()
    n_noshow_change = int(len(idx_noshow_1) * status_change_pct)
    if n_noshow_change > 0:
        idx_noshow_change = np.random.choice(idx_noshow_1, size=n_noshow_change, replace=False)
        X_test_cf_stage1.loc[idx_noshow_change, "Last_Appointment_Status_No Show"] = 0

    idx_completed_0 = X_test_cf_stage1.index[X_test_cf_stage1["Last_Appointment_Status_Completed"] == 0].tolist()
    n_completed_change = int(len(idx_completed_0) * status_change_pct)
    if n_completed_change > 0:
        idx_completed_change = np.random.choice(idx_completed_0, size=n_completed_change, replace=False)
        X_test_cf_stage1.loc[idx_completed_change, "Last_Appointment_Status_Completed"] = 1

    # Predictions
    cf_X_test_stage1 = X_test_cf_stage1[final_features].values
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage1, threshold=stage1_threshold)

    orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
    cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
    cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100

    print("\n========== STAGE 1 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original cancellation rate:         {orig_cancellation_rate*100:.2f}%")
    print(f"Counterfactual cancellation rate:   {cf_cancellation_rate*100:.2f}%")
    print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_stage1_conf.append([seed, cancel_reduction])

    # ======================================================
    # Stage 2 Counterfactual (No-shows Only)
    # ======================================================
    print("\n--- Counterfactual Analysis: Stage 2 (No-shows Only) ---")

    X_test_cf_stage2 = X_test_original_df.copy()
    percent_changes_stage2 = percent_changes_stage1  # same values

    # Apply changes
    X_test_cf_stage2["Cancellation_Count"] *= (1 + percent_changes_stage2["Cancellation_Count_increase"]/100)
    X_test_cf_stage2["No_Show_Left_Count"] *= (1 - percent_changes_stage2["No_Show_Left_Count_reduce"]/100)
    X_test_cf_stage2["Rescheduled_Appointments_Count"] *= (1 + percent_changes_stage2["Rescheduled_Appointments_Count_increase"]/100)
    X_test_cf_stage2["Average_Arrival_Lag_Hours"] *= (1 - percent_changes_stage2["Average_Arrival_Lag_Hours_reduce"]/100)
    X_test_cf_stage2["Patient Choice"] *= (1 - percent_changes_stage2["Patient_Choice_reduce"]/100)

    # Round + clip
    X_test_cf_stage2["Cancellation_Count"] = X_test_cf_stage2["Cancellation_Count"].round().astype(int)
    X_test_cf_stage2["No_Show_Left_Count"] = X_test_cf_stage2["No_Show_Left_Count"].round().astype(int)
    X_test_cf_stage2["Rescheduled_Appointments_Count"] = X_test_cf_stage2["Rescheduled_Appointments_Count"].round().astype(int)
    X_test_cf_stage2["Patient Choice"] = X_test_cf_stage2["Patient Choice"].round().astype(int).clip(lower=0)

    # Last appointment status
    status_change_pct_2 = percent_changes_stage2["Last_Appointment_Status_change"] / 100
    np.random.seed(seed)
    idx_noshow_1 = X_test_cf_stage2.index[X_test_cf_stage2["Last_Appointment_Status_No Show"] == 1].tolist()
    n_noshow_change = int(len(idx_noshow_1) * status_change_pct_2)
    if n_noshow_change > 0:
        idx_noshow_change = np.random.choice(idx_noshow_1, size=n_noshow_change, replace=False)
        X_test_cf_stage2.loc[idx_noshow_change, "Last_Appointment_Status_No Show"] = 0

    idx_completed_0 = X_test_cf_stage2.index[X_test_cf_stage2["Last_Appointment_Status_Completed"] == 0].tolist()
    n_completed_change = int(len(idx_completed_0) * status_change_pct_2)
    if n_completed_change > 0:
        idx_completed_change = np.random.choice(idx_completed_0, size=n_completed_change, replace=False)
        X_test_cf_stage2.loc[idx_completed_change, "Last_Appointment_Status_Completed"] = 1

    # Predictions
    cf_X_test_stage2 = X_test_cf_stage2[final_features].values
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)

    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage2, threshold=stage1_threshold)
    stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test_stage2, stage1_preds_cf, threshold=stage2_threshold)

    orig_noshow_rate = np.mean(stage2_preds_orig == 0)
    cf_noshow_rate = np.mean(stage2_preds_cf == 0)
    noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

    print("\n========== STAGE 2 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original no-show rate:              {orig_noshow_rate*100:.2f}%")
    print(f"Counterfactual no-show rate:        {cf_noshow_rate*100:.2f}%")
    print(f"Absolute reduction in no-shows:     {noshow_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_stage2_conf.append([seed, noshow_reduction])

# ======================================================
# Final Summary Across Seeds
# ======================================================
df_stage1_conf = pd.DataFrame(all_results_stage1_conf, columns=["Seed", "Δ Cancel (%)"])
df_stage2_conf = pd.DataFrame(all_results_stage2_conf, columns=["Seed", "Δ No-show (%)"])

summary_stage1_conf = df_stage1_conf["Δ Cancel (%)"].agg(["mean", "std"])
summary_stage2_conf = df_stage2_conf["Δ No-show (%)"].agg(["mean", "std"])

print("\n================ FINAL SUMMARY (Mean ± SD across 10 models) =================")
print(f"Stage 1 (Cancel): {summary_stage1_conf['mean']:.2f} ± {summary_stage1_conf['std']:.2f} pp")
print(f"Stage 2 (No-show): {summary_stage2_conf['mean']:.2f} ± {summary_stage2_conf['std']:.2f} pp")
print("============================================================================")

# =========================================
# Clinic/provider consistency — 10 seeds
# =========================================
import numpy as np
import pandas as pd

stage1_threshold = 0.43
stage2_threshold = 0.83

all_results_stage1_split = []
all_results_stage2_split = []

print("\n=== Counterfactual Analysis: Stage 1 and Stage 2 Split ===")

# Loop across all 10 trained models
for seed in range(10):
    print(f"\n================ SEED {seed} =================")
    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    X_test_original_df = X_test_df.copy()
    orig_X_test = X_test_original_df[final_features].values

    # ---------------------------------------------------------
    # Stage 1 Counterfactual (Cancellations Only)
    # ---------------------------------------------------------
    print("\n--- Counterfactual Analysis: Stage 1 (Cancellations Only) ---")

    X_test_cf_stage1 = X_test_original_df.copy()

    # Apply capping & edits
    edits_stage1 = {
        "Cancellation_Count": 9,
        "Rescheduled_Appointments_Count": 9,
        "Patient Choice": 9,
        "Provider_Change_Count": 6,
        "Department_Change_Count": 6,
    }
    for col, cap in edits_stage1.items():
        if col in X_test_cf_stage1.columns:
            X_test_cf_stage1[col] = X_test_cf_stage1[col].clip(upper=cap)

    if "Provider_Changed_Last_Visit" in X_test_cf_stage1.columns:
        X_test_cf_stage1["Provider_Changed_Last_Visit"] = 0
    if "Department_Changed_Last_Visit" in X_test_cf_stage1.columns:
        X_test_cf_stage1["Department_Changed_Last_Visit"] = 0

    cf_X_test_stage1 = X_test_cf_stage1[final_features].values

    # Predictions
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage1, threshold=stage1_threshold)

    orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
    cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
    cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100

    print("\n========== STAGE 1 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original cancellation rate:         {orig_cancellation_rate*100:.2f}%")
    print(f"Counterfactual cancellation rate:   {cf_cancellation_rate*100:.2f}%")
    print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_stage1_split.append([seed, cancel_reduction])

    # ---------------------------------------------------------
    # Stage 2 Counterfactual (No-shows Only)
    # ---------------------------------------------------------
    print("\n--- Counterfactual Analysis: Stage 2 (No-shows Only) ---")

    X_test_cf_stage2 = X_test_original_df.copy()

    edits_stage2 = {
        "Cancellation_Count": 9,
        "Rescheduled_Appointments_Count": 9,
        "Patient Choice": 9,
        "Provider_Change_Count": 6,
        "Department_Change_Count": 6,
    }
    for col, cap in edits_stage2.items():
        if col in X_test_cf_stage2.columns:
            X_test_cf_stage2[col] = X_test_cf_stage2[col].clip(upper=cap)

    if "Provider_Changed_Last_Visit" in X_test_cf_stage2.columns:
        X_test_cf_stage2["Provider_Changed_Last_Visit"] = 0
    if "Department_Changed_Last_Visit" in X_test_cf_stage2.columns:
        X_test_cf_stage2["Department_Changed_Last_Visit"] = 0

    cf_X_test_stage2 = X_test_cf_stage2[final_features].values

    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)

    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage2, threshold=stage1_threshold)
    stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test_stage2, stage1_preds_cf, threshold=stage2_threshold)

    orig_noshow_rate = np.mean(stage2_preds_orig == 0)
    cf_noshow_rate = np.mean(stage2_preds_cf == 0)
    noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

    print("\n========== STAGE 2 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original no-show rate:              {orig_noshow_rate*100:.2f}%")
    print(f"Counterfactual no-show rate:        {cf_noshow_rate*100:.2f}%")
    print(f"Absolute reduction in no-shows:     {noshow_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_stage2_split.append([seed, noshow_reduction])

# ---------------------------------------------------------
# Final Mean ± SD Summary
# ---------------------------------------------------------
df_stage1_split = pd.DataFrame(all_results_stage1_split, columns=["Seed", "Δ Cancel (%)"])
df_stage2_split = pd.DataFrame(all_results_stage2_split, columns=["Seed", "Δ No-show (%)"])

summary_stage1_split = df_stage1_split["Δ Cancel (%)"].agg(["mean", "std"])
summary_stage2_split = df_stage2_split["Δ No-show (%)"].agg(["mean", "std"])

print("\n================ FINAL SUMMARY (Mean ± SD across 10 models) =================")
print(f"Stage 1 (Cancel, Split Caps): {summary_stage1_split['mean']:.2f} ± {summary_stage1_split['std']:.2f} pp")
print(f"Stage 2 (No-show, Split Caps): {summary_stage2_split['mean']:.2f} ± {summary_stage2_split['std']:.2f} pp")
print("============================================================================")

# =========================================
# Scheduling time literature — 10 seeds
# =========================================
import numpy as np
import pandas as pd

stage1_threshold = 0.43
stage2_threshold = 0.83

all_results_sched_stage1 = []
all_results_sched_stage2 = []

print("\n=== Counterfactual Analysis: SCHEDULING INTERVENTIONS Split by Stage ===")

# Loop across all 10 models
for seed in range(10):
    print(f"\n================ SEED {seed} =================")
    stage1_model = models[seed]["stage1"]
    stage2_model = models[seed]["stage2"]

    X_test_original_df = X_test_df.copy()
    orig_X_test = X_test_original_df[final_features].values

    # ======================================================
    # Stage 1 Counterfactual (Cancellations Only)
    # ======================================================
    print("\n--- Counterfactual Analysis: Stage 1 (Cancellations Only) ---")
    X_test_cf_stage1 = X_test_original_df.copy()

    percent_changes_stage1 = {
        "Avg_Schd_Encounter_Diff_Hours_cap": 168,
        "Cancellation_Count_reduce": 50,
        "No_Show_Left_Count_reduce": 50,
        "Rescheduled_Appointments_Count_reduce": 50,
        "Patient_Choice_reduce": 50,
        "Provider_Choice_reduce": 50,
        "Clinic_Management_reduce": 50,
        "Socioeconomic_reduce": 50,
        "Weekend_Indicator_change": 30,
        "Encounter_Hour_change": 30
    }

    # Cap Avg_Schd_Encounter_Diff_Hours
    if "Avg_Schd_Encounter_Diff_Hours" in X_test_cf_stage1.columns:
        X_test_cf_stage1["Avg_Schd_Encounter_Diff_Hours"] = X_test_cf_stage1["Avg_Schd_Encounter_Diff_Hours"].apply(
            lambda x: min(x, percent_changes_stage1["Avg_Schd_Encounter_Diff_Hours_cap"]) if pd.notnull(x) else x
        )

    # Apply reductions
    for var, key in [
        ("Cancellation_Count", "Cancellation_Count_reduce"),
        ("No_Show_Left_Count", "No_Show_Left_Count_reduce"),
        ("Rescheduled_Appointments_Count", "Rescheduled_Appointments_Count_reduce"),
        ("Patient Choice", "Patient_Choice_reduce"),
        ("Provider Choice", "Provider_Choice_reduce"),
        ("Clinic Management", "Clinic_Management_reduce"),
        ("Socioeconomic", "Socioeconomic_reduce")
    ]:
        if var in X_test_cf_stage1.columns:
            X_test_cf_stage1[var] = (
                X_test_cf_stage1[var] * (1 - percent_changes_stage1[key] / 100)
            ).round().astype(int)

    # Change Weekend_Indicator
    weekend_indices = X_test_cf_stage1.index[X_test_cf_stage1["Weekend_Indicator_1"] == 1].tolist()
    n_to_change = int(len(weekend_indices) * percent_changes_stage1["Weekend_Indicator_change"] / 100)
    np.random.seed(42)
    if n_to_change > 0:
        indices_to_weekday = np.random.choice(weekend_indices, size=n_to_change, replace=False)
        X_test_cf_stage1.loc[indices_to_weekday, "Weekend_Indicator_1"] = 0
        X_test_cf_stage1.loc[indices_to_weekday, "Weekend_Indicator_0"] = 1

    # Change Encounter_Hour
    indices_before_12 = X_test_cf_stage1.index[X_test_cf_stage1["Encounter_Hour_before 12pm"] == 1].tolist()
    n_change = int(len(indices_before_12) * percent_changes_stage1["Encounter_Hour_change"] / 100)
    if n_change > 0:
        indices_to_afternoon = np.random.choice(indices_before_12, size=n_change, replace=False)
        X_test_cf_stage1.loc[indices_to_afternoon, "Encounter_Hour_before 12pm"] = 0
        X_test_cf_stage1.loc[indices_to_afternoon, "Encounter_Hour_between 12-18pm"] = 1
        X_test_cf_stage1.loc[indices_to_afternoon, "Encounter_Hour_after 18pm"] = 0

    # Predictions
    cf_X_test_stage1 = X_test_cf_stage1[final_features].values
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage1, threshold=stage1_threshold)

    orig_cancellation_rate = np.mean(stage1_preds_orig == 1)
    cf_cancellation_rate = np.mean(stage1_preds_cf == 1)
    cancel_reduction = (orig_cancellation_rate - cf_cancellation_rate) * 100

    print("\n========== STAGE 1 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original cancellation rate:         {orig_cancellation_rate*100:.2f}%")
    print(f"Counterfactual cancellation rate:   {cf_cancellation_rate*100:.2f}%")
    print(f"Absolute reduction in cancellations: {cancel_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_sched_stage1.append([seed, cancel_reduction])

    # ======================================================
    # Stage 2 Counterfactual (No-shows Only)
    # ======================================================
    print("\n--- Counterfactual Analysis: Stage 2 (No-shows Only) ---")
    X_test_cf_stage2 = X_test_original_df.copy()

    percent_changes_stage2 = percent_changes_stage1.copy()  # same interventions

    if "Avg_Schd_Encounter_Diff_Hours" in X_test_cf_stage2.columns:
        X_test_cf_stage2["Avg_Schd_Encounter_Diff_Hours"] = X_test_cf_stage2["Avg_Schd_Encounter_Diff_Hours"].apply(
            lambda x: min(x, percent_changes_stage2["Avg_Schd_Encounter_Diff_Hours_cap"]) if pd.notnull(x) else x
        )

    for var, key in [
        ("Cancellation_Count", "Cancellation_Count_reduce"),
        ("No_Show_Left_Count", "No_Show_Left_Count_reduce"),
        ("Rescheduled_Appointments_Count", "Rescheduled_Appointments_Count_reduce"),
        ("Patient Choice", "Patient_Choice_reduce"),
        ("Provider Choice", "Provider_Choice_reduce"),
        ("Clinic Management", "Clinic_Management_reduce"),
        ("Socioeconomic", "Socioeconomic_reduce")
    ]:
        if var in X_test_cf_stage2.columns:
            X_test_cf_stage2[var] = (
                X_test_cf_stage2[var] * (1 - percent_changes_stage2[key] / 100)
            ).round().astype(int)

    weekend_indices = X_test_cf_stage2.index[X_test_cf_stage2["Weekend_Indicator_1"] == 1].tolist()
    n_to_change = int(len(weekend_indices) * percent_changes_stage2["Weekend_Indicator_change"] / 100)
    np.random.seed(42)
    if n_to_change > 0:
        indices_to_weekday = np.random.choice(weekend_indices, size=n_to_change, replace=False)
        X_test_cf_stage2.loc[indices_to_weekday, "Weekend_Indicator_1"] = 0
        X_test_cf_stage2.loc[indices_to_weekday, "Weekend_Indicator_0"] = 1

    indices_before_12 = X_test_cf_stage2.index[X_test_cf_stage2["Encounter_Hour_before 12pm"] == 1].tolist()
    n_change = int(len(indices_before_12) * percent_changes_stage2["Encounter_Hour_change"] / 100)
    if n_change > 0:
        indices_to_afternoon = np.random.choice(indices_before_12, size=n_change, replace=False)
        X_test_cf_stage2.loc[indices_to_afternoon, "Encounter_Hour_before 12pm"] = 0
        X_test_cf_stage2.loc[indices_to_afternoon, "Encounter_Hour_between 12-18pm"] = 1
        X_test_cf_stage2.loc[indices_to_afternoon, "Encounter_Hour_after 18pm"] = 0

    cf_X_test_stage2 = X_test_cf_stage2[final_features].values
    stage1_preds_orig = stage_one_predict_only(stage1_model, orig_X_test, threshold=stage1_threshold)
    stage2_preds_orig, _ = stage_two_predict_only(stage2_model, orig_X_test, stage1_preds_orig, threshold=stage2_threshold)

    stage1_preds_cf = stage_one_predict_only(stage1_model, cf_X_test_stage2, threshold=stage1_threshold)
    stage2_preds_cf, _ = stage_two_predict_only(stage2_model, cf_X_test_stage2, stage1_preds_cf, threshold=stage2_threshold)

    orig_noshow_rate = np.mean(stage2_preds_orig == 0)
    cf_noshow_rate = np.mean(stage2_preds_cf == 0)
    noshow_reduction = (orig_noshow_rate - cf_noshow_rate) * 100

    print("\n========== STAGE 2 COUNTERFACTUAL SUMMARY ==========")
    print(f"Original no-show rate:              {orig_noshow_rate*100:.2f}%")
    print(f"Counterfactual no-show rate:        {cf_noshow_rate*100:.2f}%")
    print(f"Absolute reduction in no-shows:     {noshow_reduction:.2f} percentage points")
    print("=====================================================")

    all_results_sched_stage2.append([seed, noshow_reduction])

# ======================================================
# Final Summary Across Seeds
# ======================================================
df_sched_stage1 = pd.DataFrame(all_results_sched_stage1, columns=["Seed", "Δ Cancel (%)"])
df_sched_stage2 = pd.DataFrame(all_results_sched_stage2, columns=["Seed", "Δ No-show (%)"])

summary_stage1 = df_sched_stage1["Δ Cancel (%)"].agg(["mean", "std"])
summary_stage2 = df_sched_stage2["Δ No-show (%)"].agg(["mean", "std"])

print("\n================ FINAL SUMMARY (Mean ± SD across 10 models) =================")
print(f"Stage 1 (Cancel, Scheduling Interventions): {summary_stage1['mean']:.2f} ± {summary_stage1['std']:.2f} pp")
print(f"Stage 2 (No-show, Scheduling Interventions): {summary_stage2['mean']:.2f} ± {summary_stage2['std']:.2f} pp")
print("============================================================================")

# =========================================
# Range of variables for extreme (Table S10 helper)
# =========================================
# Variables you included in extreme_values + special cases
all_vars = list(extreme_values.keys()) + ["Encounter_Hour", "Weekend_Indicator"]

max_values = {}

for var in extreme_values.keys():
    if var in X_test_original_df.columns:
        series = X_test_original_df[var]
        max_values[var] = series.max()
    else:
        max_values[var] = "N/A"

# Handle Encounter_Hour (dummy vars collapsed into categories)
max_values["Encounter_Hour"] = "before 12pm / 12–18pm / after 18pm"

# Handle Weekend_Indicator (dummy vars collapsed into categories)
max_values["Weekend_Indicator"] = "Weekday / Weekend"

# Print nicely
print("\n=== Variable Maximums for Table S10 ===")
for var, val in max_values.items():
    print(f"{var}: {val}")

# =========================================
# END OF SCRIPT
# =========================================
