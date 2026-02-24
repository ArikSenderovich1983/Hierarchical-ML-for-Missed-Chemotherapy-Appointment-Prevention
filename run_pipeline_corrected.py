"""
Corrected Pipeline for HCMS Hierarchical ML Paper
==================================================
Reproduces all tables from revised tables_all cutoffs.tex with the corrected
counterfactual analysis (Table 5 bug fix: no-show rates use total population denominator).

Usage:
  Set DATA_DIR to your data path (containing df_variables_0717.csv and df_with_valid_indices_0717.csv)
  python run_pipeline_corrected.py

Output:
  - revised_tables_corrected.tex
  - Console summary of results
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

# Try keras for to_categorical; fallback to manual one-hot
try:
    from keras.utils import to_categorical
except ImportError:
    def to_categorical(y, num_classes=None):
        y = np.asarray(y, dtype=int)
        if num_classes is None:
            num_classes = np.max(y) + 1
        return np.eye(num_classes)[y]

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = os.environ.get("HCMS_DATA_DIR", ".")
CUTOFFS_H = [24, 48, 72, 96, 168]  # No-show cutoff definitions (hours)
N_SEEDS = 10
STAGE1_THRESHOLD = 0.43
STAGE2_THRESHOLD = 0.83

df1_path = os.path.join(DATA_DIR, "df_variables_0717.csv")
df2_path = os.path.join(DATA_DIR, "df_with_valid_indices_0717.csv")

# =============================================================================
# LABEL BUILDING (cutoff-aware)
# =============================================================================
def build_primary_labels(df: pd.DataFrame, cutoff_hours: int) -> np.ndarray:
    """Build primary_head_label for given no-show cutoff.
    0=Show, 1=No-show (incl. cancel within cutoff), 2=Cancel (before cutoff).
    """
    if "time_difference_hours" not in df.columns:
        df = df.copy()
        df["ENCOUNTER_DTTM"] = pd.to_datetime(df["ENCOUNTER_DTTM"], errors="coerce")
        df["CNCL_DTTM"] = pd.to_datetime(df["CNCL_DTTM"], errors="coerce")
        df["time_difference_hours"] = (
            (df["ENCOUNTER_DTTM"] - df["CNCL_DTTM"]).dt.total_seconds() / 3600
        )
    cancelled_within = (
        (df["STATUS_CD"] == "Canceled")
        & (df["time_difference_hours"] >= 0)
        & (df["time_difference_hours"] <= cutoff_hours)
    ).astype(int)
    lbl = np.full(len(df), np.nan)
    lbl[df["STATUS_CD"] == "Completed"] = 0
    lbl[(df["STATUS_CD"].isin(["No Show", "Left without seen"])) | (cancelled_within == 1)] = 1
    lbl[(df["STATUS_CD"] == "Canceled") & (cancelled_within == 0)] = 2
    return lbl, cancelled_within


# =============================================================================
# HIERARCHICAL MODEL HELPERS
# =============================================================================
def apply_threshold(y_scores, threshold):
    return (y_scores >= threshold).astype(int)


def stage_one_predict_only(model, X_test, threshold=0.43):
    y_proba = model.predict_proba(X_test)[:, 1]
    return apply_threshold(y_proba, threshold)


def stage_two_predict_only(model, X_test, stage1_pred, threshold=0.83):
    idx = np.where(stage1_pred == 0)[0]
    X_sub = X_test[idx]
    y_proba = model.predict_proba(X_sub)[:, 1]
    return apply_threshold(y_proba, threshold), X_sub, idx


def compute_noshow_rate_fixed(stage2_pred, n_total):
    """
    CORRECTED: No-show rate as fraction of TOTAL population (not Stage 2 subset).
    stage2_pred = predictions for rows where Stage 1 predicted Not-Cancel.
    Count of predicted no-shows / total test size.
    """
    n_noshow = np.sum(stage2_pred == 0)
    return n_noshow / n_total if n_total > 0 else 0.0


# =============================================================================
# LOAD & PREPARE DATA
# =============================================================================
def load_and_prepare_data():
    if not os.path.exists(df1_path) or not os.path.exists(df2_path):
        raise FileNotFoundError(
            f"Data files not found. Set HCMS_DATA_DIR or place df_variables_0717.csv "
            f"and df_with_valid_indices_0717.csv in {DATA_DIR}"
        )
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df2_subset = df2[["PT_ID", "ENCOUNTER_DTTM", "CLIN_DEPT_NM", "FLOOR_LOCATION_NM"]].drop_duplicates(
        subset=["PT_ID", "ENCOUNTER_DTTM"]
    )
    df = pd.merge(df1, df2_subset, on=["PT_ID", "ENCOUNTER_DTTM"], how="left")
    return df


def prepare_features_and_labels(df, cutoff_hours):
    """Build features and labels for a given cutoff. Returns (X, y, final_features, df_processed)."""
    df = df.copy()
    df["ENCOUNTER_DTTM"] = pd.to_datetime(df["ENCOUNTER_DTTM"], errors="coerce")
    df["SCHD_DTTM"] = pd.to_datetime(df["SCHD_DTTM"], errors="coerce")
    df["CNCL_DTTM"] = pd.to_datetime(df["CNCL_DTTM"], errors="coerce")
    df["time_difference_hours"] = (
        (df["ENCOUNTER_DTTM"] - df["CNCL_DTTM"]).dt.total_seconds() / 3600
    )
    cancelled_within = (
        (df["STATUS_CD"] == "Canceled")
        & (df["time_difference_hours"] >= 0)
        & (df["time_difference_hours"] <= cutoff_hours)
    ).astype(int)
    df["cancelled_within_Xh"] = cancelled_within

    lbl = np.full(len(df), np.nan)
    lbl[df["STATUS_CD"] == "Completed"] = 0
    lbl[(df["STATUS_CD"].isin(["No Show", "Left without seen"])) | (cancelled_within == 1)] = 1
    lbl[(df["STATUS_CD"] == "Canceled") & (cancelled_within == 0)] = 2
    df["primary_head_label"] = lbl
    df = df.dropna(subset=["primary_head_label"])
    df["primary_head_label"] = df["primary_head_label"].astype(int)

    for col in ["ENCOUNTER_DTTM", "SCHD_DTTM"]:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_hour"] = df[col].dt.hour

    categorical_cols = ["First_Follow_Up", "Encounter_Month", "Encounter_Weekday", "Last_Appointment_Status", "Encounter_Hour", "Weekend_Indicator"]
    for c in categorical_cols:
        if c in df.columns:
            df = pd.get_dummies(df, columns=[c], prefix=c, drop_first=False)
    for c in ["VISIT_TYPE", "ENCOUNTER_TYPE", "CLIN_DEPT_NM", "FLOOR_LOCATION_NM"]:
        if c in df.columns:
            df = pd.get_dummies(df, columns=[c], prefix=c, drop_first=False)

    time_features = [
        "ENCOUNTER_DTTM_year", "ENCOUNTER_DTTM_month", "ENCOUNTER_DTTM_day",
        "ENCOUNTER_DTTM_weekday", "ENCOUNTER_DTTM_hour",
        "SCHD_DTTM_year", "SCHD_DTTM_month", "SCHD_DTTM_day",
        "SCHD_DTTM_weekday", "SCHD_DTTM_hour",
    ]
    numerical_features = [
        "Cancellation_Count", "No_Show_Left_Count", "Provider_Change_Count", "Department_Change_Count",
        "Last_Visit_Duration", "Avg_STD_DURATION", "Rescheduled_Appointments_Count",
        "Avg_Encounter_Frequency_Hours", "Avg_Schd_Encounter_Diff_Hours", "Average_Arrival_Lag_Hours",
        "Time_Since_Last_Visit",
        "Patient Choice", "Clinic Management", "Patient Health", "Socioeconomic", "Provider Choice",
    ]
    binary_features = ["Department_Changed_Last_Visit", "Provider_Changed_Last_Visit"]
    prefixes = ["VISIT_TYPE_", "ENCOUNTER_TYPE_", "CLIN_DEPT_NM_", "FLOOR_LOCATION_NM_",
                "First_Follow_Up_", "Last_Appointment_Status_", "Encounter_Hour_", "Weekend_Indicator_"]
    one_hot = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    final_features = [f for f in (time_features + numerical_features + one_hot + binary_features) if f in df.columns]

    X = df[final_features].fillna(0)
    y = df["primary_head_label"].values
    return X, y, final_features, numerical_features, df


# =============================================================================
# TRAIN HIERARCHICAL MODELS (10 seeds)
# =============================================================================
def train_models(X_train, X_test, y_train, y_test, final_features, numerical_features):
    scaler = StandardScaler()
    X_train_df = pd.DataFrame(X_train, columns=final_features)
    X_test_df = pd.DataFrame(X_test, columns=final_features)
    for f in numerical_features:
        if f in X_train_df.columns:
            X_train_df[f] = scaler.fit_transform(X_train_df[[f]])
            X_test_df[f] = scaler.transform(X_test_df[[f]])
    X_train = np.array(X_train_df)
    X_test = np.array(X_test_df)

    models = {}
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        dt = DecisionTreeClassifier(random_state=seed)
        y_tr_bin = (y_train == 2).astype(int)
        dt.fit(X_train, y_tr_bin)
        xgb = XGBClassifier(eval_metric="logloss", random_state=seed, nthread=1, use_label_encoder=False, verbosity=0)
        idx_tr = np.where(y_train != 2)[0]
        X_tr_s2 = X_train[idx_tr]
        y_tr_s2 = (y_train[idx_tr] == 0).astype(int)
        xgb.fit(X_tr_s2, y_tr_s2)
        models[seed] = {"stage1": dt, "stage2": xgb}
    return models, X_test_df, scaler


# =============================================================================
# COUNTERFACTUAL INTERVENTIONS (with fixed no-show rate)
# =============================================================================
def run_counterfactuals(models, X_test_df, final_features, stage1_thr, stage2_thr):
    results = {
        "SMS Reminders": {"cancel": [], "noshow": []},
        "Patient Confirmation": {"cancel": [], "noshow": []},
        "Reduced Lead Time": {"cancel": [], "noshow": []},
        "Provider Consistency": {"cancel": [], "noshow": []},
        "Commitment-Based": {"cancel": [], "noshow": []},
    }
    n_total = len(X_test_df)

    for seed in range(N_SEEDS):
        s1, s2 = models[seed]["stage1"], models[seed]["stage2"]
        X_orig = X_test_df[final_features].values

        # --- SMS Reminders (Literature-based) ---
        X_cf = X_test_df.copy()
        X_cf["Cancellation_Count"] = (X_test_df["Cancellation_Count"] * 1.12).round().astype(int)
        X_cf["No_Show_Left_Count"] = (X_test_df["No_Show_Left_Count"] * 0.91).round().astype(int)
        X_cf["Rescheduled_Appointments_Count"] = (X_test_df["Rescheduled_Appointments_Count"] * 1.15).round().astype(int)
        X_cf["Average_Arrival_Lag_Hours"] = X_test_df["Average_Arrival_Lag_Hours"] * 0.9
        X_cf["Patient Choice"] = (X_test_df["Patient Choice"] * 0.93).round().astype(int).clip(lower=0)
        cf = X_cf[final_features].values
        p1_o = stage_one_predict_only(s1, X_orig, stage1_thr)
        p1_c = stage_one_predict_only(s1, cf, stage1_thr)
        p2_o, _, idx_o = stage_two_predict_only(s2, X_orig, p1_o, stage2_thr)
        p2_c, _, _ = stage_two_predict_only(s2, cf, p1_c, stage2_thr)
        dc = (np.mean(p1_o == 1) - np.mean(p1_c == 1)) * 100
        dn = (compute_noshow_rate_fixed(p2_o, n_total) - compute_noshow_rate_fixed(p2_c, n_total)) * 100
        results["SMS Reminders"]["cancel"].append(dc)
        results["SMS Reminders"]["noshow"].append(dn)

        # --- Patient Confirmation (doubled) ---
        X_cf = X_test_df.copy()
        X_cf["Cancellation_Count"] = (X_test_df["Cancellation_Count"] * 1.24).round().astype(int)
        X_cf["No_Show_Left_Count"] = (X_test_df["No_Show_Left_Count"] * 0.82).round().astype(int)
        X_cf["Rescheduled_Appointments_Count"] = (X_test_df["Rescheduled_Appointments_Count"] * 1.30).round().astype(int)
        X_cf["Average_Arrival_Lag_Hours"] = X_test_df["Average_Arrival_Lag_Hours"] * 0.80
        X_cf["Patient Choice"] = (X_test_df["Patient Choice"] * 0.86).round().astype(int).clip(lower=0)
        cf = X_cf[final_features].values
        p1_c = stage_one_predict_only(s1, cf, stage1_thr)
        p2_c, _, _ = stage_two_predict_only(s2, cf, p1_c, stage2_thr)
        dc = (np.mean(p1_o == 1) - np.mean(p1_c == 1)) * 100
        dn = (compute_noshow_rate_fixed(p2_o, n_total) - compute_noshow_rate_fixed(p2_c, n_total)) * 100
        results["Patient Confirmation"]["cancel"].append(dc)
        results["Patient Confirmation"]["noshow"].append(dn)

        # --- Reduced Lead Time (cap Avg_Schd_Encounter_Diff) ---
        X_cf = X_test_df.copy()
        if "Avg_Schd_Encounter_Diff_Hours" in X_cf.columns:
            X_cf["Avg_Schd_Encounter_Diff_Hours"] = X_cf["Avg_Schd_Encounter_Diff_Hours"].clip(upper=168)
        for col in ["Cancellation_Count", "No_Show_Left_Count", "Rescheduled_Appointments_Count", "Patient Choice", "Provider Choice", "Clinic Management", "Socioeconomic"]:
            if col in X_cf.columns:
                X_cf[col] = (X_cf[col] * 0.5).round().astype(int)
        cf = X_cf[final_features].values
        p1_c = stage_one_predict_only(s1, cf, stage1_thr)
        p2_c, _, _ = stage_two_predict_only(s2, cf, p1_c, stage2_thr)
        dc = (np.mean(p1_o == 1) - np.mean(p1_c == 1)) * 100
        dn = (compute_noshow_rate_fixed(p2_o, n_total) - compute_noshow_rate_fixed(p2_c, n_total)) * 100
        results["Reduced Lead Time"]["cancel"].append(dc)
        results["Reduced Lead Time"]["noshow"].append(dn)

        # --- Provider Consistency ---
        X_cf = X_test_df.copy()
        for col, cap in [("Cancellation_Count", 9), ("Rescheduled_Appointments_Count", 9), ("Patient Choice", 9), ("Provider_Change_Count", 6), ("Department_Change_Count", 6)]:
            if col in X_cf.columns:
                X_cf[col] = X_cf[col].clip(upper=cap)
        if "Provider_Changed_Last_Visit" in X_cf.columns:
            X_cf["Provider_Changed_Last_Visit"] = 0
        if "Department_Changed_Last_Visit" in X_cf.columns:
            X_cf["Department_Changed_Last_Visit"] = 0
        cf = X_cf[final_features].values
        p1_c = stage_one_predict_only(s1, cf, stage1_thr)
        p2_c, _, _ = stage_two_predict_only(s2, cf, p1_c, stage2_thr)
        dc = (np.mean(p1_o == 1) - np.mean(p1_c == 1)) * 100
        dn = (compute_noshow_rate_fixed(p2_o, n_total) - compute_noshow_rate_fixed(p2_c, n_total)) * 100
        results["Provider Consistency"]["cancel"].append(dc)
        results["Provider Consistency"]["noshow"].append(dn)

        # --- Commitment-Based (CSS-BP) ---
        X_cf = X_test_df.copy()
        X_cf["No_Show_Left_Count"] = (X_test_df["No_Show_Left_Count"] * 0.8).round().astype(int)
        X_cf["Average_Arrival_Lag_Hours"] = X_test_df["Average_Arrival_Lag_Hours"] * 0.8
        if "Socioeconomic" in X_cf.columns:
            X_cf["Socioeconomic"] = (X_test_df["Socioeconomic"] * 0.8).round().astype(int)
        cf = X_cf[final_features].values
        p1_c = stage_one_predict_only(s1, cf, stage1_thr)
        p2_c, _, _ = stage_two_predict_only(s2, cf, p1_c, stage2_thr)
        dc = (np.mean(p1_o == 1) - np.mean(p1_c == 1)) * 100
        dn = (compute_noshow_rate_fixed(p2_o, n_total) - compute_noshow_rate_fixed(p2_c, n_total)) * 100
        results["Commitment-Based"]["cancel"].append(dc)
        results["Commitment-Based"]["noshow"].append(dn)

    return results


# =============================================================================
# MAIN: RUN FOR EACH CUTOFF & COLLECT RESULTS
# =============================================================================
def main():
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} rows")

    all_table1 = {}
    all_table5 = {}

    for cutoff in CUTOFFS_H:
        print(f"\n{'='*60}\nProcessing cutoff {cutoff}h\n{'='*60}")
        X, y, final_features, numerical_features, _ = prepare_features_and_labels(df, cutoff)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        models, X_test_df, _ = train_models(
            X_train.values, X_test.values, y_train, y_test,
            final_features, numerical_features
        )

        # Table 1: Pipeline metrics (single seed for brevity; use seed 0)
        s1, s2 = models[0]["stage1"], models[0]["stage2"]
        X_te = X_test_df[final_features].values
        p1 = stage_one_predict_only(s1, X_te, STAGE1_THRESHOLD)
        p2, _, _ = stage_two_predict_only(s2, X_te, p1, STAGE2_THRESHOLD)
        y_te_bin_s1 = (y_test == 2).astype(int)
        idx_s2 = np.where(p1 == 0)[0]
        y_te_s2 = y_test[idx_s2]
        y_te_bin_s2 = (y_te_s2 == 0).astype(int)
        all_table1[cutoff] = {
            "stage1": {
                "prec_cancel": precision_score(y_te_bin_s1, p1, pos_label=1, zero_division=0),
                "rec_cancel": recall_score(y_te_bin_s1, p1, pos_label=1, zero_division=0),
                "f1_cancel": f1_score(y_te_bin_s1, p1, pos_label=1, zero_division=0),
                "support_cancel": int(np.sum(y_te_bin_s1 == 1)),
                "support_notcancel": int(np.sum(y_te_bin_s1 == 0)),
            },
            "stage2": {
                "prec_noshow": precision_score(y_te_bin_s2, p2, pos_label=0, zero_division=0),
                "rec_noshow": recall_score(y_te_bin_s2, p2, pos_label=0, zero_division=0),
                "f1_noshow": f1_score(y_te_bin_s2, p2, pos_label=0, zero_division=0),
                "support_noshow": int(np.sum(y_te_bin_s2 == 0)),
                "support_show": int(np.sum(y_te_bin_s2 == 1)),
            },
        }

        # Table 5: Counterfactuals (corrected)
        cf_results = run_counterfactuals(models, X_test_df, final_features, STAGE1_THRESHOLD, STAGE2_THRESHOLD)
        all_table5[cutoff] = cf_results

    # =========================================================================
    # WRITE REVISED TABLES (CORRECTED)
    # =========================================================================
    out_path = os.path.join(os.path.dirname(__file__) or ".", "revised_tables_corrected.tex")
    with open(out_path, "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{tabularx}" + "\n")
        f.write(r"\usepackage{multirow}" + "\n")
        f.write(r"\usepackage{geometry}" + "\n")
        f.write(r"\geometry{margin=1in}" + "\n")
        f.write(r"\begin{document}" + "\n\n")

        # Table 1
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Performance of the two-stage pipeline (DT $\rightarrow$ XGBoost) across no-show cutoff definitions. \textbf{CORRECTED} pipeline.}" + "\n")
        f.write(r"\label{tab:multi_cutoff_corrected}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{l c c c c c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Metric} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{6}{l}{\textbf{Stage 1: Decision Tree (Cancel vs Not-Cancel)}} \\" + "\n")
        for m, k in [("Precision (Cancel)", "prec_cancel"), ("Recall (Cancel)", "rec_cancel"), ("F1 (Cancel)", "f1_cancel")]:
            row = " & ".join([f"{all_table1[c]['stage1'][k]:.2f}" for c in CUTOFFS_H])
            f.write(f"{m} & {row} \\\\\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{6}{l}{\textbf{Stage 2: XGBoost (Show vs No-Show)}} \\" + "\n")
        for m, k in [("Precision (No-Show)", "prec_noshow"), ("Recall (No-Show)", "rec_noshow"), ("F1 (No-Show)", "f1_noshow")]:
            row = " & ".join([f"{all_table1[c]['stage2'][k]:.2f}" for c in CUTOFFS_H])
            f.write(f"{m} & {row} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n\n")

        # Table 5 (Counterfactual - CORRECTED)
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Counterfactual Analysis Sensitivity (CORRECTED): Absolute change in cancellation and no-show rates. \textbf{Bug fix:} No-show rates use total population denominator.}" + "\n")
        f.write(r"\label{tab:counterfactual_corrected}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabularx}{\textwidth}{l l c c c c c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\multirow{2}{*}{\textbf{Intervention}} & \multirow{2}{*}{\textbf{Metric}} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\" + "\n")
        f.write(r"\cmidrule(lr){3-7}" + "\n")
        f.write(r"& & \multicolumn{5}{c}{\textit{(Mean $\\pm$ SD across 10 seeds)}} \\" + "\n")
        f.write(r"\midrule" + "\n")
        for interv in ["SMS Reminders", "Patient Confirmation", "Reduced Lead Time", "Provider Consistency", "Commitment-Based"]:
            dc = [np.mean(all_table5[c][interv]["cancel"]) for c in CUTOFFS_H]
            dc_std = [np.std(all_table5[c][interv]["cancel"]) for c in CUTOFFS_H]
            dn = [np.mean(all_table5[c][interv]["noshow"]) for c in CUTOFFS_H]
            dn_std = [np.std(all_table5[c][interv]["noshow"]) for c in CUTOFFS_H]
            f.write(r"\multirow{2}{*}{" + interv + r"} " + "\n")
            row_c = " & ".join([f"${dc[i]:.2f} \\pm {dc_std[i]:.2f}$" for i in range(5)])
            row_n = " & ".join([f"${dn[i]:.2f} \\pm {dn_std[i]:.2f}$" for i in range(5)])
            f.write(f"& $\\Delta$ Cancel (\\%) & {row_c} \\\\\n")
            f.write(f"& $\\Delta$ No-show (\\%) & {row_n} \\\\\n")
            if interv != "Commitment-Based":
                f.write(r"\midrule" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabularx}" + "\n")
        f.write(r"\end{table*}" + "\n\n")
        f.write(r"\end{document}" + "\n")

    print(f"\nOutput written to {out_path}")
    return all_table1, all_table5


if __name__ == "__main__":
    main()
