# =============================================================================
#                          TABLE OF CONTENTS
# =============================================================================
#  0) Imports & Setup
#  1) Load Data
#  2) Labels (Primary / Masks)
#  3) Cancellation / No-show Reason Mapping
#  4) Datetime Expansion & Categorical Encoding
#  5) Feature Assembly
#  6) Reason Labels (Argmax over Dummies)
#  7) Row-wise Train/Test Split + Standardization
#  8) Models — One-stage Decision Tree
#  9) Models — Two-stage Decision Tree
# 10) Models — One-stage XGBoost
# 11) Models — Two-stage XGBoost
# 12) Models — One-stage MLP
# 13) Models — Two-stage MLP
# 14) Two-stage DT Variants (9 Permutations)
# 15) 3×3 Two-stage Grid (DT/XGB/MLP)
# 16) Threshold Tuning (Precision-Recall Curve)
# 17) Patient-level Split + Standardization
# 18) Final Model (DT → XGB) with Fixed Thresholds
# 19) Per-patient Sequence Split + Standardization
# =============================================================================



# =========================
# 0) IMPORTS & SETUP
# =========================
import os
import glob
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Visualization (optional — used sparsely)
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.utils import resample

# Classical ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# TensorFlow/Keras (for to_categorical)
from keras.utils import to_categorical

# XGBoost (used in later blocks; install if needed)
# from xgboost import XGBClassifier  # Imported later where used

# =========================
# 0.1) ENVIRONMENT / COLAB
# =========================
# If running in Google Colab and reading from Google Drive, keep this:
try:
    from google.colab import drive
    _RUNNING_IN_COLAB = True
except Exception:
    _RUNNING_IN_COLAB = False

if _RUNNING_IN_COLAB:
    # Mount Drive to access CSVs
    drive.mount('/content/drive')

# =========================
# 1) LOAD DATA
# =========================
# Update paths if needed
df1_path = '/content/drive/My Drive/Merged/df_variables_0717.csv'
df2_path = '/content/drive/My Drive/Merged/df_with_valid_indices_0717.csv'

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

print("Rows in df1:", df1.shape[0])
print("Rows in df2:", df2.shape[0])

# Keep unique encounter rows from df2 on (PT_ID, ENCOUNTER_DTTM) and a few extra columns
df2_subset = df2[['PT_ID', 'ENCOUNTER_DTTM', 'CLIN_DEPT_NM', 'FLOOR_LOCATION_NM']].drop_duplicates(
    subset=['PT_ID', 'ENCOUNTER_DTTM']
)

# Left-join to preserve df1 rows
df = pd.merge(df1, df2_subset, on=['PT_ID', 'ENCOUNTER_DTTM'], how='left')
print("Rows after merge (should match df1):", df.shape[0])

print(f"Shape of merged DataFrame: {df.shape}")
# Peek columns with one example value
for column in df.columns:
    try:
        print(f"{column}: {df[column].iloc[0]}")
    except Exception:
        print(f"{column}: <unavailable>")

# Basic cohort counts
total_data_points = len(df)
unique_patients = df['PT_ID'].nunique()
print(f"Total data points: {total_data_points}")
print(f"Unique patients: {unique_patients}")

# =========================
# 2) LABELS (PRIMARY / MASKS)
# =========================
def build_primary_head_label_vectorized(df_in: pd.DataFrame) -> np.ndarray:
    """
    Create primary_head_label:
      0 = Completed/Show
      1 = No Show / Left without seen / Cancellation within 24h
      2 = Canceled (before 24h)
    """
    lbl = np.full(df_in.shape[0], pd.NA)

    # 0: Completed
    lbl[df_in['STATUS_CD'] == 'Completed'] = 0

    # 1: No-shows or canceled within 24h
    lbl[
        (df_in['STATUS_CD'].isin(['No Show', 'Left without seen'])) |
        (df_in['cancelled_within_24h'] == 1)
    ] = 1

    # 2: Canceled before 24h
    lbl[
        (df_in['STATUS_CD'] == 'Canceled') &
        (df_in['cancelled_within_24h'] == 0)
    ] = 2

    return lbl

def build_third_head_mask_vectorized(df_in: pd.DataFrame) -> np.ndarray:
    """Binary mask for no-show/left-without-seen."""
    return df_in['STATUS_CD'].isin(['No Show', 'Left without seen']).astype(int)

# Apply labels
df['primary_head_label'] = build_primary_head_label_vectorized(df)
df['third_head_mask'] = build_third_head_mask_vectorized(df)

# Drop rows with missing primary label
nan_count = df['primary_head_label'].isna().sum()
print(f"Rows with NaN in 'primary_head_label': {nan_count}")
df = df.dropna(subset=['primary_head_label'])
print(f"Rows after dropping NaN primary labels: {df.shape[0]}")

# One-hot of primary label (0/1/2)
one_hot_primary = to_categorical(df['primary_head_label'].astype(int), num_classes=3)
one_hot_primary_df = pd.DataFrame(one_hot_primary, columns=['Completed', 'No Show', 'Canceled'], index=df.index)
df = pd.concat([df, one_hot_primary_df], axis=1)

print(df[['STATUS_CD', 'cancelled_within_24h', 'primary_head_label',
          'Completed', 'No Show', 'Canceled', 'third_head_mask']].head())

# =========================
# 3) CANCELLATION / NOSHOW REASON MAPPING → DUMMIES
# =========================
# High-level category mapping
reason_category_mapping = {
    'Patient Choice': [
        'Canceled via automated reminder system', 'Cancelled via Interface', 'Deleted via Interface',
        'Moved', 'Patient', 'Patient - Personal', 'Personal Reasons', 'Patient - Sought Care Elsewhere',
        'Unhappy/Changed Provider','Sought Care Elsewhere'
    ],
    'Clinic Management': [
        'Changed by Radiology', 'Discharged', 'Displaced Appointment', 'Edu/Meeting', 'Error',
        'Institution', 'Order Discontinued', 'Prep/Med/Results Unavailable', 'Schedule Order Error',
        'Scheduled from Wait List', 'Institution - Appt Made in Error',
        'Cancelled via automated reminder system', 'Level of Care Change',
        'Institution - Condition Warrants Cancellation'
    ],
    'Patient Health': [
        'Clinically Caused', 'Deceased', 'Feeling Better', 'Hospitalized',
        'Labs Out of Acceptable Range', 'Level of Care Change',
        'Oncology Treatment Plan Changes', 'Patient Dismissed From Practice'
    ],
    'Socioeconomic': [
        'Financial', 'Lack of Transportation', 'Financial Concerns'
    ],
    'Provider Choice': [
        'MD Appointment', 'Provider', 'Provider - Personal', 'Provider - Professional', 'Provider Departure'
    ]
}

# Reverse lookup mapping
reason_to_category = {reason: cat for cat, reasons in reason_category_mapping.items() for reason in reasons}

def map_reason_to_category(reason: str) -> str:
    """Map raw cancellation/no-show reason to high-level category."""
    return reason_to_category.get(reason, 'Unknown')

# Label reasons
df['reason_category_label'] = df['CNCL_REASON_DESCR'].apply(map_reason_to_category)

# Flags for cancellation/no-show subsets
df['cancellation_flag'] = (df['STATUS_CD'] == 'Canceled') & (df['cancelled_within_24h'] == 0)
df['noshows_flag'] = (df['STATUS_CD'].isin(['No Show', 'Left without seen'])) | (df['cancelled_within_24h'] == 1)

# One-hot for reason categories (separately for cancel vs no-show)
df_cancellation = pd.get_dummies(
    df[df['cancellation_flag']], columns=['reason_category_label'], prefix='cancellation_reasons'
)
df_noshows = pd.get_dummies(
    df[df['noshows_flag']], columns=['reason_category_label'], prefix='noshows_reasons'
)

# Merge dummy columns back; fill missing with 0
cancel_cols = [c for c in df_cancellation.columns if c.startswith('cancellation_reasons_')]
noshow_cols = [c for c in df_noshows.columns if c.startswith('noshows_reasons_')]

df = pd.merge(df, df_cancellation[cancel_cols], left_index=True, right_index=True, how='left')
df = pd.merge(df, df_noshows[noshow_cols], left_index=True, right_index=True, how='left')
df[cancel_cols + noshow_cols] = df[cancel_cols + noshow_cols].fillna(0).astype(int)

print("Reason dummies added. Sample:")
print(df.head())
print(df.columns.tolist())

# =========================
# 4) DATETIME EXPANSION & CATEGORICAL OHE
# =========================
# Parse datetime columns & derive components
datetime_cols = ['ENCOUNTER_DTTM', 'SCHD_DTTM']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_weekday'] = df[col].dt.weekday
    df[f'{col}_hour'] = df[col].dt.hour

# Categorical sets (some may already exist in df1)
categorical_cols_set1 = ['VISIT_TYPE', 'ENCOUNTER_TYPE', 'CLIN_DEPT_NM', 'FLOOR_LOCATION_NM']
categorical_cols_set2 = ['First_Follow_Up', 'Encounter_Month', 'Encounter_Weekday',
                         'Last_Appointment_Status', 'Encounter_Hour', 'Weekend_Indicator']

# One-hot encode union (deduped)
all_categorical = list(set(categorical_cols_set1 + categorical_cols_set2))
df = pd.get_dummies(df, columns=[c for c in all_categorical if c in df.columns])

# Print datetime-derived columns
prefixes = ['ENCOUNTER_DTTM', 'SCHD_DTTM']
matching_columns = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
print(f"Columns starting with {prefixes}:")
for c in matching_columns:
    print(f" - {c}")
print(f"Total columns found: {len(matching_columns)}")

# Inspect one-hot groups per original categorical variable
from collections import defaultdict
encoded_columns = defaultdict(list)
all_cat_prefixes = categorical_cols_set1 + categorical_cols_set2
for prefix in all_cat_prefixes:
    encoded_columns[prefix] = [c for c in df.columns if c.startswith(prefix + "_")]
for cat_var, cols in encoded_columns.items():
    if len(cols) > 0:
        print(f"{cat_var}: {len(cols)} one-hot columns")
        # for col in cols: print("  -", col)

# =========================
# 5) FEATURE ASSEMBLY
# =========================
# 5.1 Time-derived features
time_features = [
    'ENCOUNTER_DTTM_year', 'ENCOUNTER_DTTM_month', 'ENCOUNTER_DTTM_day',
    'ENCOUNTER_DTTM_weekday', 'ENCOUNTER_DTTM_hour',
    'SCHD_DTTM_year', 'SCHD_DTTM_month', 'SCHD_DTTM_day',
    'SCHD_DTTM_weekday', 'SCHD_DTTM_hour'
]

# 5.2 Numerical features (ensure these exist in df1/df2)
numerical_features = [
    'Cancellation_Count', 'No_Show_Left_Count', 'Provider_Change_Count', 'Department_Change_Count',
    'Last_Visit_Duration', 'Avg_STD_DURATION', 'Rescheduled_Appointments_Count',
    'Avg_Encounter_Frequency_Hours', 'Avg_Schd_Encounter_Diff_Hours', 'Average_Arrival_Lag_Hours',
    'Time_Since_Last_Visit',
    # If these 5 are present as counts/indicators in your df1, keep them:
    'Patient Choice', 'Clinic Management', 'Patient Health', 'Socioeconomic', 'Provider Choice'
]

# 5.3 Binary features
binary_features = ['Department_Changed_Last_Visit', 'Provider_Changed_Last_Visit']

# 5.4 One-hot categorical feature prefixes to include
categorical_prefixes = [
    'VISIT_TYPE_', 'ENCOUNTER_TYPE_', 'CLIN_DEPT_NM_', 'FLOOR_LOCATION_NM_',
    'First_Follow_Up_', 'Last_Appointment_Status_', 'Encounter_Hour_', 'Weekend_Indicator_'
]
one_hot_features = [c for c in df.columns if any(c.startswith(p) for p in categorical_prefixes)]

# 5.5 Final feature list
final_features = time_features + numerical_features + one_hot_features + binary_features

# Feature matrix
X = df[final_features]
print("Final feature matrix shape:", X.shape)
print(X.head())

# =========================
# 6) REASON LABELS (ARGMAX OVER DUMMIES)
# =========================
cancellation_reason_cols = [c for c in df.columns if c.startswith('cancellation_reasons_')]
noshows_reason_cols = [c for c in df.columns if c.startswith('noshows_reasons_')]

# Argmax per row if any present; else NaN
df['cancellation_reason_label'] = df[cancellation_reason_cols].apply(
    lambda row: np.argmax(row.values) if row.sum() > 0 else np.nan, axis=1
)
df['noshow_reason_label'] = df[noshows_reason_cols].apply(
    lambda row: np.argmax(row.values) if row.sum() > 0 else np.nan, axis=1
)

print("Cancellation Reason Labels (sample):")
print(df[['cancellation_reason_label'] + cancellation_reason_cols].head(20))
print("\nNo-Show Reason Labels (sample):")
print(df[['noshow_reason_label'] + noshows_reason_cols].head())

# =========================
# 7) ROW-WISE TRAIN/TEST SPLIT + STANDARDIZATION
# =========================
# Extract base arrays for row-wise split
X_benchmark = df[final_features].values
PT_IDs = df['PT_ID'].values
y_primary_benchmark = df['primary_head_label'].values
y_cancellation_benchmark = df['cancellation_reason_label'].values
y_noshow_benchmark = df['noshow_reason_label'].values

# Filters for specific heads
cancellation_filter = (df['STATUS_CD'] == 'Canceled') & (df['cancelled_within_24h'] == 0)
X_cancellation_benchmark = X_benchmark[cancellation_filter]
y_cancellation_benchmark = y_cancellation_benchmark[cancellation_filter]
PT_IDs_cancellation = PT_IDs[cancellation_filter]

noshow_filter = (df['STATUS_CD'].isin(['No Show', 'Left without seen'])) | (
    (df['STATUS_CD'] == 'Canceled') & (df['cancelled_within_24h'] == 1)
)
X_noshow_benchmark = X_benchmark[noshow_filter]
y_noshow_benchmark = y_noshow_benchmark[noshow_filter]
PT_IDs_noshow = PT_IDs[noshow_filter]

# Row-wise split (primary)
X_train_primary_benchmark, X_test_primary_benchmark, y_train_primary_benchmark, y_test_primary_benchmark, train_PT_IDs_primary, test_PT_IDs_primary = train_test_split(
    X_benchmark, y_primary_benchmark, PT_IDs, test_size=0.2, random_state=42, stratify=y_primary_benchmark
)

# Row-wise split (cancellation head)
X_train_cancellation_benchmark, X_test_cancellation_benchmark, y_train_cancellation_benchmark, y_test_cancellation_benchmark, train_PT_IDs_cancellation, test_PT_IDs_cancellation = train_test_split(
    X_cancellation_benchmark, y_cancellation_benchmark, PT_IDs_cancellation, test_size=0.2, random_state=42, stratify=y_cancellation_benchmark
)

# (No-show head split left commented out in your original)
# X_train_noshow_benchmark, X_test_noshow_benchmark, y_train_noshow_benchmark, y_test_noshow_benchmark, train_PT_IDs_noshow, test_PT_IDs_noshow = train_test_split(
#     X_noshow_benchmark, y_noshow_benchmark, PT_IDs_noshow, test_size=0.2, random_state=42, stratify=y_noshow_benchmark
# )

# Standardize numerical features (row-wise split)
X_train_primary_benchmark = pd.DataFrame(X_train_primary_benchmark, columns=final_features)
X_test_primary_benchmark = pd.DataFrame(X_test_primary_benchmark, columns=final_features)

scaler_primary_row = StandardScaler()
if set(numerical_features).issubset(set(X_train_primary_benchmark.columns)):
    X_train_primary_benchmark[numerical_features] = scaler_primary_row.fit_transform(
        X_train_primary_benchmark[numerical_features]
    )
    X_test_primary_benchmark[numerical_features] = scaler_primary_row.transform(
        X_test_primary_benchmark[numerical_features]
    )

# =========================
# 8) MODELS — ONE-STAGE DECISION TREE
# =========================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_primary_benchmark, y_train_primary_benchmark)
y_pred_dt = dt_model.predict(X_test_primary_benchmark)

print("\n=== One-Stage Decision Tree (Multiclass) ===")
print("Accuracy:", accuracy_score(y_test_primary_benchmark, y_pred_dt))
print("Precision (macro):", precision_score(y_test_primary_benchmark, y_pred_dt, average='macro'))
print("Recall (macro):", recall_score(y_test_primary_benchmark, y_pred_dt, average='macro'))
print("F1 Score (macro):", f1_score(y_test_primary_benchmark, y_pred_dt, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_primary_benchmark, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test_primary_benchmark, y_pred_dt))

# =========================
# 9) MODELS — TWO-STAGE DECISION TREE
# =========================
def evaluate_model_binary(name, model, X_train, X_test, y_train, y_test):
    """
    Fit binary classifier and print metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Binary Model Performance:")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred

def stage_one_binary_classification(models, X_train, X_test, y_train, y_test):
    """
    Stage 1: predict Show (0) vs Not 0 (1/2).
    """
    results = {}
    for name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Stage 1: Binary Classification for {name} (0 vs. Not 0)")

        y_train_bin = (y_train != 0).astype(int)
        y_test_bin = (y_test != 0).astype(int)

        y_pred = evaluate_model_binary(name + " (Binary Stage 1)", model, X_train, X_test, y_train_bin, y_test_bin)
        results[name] = y_pred
    return results

def stage_two_binary_classification(models, X_train, X_test, y_train, y_test, stage_one_results):
    """
    Stage 2: predict No-show (1) vs Cancellation (2) on the subset predicted as Not 0 in Stage 1.
    """
    for name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Stage 2: Binary Classification for {name} (1 vs. 2)")

        not0_idx_test = np.where(stage_one_results[name] == 1)[0]
        not0_idx_train = np.where(y_train != 0)[0]

        # Subset
        if isinstance(X_train, pd.DataFrame):
            X_train_not0 = X_train.iloc[not0_idx_train]
            y_train_not0 = pd.Series(y_train).iloc[not0_idx_train]
            X_test_not0 = X_test.iloc[not0_idx_test]
            y_test_not0 = pd.Series(y_test).iloc[not0_idx_test]
        else:
            X_train_not0 = X_train[not0_idx_train]
            y_train_not0 = y_train[not0_idx_train]
            X_test_not0 = X_test[not0_idx_test]
            y_test_not0 = y_test[not0_idx_test]

        # Keep only labels 1/2
        y_train_not0 = pd.Series(y_train_not0)
        y_test_not0 = pd.Series(y_test_not0)
        mask_tr = y_train_not0.isin([1, 2])
        mask_te = y_test_not0.isin([1, 2])

        X_tr = X_train_not0[mask_tr]
        y_tr = (y_train_not0[mask_tr] == 1).astype(int)  # 1 = No-show, 0 = Cancellation
        X_te = X_test_not0[mask_te]
        y_te = (y_test_not0[mask_te] == 1).astype(int)

        evaluate_model_binary(name + " (Binary Stage 2)", model, X_tr, X_te, y_tr, y_te)

# Run two-stage with Decision Tree
print("\nStarting Two-Stage Decision Tree Classification...")
models_dt = {'Decision Tree': DecisionTreeClassifier(random_state=42)}
stage1_results_dt = stage_one_binary_classification(
    models_dt, X_train_primary_benchmark, X_test_primary_benchmark, y_train_primary_benchmark, y_test_primary_benchmark
)
stage_two_binary_classification(
    models_dt, X_train_primary_benchmark, X_test_primary_benchmark, y_train_primary_benchmark, y_test_primary_benchmark, stage1_results_dt
)
print("\nTwo-Stage DT Classification Complete.")

# =========================
# 10) MODELS — ONE-STAGE XGBOOST
# =========================
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_primary_benchmark, y_train_primary_benchmark)

y_pred_xgb = xgb_model.predict(X_test_primary_benchmark)
y_proba_xgb = xgb_model.predict_proba(X_test_primary_benchmark)

print("\n=== One-Stage XGBoost (Multiclass) ===")
print("Accuracy:", round(accuracy_score(y_test_primary_benchmark, y_pred_xgb), 4))
print("Precision (macro):", round(precision_score(y_test_primary_benchmark, y_pred_xgb, average='macro'), 4))
print("Recall (macro):", round(recall_score(y_test_primary_benchmark, y_pred_xgb, average='macro'), 4))
print("F1 Score (macro):", round(f1_score(y_test_primary_benchmark, y_pred_xgb, average='macro'), 4))

# Macro AUC (One-vs-Rest)
y_test_bin = label_binarize(y_test_primary_benchmark, classes=[0, 1, 2])
auc_macro = roc_auc_score(y_test_bin, y_proba_xgb, average='macro', multi_class='ovr')
print("AUC (macro):", round(auc_macro, 4))

print("\nClassification Report:\n", classification_report(
    y_test_primary_benchmark, y_pred_xgb, target_names=['Show', 'No-show', 'Cancellation']
))

# =========================
# 11) MODELS — TWO-STAGE XGBOOST
# =========================
def evaluate_model_binary_simple(name, model, X_train, X_test, y_train, y_test):
    """Train/eval binary model with accuracy + report."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred

# Stage 1: 0 vs Others
y_train_stage1 = (y_train_primary_benchmark != 0).astype(int)
y_test_stage1 = (y_test_primary_benchmark != 0).astype(int)

xgb_stage1 = XGBClassifier(eval_metric='logloss', random_state=42)
stage1_pred = evaluate_model_binary_simple(
    "Stage 1 XGBoost (0 vs Others)",
    xgb_stage1,
    X_train_primary_benchmark, X_test_primary_benchmark,
    y_train_stage1, y_test_stage1
)

# Stage 2: 1 vs 2 on predicted Not-0 subset
stage2_idx = np.where(stage1_pred == 1)[0]
X_test_stage2 = X_test_primary_benchmark[stage2_idx]
y_test_stage2 = np.array(y_test_primary_benchmark)[stage2_idx]

mask_test_12 = np.isin(y_test_stage2, [1, 2])
X_test_stage2 = X_test_stage2[mask_test_12]
y_test_stage2 = y_test_stage2[mask_test_12]

train_mask_12 = np.isin(y_train_primary_benchmark, [1, 2])
X_train_stage2 = X_train_primary_benchmark[train_mask_12]
y_train_stage2 = np.array(y_train_primary_benchmark)[train_mask_12]

y_train_binary = (y_train_stage2 == 1).astype(int)
y_test_binary = (y_test_stage2 == 1).astype(int)

xgb_stage2 = XGBClassifier(eval_metric='logloss', random_state=42)
evaluate_model_binary_simple(
    "Stage 2 XGBoost (1 vs 2)",
    xgb_stage2,
    X_train_stage2, X_test_stage2,
    y_train_binary, y_test_binary
)

# =========================
# 12) MODELS — ONE-STAGE MLP
# =========================
# Fill NaNs to avoid issues in MLP
X_train_filled = pd.DataFrame(X_train_primary_benchmark, columns=final_features).fillna(0)
X_test_filled = pd.DataFrame(X_test_primary_benchmark, columns=final_features).fillna(0)

mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)
mlp.fit(X_train_filled, y_train_primary_benchmark)

y_pred_mlp = mlp.predict(X_test_filled)
y_proba_mlp = mlp.predict_proba(X_test_filled)

print("\n=== One-Stage MLP (Multiclass) ===")
print("Accuracy:", round(accuracy_score(y_test_primary_benchmark, y_pred_mlp), 4))
print("Precision (macro):", round(precision_score(y_test_primary_benchmark, y_pred_mlp, average='macro'), 4))
print("Recall (macro):", round(recall_score(y_test_primary_benchmark, y_pred_mlp, average='macro'), 4))
print("F1 Score (macro):", round(f1_score(y_test_primary_benchmark, y_pred_mlp, average='macro'), 4))

y_test_bin_mlp = label_binarize(y_test_primary_benchmark, classes=[0, 1, 2])
auc_mlp = roc_auc_score(y_test_bin_mlp, y_proba_mlp, average='macro', multi_class='ovr')
print("AUC (macro):", round(auc_mlp, 4))

print("\nClassification Report:\n", classification_report(
    y_test_primary_benchmark, y_pred_mlp, target_names=['Show', 'No-show', 'Cancellation'], zero_division=0))

# =========================
# 13) MODELS — TWO-STAGE MLP
# =========================
def evaluate_mlp_binary(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred

# Use NumPy arrays and fill NaNs
X_train_filled_np = np.nan_to_num(np.array(X_train_primary_benchmark, dtype=np.float32), nan=0.0)
X_test_filled_np = np.nan_to_num(np.array(X_test_primary_benchmark, dtype=np.float32), nan=0.0)

# Stage 1: 0 vs Others
y_train_stage1 = (y_train_primary_benchmark != 0).astype(int)
y_test_stage1 = (y_test_primary_benchmark != 0).astype(int)

mlp_stage1 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)
stage1_pred_mlp = evaluate_mlp_binary(
    "Stage 1 MLP (0 vs Others)", mlp_stage1,
    X_train_filled_np, X_test_filled_np,
    y_train_stage1, y_test_stage1
)

# Stage 2: 1 vs 2 for Stage 1 positives
stage2_idx_mlp = np.where(stage1_pred_mlp == 1)[0]
X_test_stage2_mlp = X_test_filled_np[stage2_idx_mlp]
y_test_stage2_mlp = y_test_primary_benchmark[stage2_idx_mlp]

mask_test_12_mlp = np.isin(y_test_stage2_mlp, [1, 2])
X_test_stage2_mlp = X_test_stage2_mlp[mask_test_12_mlp]
y_test_stage2_mlp = y_test_stage2_mlp[mask_test_12_mlp]

train_mask_12_mlp = np.isin(y_train_primary_benchmark, [1, 2])
X_train_stage2_mlp = X_train_filled_np[train_mask_12_mlp]
y_train_stage2_mlp = y_train_primary_benchmark[train_mask_12_mlp]

y_train_binary_mlp = (y_train_stage2_mlp == 1).astype(int)
y_test_binary_mlp = (y_test_stage2_mlp == 1).astype(int)

mlp_stage2 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=10, random_state=42)
evaluate_mlp_binary(
    "Stage 2 MLP (1 vs 2)", mlp_stage2,
    X_train_stage2_mlp, X_test_stage2_mlp,
    y_train_binary_mlp, y_test_binary_mlp
)

# =========================
# 14) 9 PERMUTATIONS — TWO-STAGE DT VARIANTS
# =========================
class_names = {0: "Show", 1: "No-show", 2: "Cancellation"}

def evaluate_model_binary_named(name, model, X_train, X_test, y_train, y_test, target_names):
    """Binary fit/eval with named target classes."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    return y_pred

for A in [0, 1, 2]:
    others = [c for c in [0, 1, 2] if c != A]
    B, C = others

    # Stage 1: A vs Not A (binary)
    y_train_stage1A = (y_train_primary_benchmark != A).astype(int)
    y_test_stage1A = (y_test_primary_benchmark != A).astype(int)

    dt_stage1 = DecisionTreeClassifier(random_state=42)
    stage1_pred_A = evaluate_model_binary_named(
        f"Stage 1: {class_names[A]} vs Not {class_names[A]}",
        dt_stage1,
        X_train_primary_benchmark,
        X_test_primary_benchmark,
        y_train_stage1A,
        y_test_stage1A,
        target_names=[class_names[A], f"Not {class_names[A]}"]
    )

    # Stage 2 subsets
    notA_mask_test = stage1_pred_A == 1
    X_test_stage2 = X_test_primary_benchmark[notA_mask_test]
    y_test_stage2 = pd.Series(y_test_primary_benchmark)[notA_mask_test]

    train_mask_notA = y_train_primary_benchmark != A
    X_train_stage2 = X_train_primary_benchmark[train_mask_notA]
    y_train_stage2 = pd.Series(y_train_primary_benchmark)[train_mask_notA]

    for second_stage_type in ['B_vs_C', 'B_vs_notB', 'C_vs_notC']:
        if second_stage_type == 'B_vs_C':
            # Only B and C
            mask_tr = y_train_stage2.isin([B, C])
            mask_te = y_test_stage2.isin([B, C])

            X_tr = X_train_stage2[mask_tr]
            y_tr_raw = y_train_stage2[mask_tr]
            X_te = X_test_stage2[mask_te]
            y_te_raw = y_test_stage2[mask_te]

            y_tr_bin = (y_tr_raw == B).astype(int)
            y_te_bin = (y_te_raw == B).astype(int)
            target_names_bin = [class_names[C], class_names[B]]
            desc = f"{class_names[B]} vs {class_names[C]}"

        elif second_stage_type == 'B_vs_notB':
            X_tr = X_train_stage2
            y_tr_raw = y_train_stage2
            X_te = X_test_stage2
            y_te_raw = y_test_stage2

            y_tr_bin = (y_tr_raw == B).astype(int)
            y_te_bin = (y_te_raw == B).astype(int)
            target_names_bin = [f"Not {class_names[B]}", class_names[B]]
            desc = f"{class_names[B]} vs Not {class_names[B]}"

        else:  # 'C_vs_notC'
            X_tr = X_train_stage2
            y_tr_raw = y_train_stage2
            X_te = X_test_stage2
            y_te_raw = y_test_stage2

            y_tr_bin = (y_tr_raw == C).astype(int)
            y_te_bin = (y_te_raw == C).astype(int)
            target_names_bin = [f"Not {class_names[C]}", class_names[C]]
            desc = f"{class_names[C]} vs Not {class_names[C]}"

        print(f"\nStage 2: {desc} (after {class_names[A]} vs Not {class_names[A]})")
        print("→ Test count from Stage 1:", notA_mask_test.sum())
        print("→ Stage 2 evaluation sample count:", len(y_te_bin))
        print("→ Binary class counts:", np.bincount(y_te_bin))

        dt_stage2 = DecisionTreeClassifier(random_state=42)
        evaluate_model_binary_named(
            f"Stage 2: {desc} (after {class_names[A]} vs Not {class_names[A]})",
            dt_stage2,
            X_tr, X_te,
            y_tr_bin, y_te_bin,
            target_names=target_names_bin
        )

# =========================
# 15) GRID OF 3×3 TWO-STAGE RUNS (DT/XGB/MLP)
# =========================
def evaluate_binary_proba(name, model, X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Train & evaluate a binary classifier using predict_proba with threshold.
    Returns predictions and probabilities.
    """
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n{name} Performance:")
    print(" Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred, y_proba

def stage_one_binary(model_name, model, X_train, X_test, y_train, y_test):
    """
    Stage 1: learn “2 vs Not 2”.
    """
    print("\n" + "="*60)
    print(f"STAGE 1 ({model_name}): 2 vs Not 2")

    y_tr_bin = (y_train == 2).astype(int)
    y_te_bin = (y_test == 2).astype(int)

    X_tr = np.nan_to_num(np.array(X_train, dtype=np.float32), nan=0.0)
    X_te = np.nan_to_num(np.array(X_test, dtype=np.float32), nan=0.0)

    return evaluate_binary_proba(f"{model_name} (Stage 1)", model, X_tr, X_te, y_tr_bin, y_te_bin)

def stage_two_binary(model_name, model, X_train, X_test, y_train, y_test, stage1_pred, threshold=0.5):
    """
    Stage 2: learn “0 vs Not 0”, on samples Stage 1 predicted Not-2.
    """
    print("\n" + "="*60)
    print(f"STAGE 2 ({model_name}): 0 vs Not 0 (on Stage 1 ≠2 subset)")

    idx_train = np.where(y_train != 2)[0]
    idx_test = np.where(stage1_pred == 0)[0]

    Xtr = X_train[idx_train] if not isinstance(X_train, pd.DataFrame) else X_train.iloc[idx_train]
    ytr = pd.Series(y_train).iloc[idx_train]
    Xte = X_test[idx_test] if not isinstance(X_test, pd.DataFrame) else X_test.iloc[idx_test]
    yte = pd.Series(y_test).iloc[idx_test]

    Xtr = np.nan_to_num(np.array(Xtr, dtype=np.float32), nan=0.0)
    Xte = np.nan_to_num(np.array(Xte, dtype=np.float32), nan=0.0)

    # Relabel for 0 vs Not 0
    ytr_bin = (ytr == 0).astype(int)
    yte_bin = (yte == 0).astype(int)

    print(f" Subset sizes → train: {len(ytr)}, test: {len(yte)}")
    return evaluate_binary_proba(f"{model_name} (Stage 2)", model, Xtr, Xte, ytr_bin, yte_bin, threshold=threshold)

from xgboost import XGBClassifier

all_models = {
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'MLP': MLPClassifier(max_iter=10)
}

# Global NaN safety for this grid
X_train_primary_benchmark_np = np.nan_to_num(np.array(X_train_primary_benchmark), nan=0.0)
X_test_primary_benchmark_np = np.nan_to_num(np.array(X_test_primary_benchmark), nan=0.0)

print("\nStarting grid of 3×3 two-stage runs…")
for name1, model1 in all_models.items():
    stage1_preds, stage1_probas = stage_one_binary(
        name1, model1,
        X_train_primary_benchmark_np, X_test_primary_benchmark_np,
        y_train_primary_benchmark, y_test_primary_benchmark
    )
    for name2, model2 in all_models.items():
        stage_two_binary(
            name2, model2,
            X_train_primary_benchmark_np, X_test_primary_benchmark_np,
            y_train_primary_benchmark, y_test_primary_benchmark,
            stage1_preds,
            threshold=0.5
        )
print("\nAll runs complete.")

# =========================
# 16) THRESHOLD TUNING (PR CURVE) — DT/XGB PERMUTATIONS
# =========================
from xgboost import XGBClassifier

def find_optimal_threshold(y_true, y_scores):
    """Choose threshold maximizing F1 from precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    idx = np.argmax(f1_scores)
    return thresholds[idx], f1_scores[idx]

def stage_one_with_tuned_threshold(model, X_train, X_test, y_train, y_test):
    """Stage 1: (y==2) vs others, with tuned threshold."""
    y_tr_bin = (y_train == 2).astype(int)
    y_te_bin = (y_test == 2).astype(int)

    model.fit(X_train, y_tr_bin)
    y_scores = model.predict_proba(X_test)[:, 1]
    thr, f1 = find_optimal_threshold(y_te_bin, y_scores)

    y_pred = (y_scores >= thr).astype(int)
    print("Stage 1 Optimal Threshold:", round(thr, 4))
    print("Stage 1 Optimal F1:", round(f1, 4))
    print(classification_report(y_te_bin, y_pred, target_names=['Not Cancellation', 'Cancellation'], zero_division=0))
    return y_pred, thr

def stage_two_with_tuned_threshold(model, X_train, X_test, y_train, y_test, stage1_pred):
    """Stage 2: 0 vs Not 0 on non-cancellation subset, with tuned threshold."""
    idx_train = np.where(y_train != 2)[0]
    idx_test = np.where(stage1_pred == 0)[0]

    Xtr = X_train[idx_train] if not isinstance(X_train, pd.DataFrame) else X_train.iloc[idx_train]
    ytr = pd.Series(y_train).iloc[idx_train]
    Xte = X_test[idx_test] if not isinstance(X_test, pd.DataFrame) else X_test.iloc[idx_test]
    yte = pd.Series(y_test).iloc[idx_test]

    ytr_bin = (ytr == 0).astype(int)
    yte_bin = (yte == 0).astype(int)

    model.fit(Xtr, ytr_bin)
    y_scores = model.predict_proba(Xte)[:, 1]
    thr, f1 = find_optimal_threshold(yte_bin, y_scores)
    y_pred = (y_scores >= thr).astype(int)

    print("\nStage 2 Optimal Threshold:", round(thr, 4))
    print("Stage 2 Optimal F1:", round(f1, 4))
    print(classification_report(yte_bin, y_pred, target_names=['Not Show', 'Show'], zero_division=0))
    return thr, f1, y_pred

models_thr = {
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}
permutations = [
    ('Decision Tree', 'Decision Tree'),
    ('Decision Tree', 'XGBoost'),
    ('XGBoost', 'Decision Tree'),
    ('XGBoost', 'XGBoost')
]

results = []
for m1, m2 in permutations:
    print("\n====================================")
    print(f"Permutation: Stage 1 = {m1} -> Stage 2 = {m2}")

    model1 = models_thr[m1]
    model2 = models_thr[m2]

    s1_pred, s1_thr = stage_one_with_tuned_threshold(
        model1, X_train_primary_benchmark, X_test_primary_benchmark,
        y_train_primary_benchmark, y_test_primary_benchmark
    )
    s2_thr, s2_f1, _ = stage_two_with_tuned_threshold(
        model2, X_train_primary_benchmark, X_test_primary_benchmark,
        y_train_primary_benchmark, y_test_primary_benchmark, s1_pred
    )

    results.append({
        'Stage1 Model': m1,
        'Stage2 Model': m2,
        'Stage1 Optimal Threshold': s1_thr,
        'Stage2 Optimal Threshold': s2_thr,
        'Stage2 Optimal F1 Score': s2_f1
    })

results_df = pd.DataFrame(results)
print("\nSummary of Optimal Thresholds and F1 Scores:")
print(results_df)

# =========================
# 17) PATIENT-LEVEL SPLIT + STANDARDIZATION
# =========================
unique_patients = df['PT_ID'].unique()
train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)

df_train_patient = df[df['PT_ID'].isin(train_ids)]
df_test_patient = df[df['PT_ID'].isin(test_ids)]

X_train_primary_benchmark_patient = df_train_patient[final_features].values
y_train_primary_benchmark_patient = df_train_patient['primary_head_label'].values
train_PT_IDs_primary_patient = df_train_patient['PT_ID'].values

X_test_primary_benchmark_patient = df_test_patient[final_features].values
y_test_primary_benchmark_patient = df_test_patient['primary_head_label'].values
test_PT_IDs_primary_patient = df_test_patient['PT_ID'].values

print(f"Total patients: {len(unique_patients)}")
print(f"Train patients: {len(train_ids)}")
print(f"Test patients: {len(test_ids)}")
print(f"Train samples: {len(df_train_patient)}")
print(f"Test samples: {len(df_test_patient)}")

# Standardize numerical features for patient-level split
X_train_primary_benchmark_patient = pd.DataFrame(X_train_primary_benchmark_patient, columns=final_features)
X_test_primary_benchmark_patient = pd.DataFrame(X_test_primary_benchmark_patient, columns=final_features)

scaler_primary_patient = StandardScaler()
if set(numerical_features).issubset(set(X_train_primary_benchmark_patient.columns)):
    X_train_primary_benchmark_patient[numerical_features] = scaler_primary_patient.fit_transform(
        X_train_primary_benchmark_patient[numerical_features]
    )
    X_test_primary_benchmark_patient[numerical_features] = scaler_primary_patient.transform(
        X_test_primary_benchmark_patient[numerical_features]
    )

# =========================
# 18) FINAL MODEL (DT → XGB) WITH FIXED THRESHOLDS — PATIENT SPLIT
# =========================
from xgboost import XGBClassifier

def apply_threshold(scores, thr):
    return (scores >= thr).astype(int)

def stage_one_fixed_threshold(model, X_train, X_test, y_train, y_test, threshold=0.3333):
    """Stage 1: Cancellation (2) vs Not 2 with fixed threshold."""
    y_tr_bin = (y_train == 2).astype(int)
    y_te_bin = (y_test == 2).astype(int)

    model.fit(X_train, y_tr_bin)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = apply_threshold(y_proba, threshold)

    print("\n=== Stage 1 Evaluation (Patient Split) ===")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(y_te_bin, y_pred, target_names=["Not Cancellation", "Cancellation"], zero_division=0))
    return y_pred

def stage_two_fixed_threshold(model, X_train, X_test, y_train, y_test, stage1_pred, threshold=0.8116):
    """Stage 2: Show (0) vs Not Show (1/2) with fixed threshold on Stage-1 Not-2 subset."""
    idx_train = np.where(y_train != 2)[0]
    idx_test = np.where(stage1_pred == 0)[0]

    Xtr = X_train[idx_train]
    ytr = y_train[idx_train]
    Xte = X_test[idx_test]
    yte = y_test[idx_test]

    ytr_bin = (ytr == 0).astype(int)
    yte_bin = (yte == 0).astype(int)

    model.fit(Xtr, ytr_bin)
    y_proba = model.predict_proba(Xte)[:, 1]
    y_pred = apply_threshold(y_proba, threshold)

    print("\n=== Stage 2 Evaluation (Patient Split) ===")
    print(f"Threshold: {threshold:.4f}")
    print(classification_report(yte_bin, y_pred, target_names=["Not Show", "Show"], zero_division=0))
    return y_pred

stage1_model = DecisionTreeClassifier(random_state=42)
stage2_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Fixed thresholds from your note (“patient level threshold 0.43 and 0.83”)
stage1_threshold = 0.43
stage2_threshold = 0.83

# Ensure NumPy arrays
X_train_p = np.array(X_train_primary_benchmark_patient)
X_test_p = np.array(X_test_primary_benchmark_patient)
y_train_p = np.array(y_train_primary_benchmark_patient)
y_test_p = np.array(y_test_primary_benchmark_patient)

stage1_predictions_patient = stage_one_fixed_threshold(
    stage1_model, X_train_p, X_test_p, y_train_p, y_test_p, threshold=stage1_threshold
)
stage2_predictions_patient = stage_two_fixed_threshold(
    stage2_model, X_train_p, X_test_p, y_train_p, y_test_p,
    stage1_predictions_patient, threshold=stage2_threshold
)

# =========================
# 19) PER-PATIENT SEQUENCE SPLIT (80%/20%) + STANDARDIZATION
# =========================
# Re-parse and sort for sequence stability
df['ENCOUNTER_DTTM'] = pd.to_datetime(df['ENCOUNTER_DTTM'], errors='coerce')
df = df.sort_values(['PT_ID', 'ENCOUNTER_DTTM'], ignore_index=True)

train_data, test_data = [], []
for _, g in df.groupby('PT_ID', sort=False):
    n = len(g)
    split_idx = int(0.8 * n)
    if split_idx > 0:
        train_data.append(g.iloc[:split_idx])
        test_data.append(g.iloc[split_idx:])
    else:
        test_data.append(g)

df_train_length = pd.concat(train_data, ignore_index=True)
df_test_length = pd.concat(test_data, ignore_index=True)

X_train_primary_benchmark_length = df_train_length[final_features].to_numpy()
y_train_primary_benchmark_length = df_train_length['primary_head_label'].to_numpy()
train_PT_IDs_primary_length = df_train_length['PT_ID'].to_numpy()

X_test_primary_benchmark_length = df_test_length[final_features].to_numpy()
y_test_primary_benchmark_length = df_test_length['primary_head_label'].to_numpy()
test_PT_IDs_primary_length = df_test_length['PT_ID'].to_numpy()

print(f"Total patients: {df['PT_ID'].nunique()}")
print(f"Train patients: {df_train_length['PT_ID'].nunique()}")
print(f"Test patients: {df_test_length['PT_ID'].nunique()}")
print(f"Train samples: {len(df_train_length)}")
print(f"Test samples: {len(df_test_length)}")

# Optional inspection for a few patients
for pt_id in df_train_length['PT_ID'].unique()[:5]:
    total = df[df['PT_ID'] == pt_id].shape[0]
    train_n = df_train_length[df_train_length['PT_ID'] == pt_id].shape[0]
    test_n = df_test_length[df_test_length['PT_ID'] == pt_id].shape[0]
    print(f"Patient {pt_id} → total={total}, train={train_n}, test={test_n}")

# Standardize for sequence split
X_train_primary_benchmark_length = pd.DataFrame(X_train_primary_benchmark_length, columns=final_features)
X_test_primary_benchmark_length = pd.DataFrame(X_test_primary_benchmark_length, columns=final_features)

scaler_primary_length = StandardScaler()
if set(numerical_features).issubset(set(X_train_primary_benchmark_length.columns)):
    X_train_primary_benchmark_length[numerical_features] = scaler_primary_length.fit_transform(
        X_train_primary_benchmark_length[numerical_features]
    )
    X_test_primary_benchmark_length[numerical_features] = scaler_primary_length.transform(
        X_test_primary_benchmark_length[numerical_features]
    )


# =============================================================================
#                                END OF SCRIPT
# =============================================================================
