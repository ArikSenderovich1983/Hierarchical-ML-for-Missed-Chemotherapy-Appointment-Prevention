"""
===========================================================
 SEMI-SUPERVISED LEARNING PIPELINE (RF + XGBoost)
 Benchmark Target: y_noshow_benchmark (Reason Labels)
===========================================================

 TABLE OF CONTENTS
 -----------------
 1. Setup & Configuration
 2. Semi-Supervised Random Forest (Threshold Sweep)
 3. Semi-Supervised XGBoost (Threshold Sweep, Calibrated)
 4. Semi-Supervised XGBoost (Threshold Sweep, Deterministic/Fast)
 5. Fixed-Threshold Self-Training @ 0.60  → y_final_060
 6. Fixed-Threshold Self-Training @ 0.85  → y_final_085
 7. Label Count Summaries for 0.60 / 0.85 runs
 8. Train/Test Split for Clean Ground Truth + Pseudo Labels (both 0.60 & 0.85)
 9. Standardization (Primary/Cancellation/No-Show tasks)
10. Reason Prediction Experiments
    10.1 Semi-supervised @0.60 (No-Show & Cancellation)
    10.2 Semi-supervised @0.85 (No-Show & Cancellation)
    10.3 Baseline (No Pseudo-Labels)
===========================================================
"""

# ===========================================================
# 1) SETUP & CONFIGURATION
# ===========================================================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

# ---------------------------
# Global configuration
# ---------------------------
thresholds = np.arange(0.60, 0.991, 0.05)  # sweep thresholds: 0.60, 0.65, ... 0.95, 1.00 (rounded)
max_iter = 20
random_state = 42

# Reproducibility
np.random.seed(random_state)
random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

# ---------------------------
# INPUTS expected from previous pipeline:
#   X_noshow_benchmark: np.array or DataFrame (features for no-show reason task)
#   y_noshow_benchmark: np.array of labels; unlabeled marked as 5.0
#   final_features: list of feature names (for DataFrame wrapping later)
# ---------------------------
# assert 'X_noshow_benchmark' in globals() and 'y_noshow_benchmark' in globals() and 'final_features' in globals()


# ===========================================================
# 2) SEMI-SUPERVISED RANDOM FOREST (THRESHOLD SWEEP)
# ===========================================================
print("\n" + "="*70)
print("2) SEMI-SUPERVISED RANDOM FOREST (Threshold Sweep)")
print("="*70)

# --- DATA PREPARATION (shared) ---
X = X_noshow_benchmark
y = y_noshow_benchmark.copy()

# Separate labeled vs unlabeled by your convention (5.0 = unlabeled)
labeled_mask = y != 5.0
unlabeled_mask = y == 5.0

X_labeled = X[labeled_mask]
y_labeled = y[labeled_mask]
X_unlabeled = X[unlabeled_mask]

# Train-test split from labeled only (clean ground truth split)
X_train, X_test, y_train, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=random_state, stratify=y_labeled
)

# --- THRESHOLD SWEEP ---
rf_results = []

for thresh in thresholds:
    print(f"\n🔍 RF — Running threshold = {thresh:.2f}")

    # Initialize semi-supervised pool each run
    X_semi = np.concatenate([X_train, X_unlabeled], axis=0)
    y_semi = np.concatenate([y_train, np.full(len(X_unlabeled), -1)], axis=0)

    # Calibrated RandomForest (isotonic calibration, 3-fold CV)
    model = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=random_state, n_jobs=-1
        ),
        method='isotonic', cv=3
    )

    # Self-training loop: add confident pseudo-labels
    iteration = 0
    while iteration < max_iter:
        iteration += 1

        model = clone(model)
        model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])

        unlabeled_indices = np.where(y_semi == -1)[0]
        if len(unlabeled_indices) == 0:
            break

        X_unlab = X_semi[unlabeled_indices]
        proba = model.predict_proba(X_unlab)
        max_proba = np.max(proba, axis=1)
        confident_mask = max_proba >= thresh

        if not np.any(confident_mask):
            break

        pseudo_labels = model.predict(X_unlab[confident_mask])
        confident_indices = unlabeled_indices[confident_mask]
        y_semi[confident_indices] = pseudo_labels

    # Final evaluation on held-out clean test set
    final_model = clone(model)
    final_model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])
    acc = accuracy_score(y_test, final_model.predict(X_test))
    total_labeled = np.sum(y_semi != -1)
    n_pseudo = total_labeled - len(y_train)

    rf_results.append((thresh, acc, n_pseudo))

# --- PRINT RESULTS ---
print("\n📈 RF — Threshold tuning results:")
for thresh, acc, num in rf_results:
    print(f"Threshold = {thresh:.2f} | Accuracy = {acc:.4f} | Pseudo-labels added = {num}")

# --- PLOT RESULTS ---
threshold_vals, acc_vals, pseudo_counts = zip(*rf_results)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(threshold_vals, acc_vals, marker='o')
plt.title('RF: Threshold vs Test Accuracy'); plt.xlabel('Threshold'); plt.ylabel('Accuracy'); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(threshold_vals, pseudo_counts, marker='s')
plt.title('RF: Threshold vs Pseudo-Labeled Samples'); plt.xlabel('Threshold'); plt.ylabel('# Pseudo-Labels'); plt.grid(True)
plt.tight_layout(); plt.show()


# ===========================================================
# 3) SEMI-SUPERVISED XGBOOST (THRESHOLD SWEEP, CALIBRATED)
# ===========================================================
print("\n" + "="*70)
print("3) SEMI-SUPERVISED XGBOOST (Calibrated, Threshold Sweep)")
print("="*70)

xgb_calib_results = []

for thresh in thresholds:
    print(f"\n🔍 XGB-Calib — Running threshold = {thresh:.2f}")

    # Reset combined data for each threshold
    X_semi = np.concatenate([X_train, X_unlabeled], axis=0)
    y_semi = np.concatenate([y_train, np.full(len(X_unlabeled), -1)], axis=0)

    xgb_base = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=random_state, n_jobs=-1
    )
    model = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        model = clone(model)
        model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])

        unlabeled_indices = np.where(y_semi == -1)[0]
        if len(unlabeled_indices) == 0:
            break

        X_unlab = X_semi[unlabeled_indices]
        proba = model.predict_proba(X_unlab)
        max_proba = np.max(proba, axis=1)
        confident_mask = max_proba >= thresh
        if not np.any(confident_mask):
            break

        pseudo_labels = model.predict(X_unlab[confident_mask])
        y_semi[unlabeled_indices[confident_mask]] = pseudo_labels

    final_model = clone(model)
    final_model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])
    acc = accuracy_score(y_test, final_model.predict(X_test))
    n_pseudo = np.sum(y_semi[len(y_train):] != -1)

    xgb_calib_results.append((thresh, acc, n_pseudo))

# Print + plot
print("\n📈 XGB-Calib — Threshold sweep results:")
for thresh, acc, num in xgb_calib_results:
    print(f"Threshold = {thresh:.2f} | Accuracy = {acc:.4f} | Pseudo-labels = {num}")

ths, accs, pseudos = zip(*xgb_calib_results)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(ths, accs, marker='o'); plt.title('XGB-Calib: Threshold vs Accuracy'); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(ths, pseudos, marker='s'); plt.title('XGB-Calib: Threshold vs #Pseudo'); plt.grid(True)
plt.tight_layout(); plt.show()


# ===========================================================
# 4) SEMI-SUPERVISED XGBOOST (DETERMINISTIC/FAST)
# ===========================================================
print("\n" + "="*70)
print("4) SEMI-SUPERVISED XGBOOST (Deterministic/Fast Threshold Sweep)")
print("="*70)

xgb_fast_results = []

for thresh in thresholds:
    print(f"\n🔍 XGB-Fast — Threshold = {thresh:.2f}")

    X_semi = np.concatenate([X_train, X_unlabeled], axis=0)
    y_semi = np.concatenate([y_train, np.full(len(X_unlabeled), -1)], axis=0)

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=random_state, n_jobs=-1
    )

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])

        unlabeled_idx = np.where(y_semi == -1)[0]
        if len(unlabeled_idx) == 0:
            break

        X_unlab = X_semi[unlabeled_idx]
        proba = model.predict_proba(X_unlab)
        max_proba = np.max(proba, axis=1)
        confident_mask = max_proba >= thresh
        if not np.any(confident_mask):
            break

        y_semi[unlabeled_idx[confident_mask]] = model.predict(X_unlab[confident_mask])

    acc = accuracy_score(y_test, model.predict(X_test))
    n_pseudo = np.sum(y_semi[len(y_train):] != -1)

    xgb_fast_results.append((thresh, acc, n_pseudo))

print("\n📈 XGB-Fast — Threshold sweep results:")
for thresh, acc, num in xgb_fast_results:
    print(f"Threshold {thresh:.2f} | Accuracy: {acc:.4f} | Pseudo-labels: {num}")

ths, accs, pseudos = zip(*xgb_fast_results)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.plot(ths, accs, marker='o'); plt.title('XGB-Fast: Threshold vs Accuracy'); plt.grid(True)
plt.subplot(1, 2, 2); plt.plot(ths, pseudos, marker='s'); plt.title('XGB-Fast: Threshold vs #Pseudo'); plt.grid(True)
plt.tight_layout(); plt.show()


# ===========================================================
# 5) FIXED THRESHOLD SELF-TRAINING @ 0.60 (GROUND TRUTH SPLIT)
# ===========================================================
print("\n" + "="*70)
print("5) FIXED THRESHOLD SELF-TRAINING @ 0.60 (Calibrated XGB)")
print("="*70)

threshold = 0.60
max_iter = 20

# Original data copies
X = X_noshow_benchmark.copy()
y_original = y_noshow_benchmark.copy()

# Split labeled / unlabeled
labeled_mask = y_original != 5.0
unlabeled_mask = y_original == 5.0

X_labeled = X[labeled_mask]
y_labeled = y_original[labeled_mask]
X_unlabeled = X[unlabeled_mask]

# Clean test set from real labels only
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=random_state, stratify=y_labeled
)

# Combine labeled train + unlabeled for self-training
X_semi = np.concatenate([X_train_real, X_unlabeled], axis=0)
y_semi = np.concatenate([y_train_real, np.full(len(X_unlabeled), -1)], axis=0)

# Calibrated XGB setup
xgb_base = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=random_state, n_jobs=-1
)
cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
model = CalibratedClassifierCV(xgb_base, method='isotonic', cv=cv_folds)

# Self-training loop at 0.60
iteration = 0
while iteration < max_iter:
    iteration += 1
    model = clone(model)
    model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])

    unlabeled_indices = np.where(y_semi == -1)[0]
    if len(unlabeled_indices) == 0:
        break

    X_unlab = X_semi[unlabeled_indices]
    proba = model.predict_proba(X_unlab)
    max_proba = np.max(proba, axis=1)
    confident_mask = max_proba >= threshold

    if not np.any(confident_mask):
        break

    pseudo_labels = model.predict(X_unlab[confident_mask])
    y_semi[unlabeled_indices[confident_mask]] = pseudo_labels

# Build final y with pseudo labels for confident ones
y_final_060 = y_original.copy()
pseudo_labels_only = y_semi[len(y_train_real):]  # only unlabeled portion
confident_final_mask = pseudo_labels_only != -1
final_confident_indices = np.where(unlabeled_mask)[0][confident_final_mask]
y_final_060[final_confident_indices] = pseudo_labels_only[confident_final_mask]

print(f"✅ (0.60) Pseudo-labeling complete: {len(final_confident_indices)} of {len(X_unlabeled)} unlabeled samples labeled.")


# ===========================================================
# 6) FIXED THRESHOLD SELF-TRAINING @ 0.85 (GROUND TRUTH SPLIT)
# ===========================================================
print("\n" + "="*70)
print("6) FIXED THRESHOLD SELF-TRAINING @ 0.85 (Calibrated XGB)")
print("="*70)

threshold = 0.85
max_iter = 20

# Original data copies
X = X_noshow_benchmark.copy()
y_original = y_noshow_benchmark.copy()

# Split labeled / unlabeled
labeled_mask = y_original != 5.0
unlabeled_mask = y_original == 5.0

X_labeled = X[labeled_mask]
y_labeled = y_original[labeled_mask]
X_unlabeled = X[unlabeled_mask]

# Clean test set from real labels only
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=random_state, stratify=y_labeled
)

# Combine labeled train + unlabeled for self-training
X_semi = np.concatenate([X_train_real, X_unlabeled], axis=0)
y_semi = np.concatenate([y_train_real, np.full(len(X_unlabeled), -1)], axis=0)

# Calibrated XGB setup
xgb_base = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=random_state, n_jobs=-1
)
cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
model = CalibratedClassifierCV(xgb_base, method='isotonic', cv=cv_folds)

# Self-training loop at 0.85
iteration = 0
while iteration < max_iter:
    iteration += 1
    model = clone(model)
    model.fit(X_semi[y_semi != -1], y_semi[y_semi != -1])

    unlabeled_indices = np.where(y_semi == -1)[0]
    if len(unlabeled_indices) == 0:
        break

    X_unlab = X_semi[unlabeled_indices]
    proba = model.predict_proba(X_unlab)
    max_proba = np.max(proba, axis=1)
    confident_mask = max_proba >= threshold

    if not np.any(confident_mask):
        break

    pseudo_labels = model.predict(X_unlab[confident_mask])
    y_semi[unlabeled_indices[confident_mask]] = pseudo_labels

# Final y for 0.85
y_final_085 = y_original.copy()
pseudo_labels_only = y_semi[len(y_train_real):]  # unlabeled portion only
confident_final_mask = pseudo_labels_only != -1
final_confident_indices = np.where(unlabeled_mask)[0][confident_final_mask]
y_final_085[final_confident_indices] = pseudo_labels_only[confident_final_mask]

print(f"✅ (0.85) Pseudo-labeling complete: {len(final_confident_indices)} of {len(X_unlabeled)} unlabeled samples labeled.")


# ===========================================================
# 7) LABEL COUNT SUMMARIES (0.60 and 0.85)
# ===========================================================
print("\n" + "="*70)
print("7) LABEL COUNT SUMMARIES")
print("="*70)

print("\nLabel counts y_final_060:")
print(pd.Series(y_final_060).value_counts().sort_index())

print("\nLabel counts y_final_085:")
print(pd.Series(y_final_085).value_counts().sort_index())


# ===========================================================
# 8) TRAIN/TEST SPLIT FOR CLEAN GROUND TRUTH + PSEUDO LABELS
#    (Build final train/test for no-show reason task)
#    We create sets for both 0.60 and 0.85 versions.
# ===========================================================
print("\n" + "="*70)
print("8) FINAL TRAIN/TEST SPLITS (with Pseudo Labels)")
print("="*70)

def build_train_test_with_pseudo(y_final):
    """
    Train set includes real + pseudo (exclude -1 only).
    Test set always uses original real labels only (no pseudo).
    """
    # Train data mask (exclude -1.0)
    train_mask = y_final != -1.0
    X_train_all = X_noshow_benchmark[train_mask]
    y_train_all = y_final[train_mask]

    # Test data from original real labels
    original_labeled_mask = y_noshow_benchmark != 5.0
    X_labeled_original = X_noshow_benchmark[original_labeled_mask]
    y_labeled_original = y_noshow_benchmark[original_labeled_mask]

    # Test set produced from real labels split
    _, X_test_only_real, _, y_test_only_real = train_test_split(
        X_labeled_original, y_labeled_original,
        test_size=0.2, random_state=42, stratify=y_labeled_original
    )

    # Training uses all real + pseudo-labeled (no further split)
    return (X_train_all, y_train_all, X_test_only_real, y_test_only_real)

# Build both 0.60 and 0.85 train/test
X_train_noshow_060, y_train_noshow_060, X_test_noshow_060, y_test_noshow_060 = build_train_test_with_pseudo(y_final_060)
X_train_noshow_085, y_train_noshow_085, X_test_noshow_085, y_test_noshow_085 = build_train_test_with_pseudo(y_final_085)

print(f"(0.60) ✅ Train set: {len(y_train_noshow_060)} (real+pseudo), Test set: {len(y_test_noshow_060)} (real only)")
print(f"(0.85) ✅ Train set: {len(y_train_noshow_085)} (real+pseudo), Test set: {len(y_test_noshow_085)} (real only)")


# ===========================================================
# 9) STANDARDIZATION (for Primary / Cancellation / No-Show tasks)
#    NOTE: Below reuses your earlier variable names for consistency.
#    Wrap no-show train/test into DataFrames with final_features as columns.
# ===========================================================
print("\n" + "="*70)
print("9) STANDARDIZATION")
print("="*70)

# These matrices are expected from your earlier pipeline steps:
# - X_train_primary_benchmark, X_test_primary_benchmark
# - X_train_cancellation_benchmark, X_test_cancellation_benchmark
# Ensure they already exist; here we only standardize. For noshow, we build DF now.

# Convert to DataFrame for safer column operations
# (primary and cancellation blocks assume they already exist in your runtime)
try:
    X_train_primary_benchmark = pd.DataFrame(X_train_primary_benchmark, columns=final_features)
    X_test_primary_benchmark  = pd.DataFrame(X_test_primary_benchmark, columns=final_features)
except Exception:
    pass  # If not present in this session, skip silently (they belong to earlier script sections)

try:
    X_train_cancellation_benchmark = pd.DataFrame(X_train_cancellation_benchmark, columns=final_features)
    X_test_cancellation_benchmark  = pd.DataFrame(X_test_cancellation_benchmark, columns=final_features)
except Exception:
    pass

# No-Show @ 0.60
X_train_noshow_benchmark_060 = pd.DataFrame(X_train_noshow_060, columns=final_features)
X_test_noshow_benchmark_060  = pd.DataFrame(X_test_noshow_060,  columns=final_features)
# No-Show @ 0.85
X_train_noshow_benchmark_085 = pd.DataFrame(X_train_noshow_085, columns=final_features)
X_test_noshow_benchmark_085  = pd.DataFrame(X_test_noshow_085,  columns=final_features)

# Numerical feature list (same as earlier pipeline)
numerical_features_standardize = [
    'Cancellation_Count', 'No_Show_Left_Count', 'Provider_Change_Count', 'Department_Change_Count',
    'Last_Visit_Duration', 'Avg_STD_DURATION', 'Rescheduled_Appointments_Count',
    'Avg_Encounter_Frequency_Hours', 'Avg_Schd_Encounter_Diff_Hours', 'Average_Arrival_Lag_Hours',
    'Time_Since_Last_Visit', 'Patient Choice','Clinic Management', 'Patient Health', 'Socioeconomic', 'Provider Choice',
]

# Separate scalers per task (avoid leakage)
scaler_primary = StandardScaler()
scaler_cancellation = StandardScaler()
scaler_noshow_060 = StandardScaler()
scaler_noshow_085 = StandardScaler()

# Primary (if dataframes exist)
if 'X_train_primary_benchmark' in globals() and isinstance(X_train_primary_benchmark, pd.DataFrame):
    cols_present = [c for c in numerical_features_standardize if c in X_train_primary_benchmark.columns]
    X_train_primary_benchmark[cols_present] = scaler_primary.fit_transform(X_train_primary_benchmark[cols_present])
    X_test_primary_benchmark[cols_present]  = scaler_primary.transform(X_test_primary_benchmark[cols_present])

# Cancellation (if dataframes exist)
if 'X_train_cancellation_benchmark' in globals() and isinstance(X_train_cancellation_benchmark, pd.DataFrame):
    cols_present = [c for c in numerical_features_standardize if c in X_train_cancellation_benchmark.columns]
    X_train_cancellation_benchmark[cols_present] = scaler_cancellation.fit_transform(X_train_cancellation_benchmark[cols_present])
    X_test_cancellation_benchmark[cols_present]  = scaler_cancellation.transform(X_test_cancellation_benchmark[cols_present])

# No-Show @ 0.60
cols_present_060 = [c for c in numerical_features_standardize if c in X_train_noshow_benchmark_060.columns]
X_train_noshow_benchmark_060[cols_present_060] = scaler_noshow_060.fit_transform(X_train_noshow_benchmark_060[cols_present_060])
X_test_noshow_benchmark_060[cols_present_060]  = scaler_noshow_060.transform(X_test_noshow_benchmark_060[cols_present_060])

# No-Show @ 0.85
cols_present_085 = [c for c in numerical_features_standardize if c in X_train_noshow_benchmark_085.columns]
X_train_noshow_benchmark_085[cols_present_085] = scaler_noshow_085.fit_transform(X_train_noshow_benchmark_085[cols_present_085])
X_test_noshow_benchmark_085[cols_present_085]  = scaler_noshow_085.transform(X_test_noshow_benchmark_085[cols_present_085])

print("Standardization complete for no-show (0.60 & 0.85).")


# ===========================================================
# 10) REASON PREDICTION EXPERIMENTS
#     (DecisionTree, XGB, MLP) — Independent of stage-1/2
# ===========================================================
print("\n" + "="*70)
print("10) REASON PREDICTION EXPERIMENTS")
print("="*70)

from sklearn.metrics import accuracy_score

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric matrix for all models; coerce non-numeric to 0."""
    return df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

def build_models():
    """Return the three models for comparison."""
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=42)
    }

def evaluate_models(X_train, X_test, y_train, y_test, label=""):
    """Train & evaluate all models, printing accuracy and weighted F1."""
    print(f"\n--- {label} Reason Prediction ---")
    results = {}

    X_train = ensure_numeric(X_train)
    X_test  = ensure_numeric(X_test)

    for model_name, model in build_models().items():
        print(f"\nModel: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = report['weighted avg']['f1-score']

        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1-score: {f1:.4f}")

        results[model_name] = {'accuracy': acc, 'f1_score': f1}

    return results

# ------------------------------------
# 10.1) Semi-supervised @ 0.60 (No-Show)
# ------------------------------------
noshow_results_060 = evaluate_models(
    X_train_noshow_benchmark_060,
    X_test_noshow_benchmark_060,
    y_train_noshow_060,
    y_test_noshow_060,
    label="No-Show (Semi-supervised @0.60)"
)

# ------------------------------------
# 10.2) Semi-supervised @ 0.85 (No-Show)
# ------------------------------------
noshow_results_085 = evaluate_models(
    X_train_noshow_benchmark_085,
    X_test_noshow_benchmark_085,
    y_train_noshow_085,
    y_test_noshow_085,
    label="No-Show (Semi-supervised @0.85)"
)

# ------------------------------------
# 10.3) Baseline (No Pseudo-Labels) for No-Show
# ------------------------------------
mask_labeled_only = y_noshow_benchmark != 5.0
X_labeled_noshow = X_noshow_benchmark[mask_labeled_only]
y_labeled_noshow = y_noshow_benchmark[mask_labeled_only]

X_train_noshow_base, X_test_noshow_base, y_train_noshow_base, y_test_noshow_base = train_test_split(
    X_labeled_noshow, y_labeled_noshow, test_size=0.2, stratify=y_labeled_noshow, random_state=42
)
X_train_noshow_base = pd.DataFrame(X_train_noshow_base, columns=final_features)
X_test_noshow_base  = pd.DataFrame(X_test_noshow_base,  columns=final_features)

noshow_results_baseline = evaluate_models(
    X_train_noshow_base,
    X_test_noshow_base,
    y_train_noshow_base,
    y_test_noshow_base,
    label="No-Show (Baseline, No Pseudo Labels)"
)

# ------------------------------------
# 10.x) Cancellation Reason Prediction (uses your earlier cancellation split)
#       Ensure X_train_cancellation_benchmark and y_* exist from earlier pipeline.
# ------------------------------------
cancel_results = {}
if 'X_train_cancellation_benchmark' in globals() and isinstance(X_train_cancellation_benchmark, pd.DataFrame):
    cancel_results = evaluate_models(
        X_train_cancellation_benchmark,
        X_test_cancellation_benchmark,
        y_train_cancellation_benchmark,
        y_test_cancellation_benchmark,
        label="Cancellation"
    )
else:
    print("\n⚠️ Skipping 'Cancellation' evaluation because cancellation split variables are not in scope here.")

# ------------------------------------
# 10.y) Summaries
# ------------------------------------
def make_summary_df(noshow_results_dict, label_prefix):
    return pd.DataFrame({
        'Model': list(noshow_results_dict.keys()),
        f'{label_prefix} Accuracy': [noshow_results_dict[m]['accuracy'] for m in noshow_results_dict],
        f'{label_prefix} F1': [noshow_results_dict[m]['f1_score'] for m in noshow_results_dict],
    })

summary_060 = make_summary_df(noshow_results_060, 'No-Show@0.60')
summary_085 = make_summary_df(noshow_results_085, 'No-Show@0.85')
summary_base = make_summary_df(noshow_results_baseline, 'No-Show@Baseline')

print("\n📊 Model Comparison Summary — No-Show @0.60:")
print(summary_060)

print("\n📊 Model Comparison Summary — No-Show @0.85:")
print(summary_085)

print("\n📊 Model Comparison Summary — No-Show Baseline:")
print(summary_base)

if cancel_results:
    cancel_summary = pd.DataFrame({
        'Model': list(cancel_results.keys()),
        'Cancel Accuracy': [cancel_results[m]['accuracy'] for m in cancel_results],
        'Cancel F1': [cancel_results[m]['f1_score'] for m in cancel_results]
    })
    print("\n📊 Model Comparison Summary — Cancellation:")
    print(cancel_summary)
