"""
===========================================================
 EXPLAINABILITY & FEATURE IMPORTANCE PIPELINE
===========================================================

 TABLE OF CONTENTS
 -----------------
 1. Setup & Imports
 2. SHAPIQ Analysis
    2.1 Row-Level Explanation
    2.2 Row-Level Summary (Static vs Temporal Pairs)
    2.3 Final Table (Interaction Category Sums)
    2.4 Marginal + Interaction Breakdown
 3. Permutation Importance
    3.1 Stage 1 (Cancellation Detection)
    3.2 Stage 2 (Show vs Not Show)
 4. Built-in Model Importances
    4.1 Decision Tree (Stage 1)
    4.2 XGBoost (Stage 2)
 5. SHAP Analysis
    5.1 Stage 2 SHAP Values & Summary
 6. Mutual Information
    6.1 Stage 1 (Cancellation)
    6.2 Stage 2 (Show vs Not Show)
===========================================================
"""

# === 1. SETUP & IMPORTS ===
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapiq
import shap
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

# ===========================================================
# 2. SHAPIQ ANALYSIS
# ===========================================================

# --- 2.1 SHAPIQ Row-Level Explanation ---
print("\nSHAPIQ Summary for Stage 2 (Subset + Interaction Effects)")

# Config
budget = 512
max_order = 2
subset_size = 100

# Ensure NumPy input
X_test_sub_shap = np.array(X_test_sub_shap)

# Choose subset
np.random.seed(42)
idx = np.random.choice(len(X_test_sub_shap),
                       size=min(subset_size, len(X_test_sub_shap)),
                       replace=False)
X_subset = X_test_sub_shap[idx]

# Explainer
explainer = shapiq.Explainer(
    model=trained_stage2_model,
    data=X_test_sub_shap,
    max_order=max_order
)

# Get explanations
interaction_list = explainer.explain_X(X_subset, budget=budget)

# Prepare storage
num_features = len(final_features)
main_effects = np.zeros((subset_size, num_features))
interaction_sums = {}

# Process each instance
for s_idx, iv in enumerate(interaction_list):
    # Main effects
    order1 = iv.get_n_order_values(1)
    main_effects[s_idx, :] = np.abs(order1)

    # Pairwise interactions
    order2 = iv.get_n_order_values(2)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            key = (i, j)
            interaction_sums.setdefault(key, []).append(abs(order2[i, j]))

# Compute means
main_effects_mean = np.mean(main_effects, axis=0)
pairwise_means = {k: np.mean(v) for k, v in interaction_sums.items()}

# Build summaries
first_order_df = pd.DataFrame({
    "Feature": final_features,
    "MeanAbsMainEffect": main_effects_mean
}).sort_values("MeanAbsMainEffect", ascending=False)

pairwise_df = pd.DataFrame([
    {
        "FeaturePair": f"{final_features[i]} & {final_features[j]}",
        "MeanAbsInteraction": val
    }
    for (i, j), val in pairwise_means.items()
]).sort_values("MeanAbsInteraction", ascending=False)

# Display
print("\n-- First-Order (Main Effects) --")
print(first_order_df.head(10).to_string(index=False))

print("\n-- Second-Order (Pairwise Interactions) --")
print(pairwise_df.head(10).to_string(index=False))


# --- 2.2 Row-Level Summary (Static vs Temporal Pairs) ---
static_prefixes = ["VISIT_TYPE", "ENCOUNTER_TYPE", "CLIN_DEPT_NM", "FLOOR_LOCATION_NM"]

def classify_pair(pair):
    f1, f2 = pair.split(" & ")
    f1_static = any(f1.strip().startswith(pref) for pref in static_prefixes)
    f2_static = any(f2.strip().startswith(pref) for pref in static_prefixes)

    if f1_static and f2_static:
        return "Static-Static"
    elif (f1_static and not f2_static) or (f2_static and not f1_static):
        return "Static-Temporal"
    else:
        return "Temporal-Temporal"

pairwise_df["Category"] = pairwise_df["FeaturePair"].apply(classify_pair)

summary = pairwise_df["Category"].value_counts()
print("\nPairwise Interaction Categories:")
print(summary)


# --- 2.3 Final Table (Interaction Category Sums) ---
category_sums = pairwise_df.groupby("Category")["MeanAbsInteraction"].sum()
category_share = category_sums / category_sums.sum() * 100

print("\nTotal contribution by category:")
print(category_sums)
print("\nPercentage contribution:")
print(category_share)


# --- 2.4 SHAPIQ with Marginal + Interaction Breakdown ---
def is_static_feature(f):
    return any(f.strip().startswith(pref) for pref in static_prefixes)

# Marginal totals
static_main_total = sum(
    main_effects_mean[i] for i, f in enumerate(final_features) if is_static_feature(f)
)
temporal_main_total = sum(
    main_effects_mean[i] for i, f in enumerate(final_features) if not is_static_feature(f)
)

# Interaction totals
def classify_pair(pair):
    f1, f2 = pair.split(" & ")
    f1_static = is_static_feature(f1)
    f2_static = is_static_feature(f2)
    if f1_static and f2_static:
        return "Static-Static"
    elif f1_static ^ f2_static:
        return "Static-Temporal"
    else:
        return "Temporal-Temporal"

pairwise_df["Category"] = pairwise_df["FeaturePair"].apply(classify_pair)
interaction_sums = pairwise_df.groupby("Category")["MeanAbsInteraction"].sum()

# Normalize
total_contrib = static_main_total + temporal_main_total + interaction_sums.sum()
results = {
    "Static Main": 100 * static_main_total / total_contrib,
    "Temporal Main": 100 * temporal_main_total / total_contrib,
    "Static-Static Interaction": 100 * interaction_sums.get("Static-Static", 0) / total_contrib,
    "Temporal-Temporal Interaction": 100 * interaction_sums.get("Temporal-Temporal", 0) / total_contrib,
    "Static-Temporal Interaction": 100 * interaction_sums.get("Static-Temporal", 0) / total_contrib,
}
results_df = pd.DataFrame(results.items(), columns=["Category", "Share (%)"])

print("\n=== Contribution Breakdown by Category ===")
print(results_df.to_string(index=False))


# ===========================================================
# 3. PERMUTATION IMPORTANCE
# ===========================================================

# --- 3.1 Stage 1 ---
print("\n=== Permutation Importance: Stage 1 ===")
y_test_stage1 = (y_test == 2).astype(int)

perm_stage1 = permutation_importance(
    trained_stage1_model, X_test, y_test_stage1,
    n_repeats=2, random_state=42, n_jobs=-1
)

perm_df_stage1 = pd.DataFrame({
    'Feature': final_features,
    'Importance': perm_stage1.importances_mean,
    'Std': perm_stage1.importances_std
}).sort_values("Importance", ascending=False)

print(perm_df_stage1.head(20))


# --- 3.2 Stage 2 ---
print("\n=== Permutation Importance: Stage 2 ===")
y_test_stage2 = (y_test_sub_shap == 0).astype(int)

perm_stage2 = permutation_importance(
    trained_stage2_model, X_test_sub_shap, y_test_stage2,
    n_repeats=2, random_state=42, n_jobs=-1
)

perm_df_stage2 = pd.DataFrame({
    'Feature': final_features,
    'Importance': perm_stage2.importances_mean,
    'Std': perm_stage2.importances_std
}).sort_values("Importance", ascending=False)

print(perm_df_stage2.head(20))


# ===========================================================
# 4. BUILT-IN MODEL IMPORTANCES
# ===========================================================

# --- 4.1 Decision Tree ---
dt_importances = pd.DataFrame({
    'Feature': final_features,
    'Importance': trained_stage1_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Stage 1 Features (Decision Tree):")
print(dt_importances.head(20))


# --- 4.2 XGBoost ---
xgb_importances = pd.DataFrame({
    'Feature': final_features,
    'Importance': trained_stage2_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Stage 2 Features (XGBoost):")
print(xgb_importances.head(20))

# Plotting helper
def plot_importances(df, title):
    df = df.sort_values("Importance", ascending=True).tail(10)
    plt.figure(figsize=(8, 6))
    plt.barh(df["Feature"], df["Importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

plot_importances(dt_importances, "Top Features - Stage 1 (Decision Tree)")
plot_importances(xgb_importances, "Top Features - Stage 2 (XGBoost)")


# ===========================================================
# 5. SHAP ANALYSIS
# ===========================================================

print("\nSHAP Summary for Stage 2 (XGBoost - Show Prediction)")

# Explainer
explainer_stage2 = shap.TreeExplainer(trained_stage2_model)
shap_values_stage2 = explainer_stage2.shap_values(X_test_sub_shap)

# SHAP summary plot
shap.summary_plot(shap_values_stage2, X_test_sub_shap, feature_names=final_features)

# Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values_stage2, columns=final_features)

# Calculate mean absolute SHAP value
shap_importance = shap_df.abs().mean().sort_values(ascending=False)

shap_summary_table = pd.DataFrame({
    'Feature': shap_importance.index,
    'Mean_Abs_SHAP_Value': shap_importance.values
})

print("\nSHAP Feature Importance Table:")
print(shap_summary_table.to_string(index=False))


# ===========================================================
# 6. MUTUAL INFORMATION
# ===========================================================

# --- 6.1 Stage 1 ---
print("\nMutual Information (Stage 1 - Cancellation)")

y_stage1 = (y_test == 2).astype(int)
X_test_df = pd.DataFrame(X_test, columns=final_features).fillna(0)

mi_stage1 = mutual_info_classif(
    X_test_df, y_stage1, discrete_features='auto', random_state=42
)

mi_df_stage1 = pd.DataFrame({
    'Feature': final_features,
    'Mutual Information': mi_stage1
}).sort_values("Mutual Information", ascending=False)

print(mi_df_stage1.head(10))


# --- 6.2 Stage 2 ---
print("\nMutual Information (Stage 2 - Show vs Not Show)")

y_stage2 = (y_test_sub_shap == 0).astype(int)
X_test_sub_df = pd.DataFrame(X_test_sub_shap, columns=final_features).fillna(0)

mi_stage2 = mutual_info_classif(
    X_test_sub_df, y_stage2, discrete_features='auto', random_state=42
)

mi_df_stage2 = pd.DataFrame({
    'Feature': final_features,
    'Mutual Information': mi_stage2
}).sort_values("Mutual Information", ascending=False)

print(mi_df_stage2.head(10))
