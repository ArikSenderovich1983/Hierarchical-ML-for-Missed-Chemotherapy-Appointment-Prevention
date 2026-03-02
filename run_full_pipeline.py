"""
run_full_pipeline.py  —  Complete HCMS Pipeline (single script)
================================================================
Loads df_with_valid_indices_0717.csv, engineers features (vectorized),
trains across 5 no-show cutoffs, and exports ALL 5 tables to LaTeX.

Table 5 uses the CORRECTED no-show-rate denominator.

Usage:   python run_full_pipeline.py
"""

import os, sys, time, ast, random, warnings, json
warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from xgboost import XGBClassifier

DIR = os.path.dirname(os.path.abspath(__file__))

# ── config ───────────────────────────────────────────────────────────────────
CUTOFFS        = [24, 48, 72, 96, 168]
N_HOLDOUTS     = 10
N_BOOTSTRAP    = 100
RS             = 42
XGB_KW         = dict(eval_metric="logloss", verbosity=0, tree_method="hist", n_jobs=-1)

NUMERICAL = [
    "Cancellation_Count", "No_Show_Left_Count", "Provider_Change_Count",
    "Department_Change_Count", "Last_Visit_Duration", "Avg_STD_DURATION",
    "Rescheduled_Appointments_Count", "Avg_Encounter_Frequency_Hours",
    "Avg_Schd_Encounter_Diff_Hours", "Average_Arrival_Lag_Hours",
    "Time_Since_Last_Visit", "Patient Choice", "Clinic Management",
    "Patient Health", "Socioeconomic", "Provider Choice",
]
TIME_FEATS = [
    "ENCOUNTER_DTTM_year", "ENCOUNTER_DTTM_month", "ENCOUNTER_DTTM_day",
    "ENCOUNTER_DTTM_weekday", "ENCOUNTER_DTTM_hour",
    "SCHD_DTTM_year", "SCHD_DTTM_month", "SCHD_DTTM_day",
    "SCHD_DTTM_weekday", "SCHD_DTTM_hour",
]
BINARY = ["Department_Changed_Last_Visit", "Provider_Changed_Last_Visit"]
CAT_PFX = [
    "VISIT_TYPE_", "ENCOUNTER_TYPE_", "CLIN_DEPT_NM_", "FLOOR_LOCATION_NM_",
    "First_Follow_Up_", "Last_Appointment_Status_", "Encounter_Hour_",
    "Weekend_Indicator_",
]
REASON_MAP = {
    "Patient Choice": [
        "Canceled via automated reminder system", "Cancelled via Interface",
        "Deleted via Interface", "Moved", "Patient", "Patient - Personal",
        "Personal Reasons", "Patient - Sought Care Elsewhere",
        "Unhappy/Changed Provider", "Sought Care Elsewhere"],
    "Clinic Management": [
        "Changed by Radiology", "Discharged", "Displaced Appointment",
        "Edu/Meeting", "Error", "Institution", "Order Discontinued",
        "Prep/Med/Results Unavailable", "Schedule Order Error",
        "Scheduled from Wait List", "Institution - Appt Made in Error",
        "Cancelled via automated reminder system", "Level of Care Change",
        "Institution - Condition Warrants Cancellation"],
    "Patient Health": [
        "Clinically Caused", "Deceased", "Feeling Better", "Hospitalized",
        "Labs Out of Acceptable Range", "Level of Care Change",
        "Oncology Treatment Plan Changes", "Patient Dismissed From Practice"],
    "Socioeconomic": ["Financial", "Lack of Transportation", "Financial Concerns"],
    "Provider Choice": [
        "MD Appointment", "Provider", "Provider - Personal",
        "Provider - Professional", "Provider Departure"],
}
_r2c = {r: c for c, rs in REASON_MAP.items() for r in rs}

# ── progress ─────────────────────────────────────────────────────────────────
_T0 = None
def _ts():
    global _T0
    if _T0 is None: _T0 = time.time()
    m, s = divmod(int(time.time() - _T0), 60)
    return f"[{m:02d}:{s:02d}]"
def phase(msg): print(f"\n{_ts()}  >>> {msg}")
def step(msg):  print(f"  {_ts()}  {msg}", flush=True)

# =============================================================================
# PART 1 — FAST FEATURE ENGINEERING  (replaces step2)
# =============================================================================

def fast_feature_engineering(df):
    """Vectorized feature engineering — replaces the slow dict-cache approach."""
    phase("Feature engineering (vectorized)")

    df = df.sort_values(["PT_ID", "ENCOUNTER_DTTM"]).reset_index(drop=True)
    g = df.groupby("PT_ID", sort=False)

    # ── binary helpers (add as columns so groupby can use them) ─────────
    df["_is_canc"]   = (df["STATUS_CD"] == "Canceled").astype(int)
    df["_is_ns"]     = df["STATUS_CD"].isin(["No Show", "Left without seen"]).astype(int)
    df["_not_canc"]  = 1 - df["_is_canc"]
    df["_has_reschd"] = df["RESCHD_DTTM"].notna().astype(int)

    # ── reason-category flags for canceled rows ──────────────────────────
    df["_reason"] = df["CNCL_REASON_DESCR"].map(_r2c).fillna("Unknown")
    cats = ["Patient Choice", "Clinic Management", "Patient Health",
            "Socioeconomic", "Provider Choice"]
    for cat in cats:
        df[f"_rc_{cat}"] = ((df["_is_canc"] == 1) & (df["_reason"] == cat)).astype(int)

    # ── cumulative counts (exclude current row) ──────────────────────────
    step("cumulative counts")
    df["Cancellation_Count"]              = g["_is_canc"].cumsum()    - df["_is_canc"]
    df["No_Show_Left_Count"]              = g["_is_ns"].cumsum()      - df["_is_ns"]
    df["Rescheduled_Appointments_Count"]  = g["_has_reschd"].cumsum() - df["_has_reschd"]
    for cat in cats:
        col = f"_rc_{cat}"
        df[cat] = g[col].cumsum() - df[col]

    # ── provider changes ─────────────────────────────────────────────────
    step("provider / department changes")
    df["_prov_nc"] = df["ENCOUNTER_PROV_ID"].where(df["_not_canc"].astype(bool))
    df["_prov_prev"] = df.groupby("PT_ID")["_prov_nc"].transform("ffill").groupby(df["PT_ID"]).shift()
    df["_prov_ch"] = (df["_prov_nc"].notna() & df["_prov_prev"].notna() &
               (df["_prov_nc"] != df["_prov_prev"])).astype(int)
    df["Provider_Change_Count"]      = g["_prov_ch"].cumsum() - df["_prov_ch"]
    df["Provider_Changed_Last_Visit"] = (
        df["_prov_prev"].notna() &
        df["ENCOUNTER_PROV_ID"].notna() &
        (df["ENCOUNTER_PROV_ID"] != df["_prov_prev"])
    ).astype(int)

    df["_dept_nc"] = df["CLIN_DEPT_ABBREV"].where(df["_not_canc"].astype(bool))
    df["_dept_prev"] = df.groupby("PT_ID")["_dept_nc"].transform("ffill").groupby(df["PT_ID"]).shift()
    df["_dept_ch"] = (df["_dept_nc"].notna() & df["_dept_prev"].notna() &
               (df["_dept_nc"] != df["_dept_prev"])).astype(int)
    df["Department_Change_Count"]       = g["_dept_ch"].cumsum() - df["_dept_ch"]
    df["Department_Changed_Last_Visit"] = (
        df["_dept_prev"].notna() &
        df["CLIN_DEPT_ABBREV"].notna() &
        (df["CLIN_DEPT_ABBREV"] != df["_dept_prev"])
    ).astype(int)

    # ── duration stats ───────────────────────────────────────────────────
    step("duration stats")
    df["_dur_h"] = (df["STD_DURATION"] / 60).where(df["_not_canc"].astype(bool))
    df["_dur_notna"] = df["_dur_h"].notna().astype(int)
    df["_dur_fill0"] = df["_dur_h"].fillna(0)
    df["_dur_cumsum"]  = g["_dur_fill0"].cumsum()  - df["_dur_fill0"]
    df["_dur_cumcnt"]  = g["_dur_notna"].cumsum()  - df["_dur_notna"]
    df["Avg_STD_DURATION"]   = df["_dur_cumsum"] / df["_dur_cumcnt"].replace(0, np.nan)
    df["Last_Visit_Duration"] = df.groupby("PT_ID")["_dur_h"].transform("ffill").groupby(df["PT_ID"]).shift()

    # ── time-derived features ────────────────────────────────────────────
    step("time features")
    enc_dt  = pd.to_datetime(df["ENCOUNTER_DTTM"], errors="coerce")
    schd_dt = pd.to_datetime(df["SCHD_DTTM"], errors="coerce")

    df["_se_diff"] = (enc_dt - schd_dt).dt.total_seconds() / 3600
    df["_se_notna"] = df["_se_diff"].notna().astype(int)
    df["_se_fill0"] = df["_se_diff"].fillna(0)
    df["_se_cumsum"] = g["_se_fill0"].cumsum() - df["_se_fill0"]
    df["_se_cumcnt"] = g["_se_notna"].cumsum() - df["_se_notna"]
    df["Avg_Schd_Encounter_Diff_Hours"] = df["_se_cumsum"] / df["_se_cumcnt"].replace(0, np.nan)

    arr_dt = pd.to_datetime(df["ARRIVED_DTTM"], errors="coerce") if "ARRIVED_DTTM" in df.columns else pd.Series(pd.NaT, index=df.index)
    df["_al_diff"] = (arr_dt - enc_dt).dt.total_seconds() / 3600
    df["_al_notna"] = df["_al_diff"].notna().astype(int)
    df["_al_fill0"] = df["_al_diff"].fillna(0)
    df["_al_cumsum"] = g["_al_fill0"].cumsum() - df["_al_fill0"]
    df["_al_cumcnt"] = g["_al_notna"].cumsum() - df["_al_notna"]
    df["Average_Arrival_Lag_Hours"] = df["_al_cumsum"] / df["_al_cumcnt"].replace(0, np.nan)

    df["_enc_nc"] = enc_dt.where(df["_not_canc"].astype(bool))
    last_enc = df.groupby("PT_ID")["_enc_nc"].transform("ffill").groupby(df["PT_ID"]).shift()
    df["Time_Since_Last_Visit"] = (schd_dt - last_enc).dt.total_seconds() / 3600

    df["_enc_diff_s"] = df.groupby("PT_ID")["_enc_nc"].diff().dt.total_seconds() / 3600
    df["_ed_notna"] = df["_enc_diff_s"].notna().astype(int)
    df["_ed_fill0"] = df["_enc_diff_s"].fillna(0)
    df["_ef_cumsum"] = g["_ed_fill0"].cumsum() - df["_ed_fill0"]
    df["_ef_cumcnt"] = g["_ed_notna"].cumsum() - df["_ed_notna"]
    df["Avg_Encounter_Frequency_Hours"] = df["_ef_cumsum"] / df["_ef_cumcnt"].replace(0, np.nan)

    # ── simple categorical / temporal features ───────────────────────────
    step("categoricals")
    df["Encounter_Month"]   = enc_dt.dt.month
    df["Encounter_Weekday"] = enc_dt.dt.day_name()
    df["Weekend_Indicator"] = (enc_dt.dt.weekday >= 5).astype(int)
    bins   = [0, 12, 18, 24]
    labels = ["before 12pm", "between 12-18pm", "after 18pm"]
    df["Encounter_Hour"] = pd.cut(enc_dt.dt.hour, bins=bins, labels=labels, right=False)
    df["First_Follow_Up"] = g.cumcount().apply(lambda x: "First" if x == 0 else "Follow-Up")
    df["Last_Appointment_Status"] = g["STATUS_CD"].shift()

    # ── datetime expansion for model features ────────────────────────────
    for col, dt_s in [("ENCOUNTER_DTTM", enc_dt), ("SCHD_DTTM", schd_dt)]:
        df[f"{col}_year"]    = dt_s.dt.year
        df[f"{col}_month"]   = dt_s.dt.month
        df[f"{col}_day"]     = dt_s.dt.day
        df[f"{col}_weekday"] = dt_s.dt.weekday
        df[f"{col}_hour"]    = dt_s.dt.hour

    # ── one-hot encode categoricals ──────────────────────────────────────
    ohe_cols = ["VISIT_TYPE", "ENCOUNTER_TYPE", "CLIN_DEPT_NM", "FLOOR_LOCATION_NM",
                "First_Follow_Up", "Encounter_Month", "Encounter_Weekday",
                "Last_Appointment_Status", "Encounter_Hour", "Weekend_Indicator"]
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns])

    # fill NaN numerics with 0 (matches original code behaviour)
    for c in NUMERICAL:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # build final feature list
    ohe_feats = [c for c in df.columns if any(c.startswith(p) for p in CAT_PFX)]
    final_features = TIME_FEATS + NUMERICAL + ohe_feats + BINARY
    final_features = [f for f in final_features if f in df.columns]

    # drop temporary columns
    tmp = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=tmp, inplace=True, errors="ignore")

    step(f"Done — {len(final_features)} features, {len(df):,} rows")
    return df, final_features

# =============================================================================
# PART 2 — LABEL BUILDING
# =============================================================================

def build_labels(df, cutoff):
    df = df.copy()
    enc_dt  = pd.to_datetime(df["ENCOUNTER_DTTM"], errors="coerce")
    cncl_dt = pd.to_datetime(df["CNCL_DTTM"], errors="coerce")
    hrs = (enc_dt - cncl_dt).dt.total_seconds() / 3600
    cw = ((hrs >= 0) & (hrs <= cutoff)).fillna(False).astype(int)
    df["cancelled_within"] = cw

    lbl = np.full(len(df), np.nan)
    lbl[df["STATUS_CD"] == "Completed"] = 0
    lbl[(df["STATUS_CD"].isin(["No Show", "Left without seen"])) | (cw == 1)] = 1
    lbl[(df["STATUS_CD"] == "Canceled") & (cw == 0)] = 2
    df["primary_head_label"] = lbl
    df = df.dropna(subset=["primary_head_label"])
    df["primary_head_label"] = df["primary_head_label"].astype(int)

    df["cancellation_flag"] = ((df["STATUS_CD"] == "Canceled") & (cw == 0)).astype(int)
    df["noshows_flag"]      = ((df["STATUS_CD"].isin(["No Show", "Left without seen"])) | (cw == 1)).astype(int)

    # reason dummies
    df["_rcat"] = df["CNCL_REASON_DESCR"].map(_r2c).fillna("Unknown")
    df_c = pd.get_dummies(df[df["cancellation_flag"] == 1], columns=["_rcat"], prefix="cr")
    df_n = pd.get_dummies(df[df["noshows_flag"] == 1],      columns=["_rcat"], prefix="nr")
    cc = [c for c in df_c.columns if c.startswith("cr_")]
    nc = [c for c in df_n.columns if c.startswith("nr_")]
    for cols, src in [(cc, df_c), (nc, df_n)]:
        if cols:
            df = pd.merge(df, src[cols], left_index=True, right_index=True, how="left")
    df[cc + nc] = df[cc + nc].fillna(0).astype(int)

    if cc:
        df["cancellation_reason_label"] = df[cc].apply(
            lambda r: np.argmax(r.values) if r.sum() > 0 else np.nan, axis=1)
    else:
        df["cancellation_reason_label"] = np.nan
    if nc:
        df["noshow_reason_label"] = df[nc].apply(
            lambda r: np.argmax(r.values) if r.sum() > 0 else np.nan, axis=1)
    else:
        df["noshow_reason_label"] = np.nan

    df.drop(columns=["_rcat"] + cc + nc, inplace=True, errors="ignore")
    return df

# =============================================================================
# PART 3 — MODEL HELPERS
# =============================================================================

def _thr(sc, t): return (sc >= t).astype(int)

def train_pipeline(X_tr, X_te, y_tr, y_te, s1_thr=0.43, s2_thr=0.83, seed=RS):
    """Train DT->XGB pipeline with specified thresholds."""
    y1tr = (y_tr == 2).astype(int); y1te = (y_te == 2).astype(int)
    s1 = DecisionTreeClassifier(random_state=seed)
    s1.fit(X_tr, y1tr)
    s1p = _thr(s1.predict_proba(X_te)[:, 1], s1_thr)

    m1 = {}
    for cls, n in [(1, "Cancel"), (0, "Not-Cancel")]:
        m1[n] = dict(
            precision=precision_score(y1te, s1p, pos_label=cls, zero_division=0),
            recall=recall_score(y1te, s1p, pos_label=cls, zero_division=0),
            f1=f1_score(y1te, s1p, pos_label=cls, zero_division=0),
            support=int(np.sum(y1te == cls)))
    m1["macro_f1"]  = f1_score(y1te, s1p, average="macro", zero_division=0)
    m1["accuracy"]  = accuracy_score(y1te, s1p)

    i2tr = np.where(y_tr != 2)[0]; i2te = np.where(s1p == 0)[0]
    X2tr, y2tr = X_tr[i2tr], y_tr[i2tr]
    X2te, y2te = X_te[i2te], y_te[i2te]
    y2tr_b = (y2tr == 0).astype(int); y2te_b = (y2te == 0).astype(int)
    s2 = XGBClassifier(random_state=seed, **XGB_KW)
    s2.fit(X2tr, y2tr_b)
    s2p = _thr(s2.predict_proba(X2te)[:, 1], s2_thr)

    m2 = {}
    for cls, n in [(0, "No-Show"), (1, "Show")]:
        m2[n] = dict(
            precision=precision_score(y2te_b, s2p, pos_label=cls, zero_division=0),
            recall=recall_score(y2te_b, s2p, pos_label=cls, zero_division=0),
            f1=f1_score(y2te_b, s2p, pos_label=cls, zero_division=0),
            support=int(np.sum(y2te_b == cls)))
    m2["macro_f1"] = f1_score(y2te_b, s2p, average="macro", zero_division=0)
    m2["accuracy"] = accuracy_score(y2te_b, s2p)
    return s1, s2, m1, m2


def tune_thresholds(s1, s2, Xval, yval):
    """
    Sweep thresholds on validation set to maximize per-stage F1.
    Returns (best_s1_thr, best_s2_thr).
    """
    Xval = np.nan_to_num(Xval.astype(np.float32), nan=0, posinf=0, neginf=0)
    y1val = (yval == 2).astype(int)
    
    # Stage 1: sweep threshold for Cancel F1 (step=0.10 to avoid over-tuning)
    s1_probs = s1.predict_proba(Xval)[:, 1]
    best_s1_thr, best_s1_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 1.0, 0.1):
        s1p = _thr(s1_probs, thr)
        f1_cancel = f1_score(y1val, s1p, pos_label=1, zero_division=0)
        if f1_cancel > best_s1_f1:
            best_s1_f1 = f1_cancel
            best_s1_thr = thr
    
    # Stage 2: filter by Stage 1, then sweep threshold for No-Show F1
    s1p_best = _thr(s1_probs, best_s1_thr)
    i2val = np.where(s1p_best == 0)[0]
    if len(i2val) == 0:
        return (best_s1_thr, 0.5)
    
    X2val, y2val = Xval[i2val], yval[i2val]
    y2val_b = (y2val == 0).astype(int)
    s2_probs = s2.predict_proba(X2val)[:, 1]
    
    best_s2_thr, best_s2_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 1.0, 0.1):
        s2p = _thr(s2_probs, thr)
        f1_noshow = f1_score(y2val_b, s2p, pos_label=0, zero_division=0)
        if f1_noshow > best_s2_f1:
            best_s2_f1 = f1_noshow
            best_s2_thr = thr
    
    return (round(best_s1_thr, 2), round(best_s2_thr, 2))


# =============================================================================
# PART 4 — SEMI-SUPERVISED
# =============================================================================

def self_train(mdl, Xtr, ytr, Xu, thr, maxiter=10, seed=RS):
    # Clean inputs (fillna(0))
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xu = np.nan_to_num(Xu, nan=0, posinf=0, neginf=0)
    
    Xp = np.vstack([Xtr, Xu]); yp = np.concatenate([ytr, np.full(len(Xu), -1)])
    for _ in range(maxiter):
        lab = yp != -1; m = clone(mdl)
        X_clean = np.nan_to_num(Xp[lab], nan=0, posinf=0, neginf=0)
        m.fit(X_clean, yp[lab])
        ui = np.where(~lab)[0]
        if not len(ui): break
        X_unlabeled_clean = np.nan_to_num(Xp[ui], nan=0, posinf=0, neginf=0)
        pr = m.predict_proba(X_unlabeled_clean); c = np.max(pr, 1) >= thr
        if not c.any(): break
        yp[ui[c]] = np.argmax(pr[c], 1)
    fm = clone(mdl)
    X_final_clean = np.nan_to_num(Xp[yp != -1], nan=0, posinf=0, neginf=0)
    fm.fit(X_final_clean, yp[yp != -1])
    return fm


def run_semi_sup(X_all, y_all, ff, thr=None, seed=RS):
    y = y_all.copy(); lab = ~np.isnan(y)
    Xl, yl = X_all[lab], y[lab]
    Xu = X_all[~lab]
    Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.2, random_state=seed, stratify=yl)
    
    # Clean NaN/inf BEFORE scaling (fillna(0))
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte, nan=0, posinf=0, neginf=0)
    Xu = np.nan_to_num(Xu, nan=0, posinf=0, neginf=0)
    
    sc = StandardScaler(); ni = [i for i, f in enumerate(ff) if f in NUMERICAL]
    Xtr2 = Xtr.copy(); Xte2 = Xte.copy(); Xu2 = Xu.copy()
    if ni:
        Xtr2[:, ni] = sc.fit_transform(Xtr[:, ni])
        Xte2[:, ni] = sc.transform(Xte[:, ni])
        if len(Xu2): Xu2[:, ni] = sc.transform(Xu[:, ni])
    
    # Clean again after scaling (fillna(0))
    Xtr2 = np.nan_to_num(Xtr2, nan=0, posinf=0, neginf=0).astype(np.float32)
    Xte2 = np.nan_to_num(Xte2, nan=0, posinf=0, neginf=0).astype(np.float32)
    Xu2  = np.nan_to_num(Xu2,  nan=0, posinf=0, neginf=0).astype(np.float32) if len(Xu2) else np.empty((0, Xtr2.shape[1]), dtype=np.float32)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "XGBoost":      XGBClassifier(use_label_encoder=False, random_state=seed, **XGB_KW),
        "MLP":          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=seed),
    }
    res = {}
    for mn, m in models.items():
        # Final NaN check specifically for MLP
        if mn == "MLP":
            if np.any(np.isnan(Xtr2)) or np.any(np.isinf(Xtr2)):
                Xtr2 = np.nan_to_num(Xtr2, nan=0, posinf=1e10, neginf=-1e10)
            if np.any(np.isnan(Xte2)) or np.any(np.isinf(Xte2)):
                Xte2 = np.nan_to_num(Xte2, nan=0, posinf=1e10, neginf=-1e10)
            if len(Xu2) and (np.any(np.isnan(Xu2)) or np.any(np.isinf(Xu2))):
                Xu2 = np.nan_to_num(Xu2, nan=0, posinf=1e10, neginf=-1e10)
        
        if thr is None:
            cm = clone(m); cm.fit(Xtr2, ytr)
        else:
            cm = self_train(m, Xtr2, ytr, Xu2, thr, maxiter=10, seed=seed)
        p = cm.predict(Xte2)
        res[mn] = (round(accuracy_score(yte, p), 3),
                   round(f1_score(yte, p, average="weighted", zero_division=0), 3))
    return res

# =============================================================================
# PART 5 — SPLIT STRATEGIES
# =============================================================================

def holdout_split(df, ff, seed, kind="row"):
    """
    Create 60/20/20 train/val/test split with StandardScaler.
    Returns: Xtr, Xval, Xte, ytr, yval, yte, Xte_df
    """
    if kind == "row":
        X = df[ff].values; y = df["primary_head_label"].values
        # Split into 60% train, 40% temp
        Xtr, Xtemp, ytr, ytemp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
        # Split temp into 50/50 -> val (20%), test (20%)
        Xval, Xte, yval, yte = train_test_split(Xtemp, ytemp, test_size=0.5, random_state=seed, stratify=ytemp)
    elif kind == "patient":
        u = df["PT_ID"].unique()
        tr_pts, temp_pts = train_test_split(u, test_size=0.4, random_state=seed)
        val_pts, te_pts = train_test_split(temp_pts, test_size=0.5, random_state=seed)
        dtr = df[df["PT_ID"].isin(tr_pts)]
        dval = df[df["PT_ID"].isin(val_pts)]
        dte = df[df["PT_ID"].isin(te_pts)]
        Xtr, ytr = dtr[ff].values, dtr["primary_head_label"].values
        Xval, yval = dval[ff].values, dval["primary_head_label"].values
        Xte, yte = dte[ff].values, dte["primary_head_label"].values
    else:  # temporal
        df = df.sort_values(["PT_ID", "ENCOUNTER_DTTM"], ignore_index=True)
        tr, val, te = [], [], []
        for _, g in df.groupby("PT_ID", sort=False):
            n = len(g)
            tr_end = max(1, int(0.6 * n))
            val_end = max(tr_end + 1, int(0.8 * n))
            tr.append(g.iloc[:tr_end])
            val.append(g.iloc[tr_end:val_end])
            te.append(g.iloc[val_end:])
        dtr = pd.concat(tr, ignore_index=True)
        dval = pd.concat(val, ignore_index=True)
        dte = pd.concat(te, ignore_index=True)
        Xtr, ytr = dtr[ff].values, dtr["primary_head_label"].values
        Xval, yval = dval[ff].values, dval["primary_head_label"].values
        Xte, yte = dte[ff].values, dte["primary_head_label"].values
    
    # Clean NaN/inf BEFORE scaling (fillna(0))
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xval = np.nan_to_num(Xval, nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte, nan=0, posinf=0, neginf=0)
    
    # StandardScaler on numerical features
    Xtr_df = pd.DataFrame(Xtr, columns=ff)
    Xval_df = pd.DataFrame(Xval, columns=ff)
    Xte_df = pd.DataFrame(Xte, columns=ff)
    sc = StandardScaler()
    np_cols = [c for c in NUMERICAL if c in ff]
    Xtr_df[np_cols] = sc.fit_transform(Xtr_df[np_cols])
    Xval_df[np_cols] = sc.transform(Xval_df[np_cols])
    Xte_df[np_cols] = sc.transform(Xte_df[np_cols])
    
    # Clean again after scaling (fillna(0))
    Xtr = np.nan_to_num(Xtr_df.values.astype(np.float32), nan=0, posinf=0, neginf=0)
    Xval = np.nan_to_num(Xval_df.values.astype(np.float32), nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte_df.values.astype(np.float32), nan=0, posinf=0, neginf=0)
    
    return Xtr, Xval, Xte, ytr, yval, yte, Xte_df

# =============================================================================
# PART 6 — COUNTERFACTUAL  (CORRECTED DENOMINATOR + BOOTSTRAP)
# =============================================================================

def _s1p(m, X, thr): return _thr(m.predict_proba(X)[:, 1], thr)
def _s2p(m, X, s1, thr): return _thr(m.predict_proba(X[s1 == 0])[:, 1], thr)
def _nsr(s2p, n): return np.sum(s2p == 0) / n if n else 0

def _flip(cf, pct, seed):
    np.random.seed(seed)
    if "Last_Appointment_Status_No Show" in cf.columns:
        ix = cf.index[cf["Last_Appointment_Status_No Show"] == 1].tolist()
        n = int(len(ix) * pct)
        if n: cf.loc[np.random.choice(ix, n, replace=False), "Last_Appointment_Status_No Show"] = 0
    if "Last_Appointment_Status_Completed" in cf.columns:
        ix = cf.index[cf["Last_Appointment_Status_Completed"] == 0].tolist()
        n = int(len(ix) * pct)
        if n: cf.loc[np.random.choice(ix, n, replace=False), "Last_Appointment_Status_Completed"] = 1

def apply_iv(X_df, name, seed=0):
    cf = X_df.copy(); np.random.seed(seed)
    if name == "SMS Reminders":
        cf["Cancellation_Count"] = (cf["Cancellation_Count"]*1.12).round().astype(int)
        cf["No_Show_Left_Count"] = (cf["No_Show_Left_Count"]*0.91).round().astype(int)
        cf["Rescheduled_Appointments_Count"] = (cf["Rescheduled_Appointments_Count"]*1.15).round().astype(int)
        cf["Average_Arrival_Lag_Hours"] *= 0.90
        cf["Patient Choice"] = (cf["Patient Choice"]*0.93).round().astype(int).clip(lower=0)
        _flip(cf, 0.11, seed)
    elif name == "Patient Confirmation":
        cf["Cancellation_Count"] = (cf["Cancellation_Count"]*1.24).round().astype(int)
        cf["No_Show_Left_Count"] = (cf["No_Show_Left_Count"]*0.82).round().astype(int)
        cf["Rescheduled_Appointments_Count"] = (cf["Rescheduled_Appointments_Count"]*1.30).round().astype(int)
        cf["Average_Arrival_Lag_Hours"] *= 0.80
        cf["Patient Choice"] = (cf["Patient Choice"]*0.86).round().astype(int).clip(lower=0)
        _flip(cf, 0.22, seed)
    elif name == "Reduced Lead Time":
        if "Avg_Schd_Encounter_Diff_Hours" in cf.columns:
            cf["Avg_Schd_Encounter_Diff_Hours"] = cf["Avg_Schd_Encounter_Diff_Hours"].clip(upper=168)
        for v in ["Cancellation_Count","No_Show_Left_Count","Rescheduled_Appointments_Count",
                   "Patient Choice","Provider Choice","Clinic Management","Socioeconomic"]:
            if v in cf.columns: cf[v] = (cf[v]*0.50).round().astype(int)
        if "Weekend_Indicator_1" in cf.columns:
            wi = cf.index[cf["Weekend_Indicator_1"]==1].tolist()
            n = int(len(wi)*0.30)
            if n:
                ix = np.random.choice(wi, n, replace=False)
                cf.loc[ix,"Weekend_Indicator_1"]=0; cf.loc[ix,"Weekend_Indicator_0"]=1
        if "Encounter_Hour_before 12pm" in cf.columns:
            bi = cf.index[cf["Encounter_Hour_before 12pm"]==1].tolist()
            n = int(len(bi)*0.30)
            if n:
                ix = np.random.choice(bi, n, replace=False)
                cf.loc[ix,"Encounter_Hour_before 12pm"]=0; cf.loc[ix,"Encounter_Hour_between 12-18pm"]=1
    elif name == "Provider Consistency":
        for v,cap in [("Cancellation_Count",9),("Rescheduled_Appointments_Count",9),
                      ("Patient Choice",9),("Provider_Change_Count",6),("Department_Change_Count",6)]:
            if v in cf.columns: cf[v] = cf[v].clip(upper=cap)
        for v in ["Provider_Changed_Last_Visit","Department_Changed_Last_Visit"]:
            if v in cf.columns: cf[v] = 0
    elif name == "Commitment-Based":
        cf["No_Show_Left_Count"] = (cf["No_Show_Left_Count"]*0.80).round().astype(int)
        cf["Average_Arrival_Lag_Hours"] *= 0.80
        cf["Socioeconomic"] = (cf["Socioeconomic"]*0.80).round().astype(int)
    return cf

IVS = ["SMS Reminders","Patient Confirmation","Reduced Lead Time",
       "Provider Consistency","Commitment-Based"]
STOCHASTIC_IVS = ["SMS Reminders", "Patient Confirmation", "Reduced Lead Time"]

def run_cf_bootstrap(Xtr, Xte_df, ytr, yte, ff, s1_thr, s2_thr, seed=RS):
    """
    Run counterfactual with bootstrap for stochastic interventions.
    Returns: dict[intervention] = {"mean_cancel": float, "var_cancel": float, 
                                     "mean_noshow": float, "var_noshow": float}
    """
    Xtr_np = np.nan_to_num(np.array(Xtr, dtype=np.float32), nan=0, posinf=0, neginf=0)
    
    # Train models once per holdout
    np.random.seed(seed)
    random.seed(seed)
    s1 = DecisionTreeClassifier(random_state=seed)
    s1.fit(Xtr_np, (ytr == 2).astype(int))
    i2 = np.where(ytr != 2)[0]
    s2 = XGBClassifier(random_state=seed, **XGB_KW)
    s2.fit(Xtr_np[i2], (ytr[i2] == 0).astype(int))
    
    # Baseline predictions (no intervention)
    orig = np.nan_to_num(Xte_df[ff].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    s1o = _s1p(s1, orig, s1_thr)
    s2o = _s2p(s2, orig, s1o, s2_thr)
    nt = len(s1o)
    oc = np.mean(s1o==1)
    ons = _nsr(s2o, nt)
    
    res = {}
    for iv in IVS:
        if iv in STOCHASTIC_IVS:
            # Bootstrap: 100 iterations with different random seeds
            cancel_deltas, noshow_deltas = [], []
            for boot_seed in range(N_BOOTSTRAP):
                cfdf = apply_iv(Xte_df, iv, seed=seed*1000 + boot_seed)
                cfnp = np.nan_to_num(cfdf[ff].values.astype(np.float32), nan=0, posinf=0, neginf=0)
                s1c = _s1p(s1, cfnp, s1_thr)
                s2c = _s2p(s2, cfnp, s1c, s2_thr)
                cancel_deltas.append((oc - np.mean(s1c==1))*100)
                noshow_deltas.append((ons - _nsr(s2c, nt))*100)
            res[iv] = {
                "mean_cancel": np.mean(cancel_deltas),
                "var_cancel": np.var(cancel_deltas, ddof=1) if len(cancel_deltas) > 1 else 0.0,
                "mean_noshow": np.mean(noshow_deltas),
                "var_noshow": np.var(noshow_deltas, ddof=1) if len(noshow_deltas) > 1 else 0.0,
            }
        else:
            # Deterministic: single run
            cfdf = apply_iv(Xte_df, iv, seed=seed)
            cfnp = np.nan_to_num(cfdf[ff].values.astype(np.float32), nan=0, posinf=0, neginf=0)
            s1c = _s1p(s1, cfnp, s1_thr)
            s2c = _s2p(s2, cfnp, s1c, s2_thr)
            cancel_delta = (oc - np.mean(s1c==1))*100
            noshow_delta = (ons - _nsr(s2c, nt))*100
            res[iv] = {
                "mean_cancel": cancel_delta,
                "var_cancel": 0.0,
                "mean_noshow": noshow_delta,
                "var_noshow": 0.0,
            }
    
    return res

# =============================================================================
# PART 7 — AGGREGATION HELPERS
# =============================================================================

def aggregate_metrics(metric_list):
    """Aggregate list of metric dicts to mean +/- SD (handles nested class dicts)."""
    keys = metric_list[0].keys()
    agg = {}
    for k in keys:
        sample = metric_list[0][k]
        if isinstance(sample, dict):
            inner_keys = sample.keys()
            agg[k] = {}
            for ik in inner_keys:
                if ik == "support":
                    agg[k][ik] = sample[ik]
                else:
                    vals = [m[k][ik] for m in metric_list]
                    agg[k][ik] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0}
        else:
            vals = [m[k] for m in metric_list]
            agg[k] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0}
    return agg

def aggregate_table5(holdout_results):
    """
    Aggregate Table 5 using law of total variance.
    holdout_results: list of dicts, each dict[iv] = {"mean_cancel": float, "var_cancel": float, ...}
    Returns: dict[iv] = {"cancel_mean": float, "cancel_std": float, ...}
    """
    ivs = holdout_results[0].keys()
    agg = {}
    for iv in ivs:
        # Collect holdout means and variances
        cancel_means = [h[iv]["mean_cancel"] for h in holdout_results]
        cancel_vars = [h[iv]["var_cancel"] for h in holdout_results]
        noshow_means = [h[iv]["mean_noshow"] for h in holdout_results]
        noshow_vars = [h[iv]["var_noshow"] for h in holdout_results]
        
        # Law of total variance: total_var = mean(within_var) + var(means)
        if len(cancel_means) > 1:
            cancel_total_var = np.mean(cancel_vars) + np.var(cancel_means, ddof=1)
            noshow_total_var = np.mean(noshow_vars) + np.var(noshow_means, ddof=1)
        else:
            cancel_total_var = np.mean(cancel_vars)
            noshow_total_var = np.mean(noshow_vars)
        
        agg[iv] = {
            "cancel_mean": np.mean(cancel_means),
            "cancel_std": np.sqrt(cancel_total_var),
            "noshow_mean": np.mean(noshow_means),
            "noshow_std": np.sqrt(noshow_total_var),
        }
    return agg

def _f(v, d=2): return f"{v:.{d}f}"
def _fb(v, d=2): return f"\\textbf{{{v:.{d}f}}}"
def _pm(m, s, d=2): return f"${m:.{d}f} \\pm {s:.{d}f}$"
def _pmb(m, s, d=2): return f"$\\textbf{{{m:.{d}f}}} \\pm {s:.{d}f}$"

def write_latex(t1, t3, t4, t5, path):
    """Write LaTeX tables with mean +/- SD."""
    L = []
    a = L.append
    a(r"\documentclass{article}")
    a(r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}")
    a(r"\usepackage{booktabs}\usepackage{tabularx}\usepackage{multirow}")
    a(r"\usepackage{caption}\usepackage{geometry}\geometry{margin=1in}")
    a(r"\newcolumntype{C}{>{\centering\arraybackslash}X}")
    a(r"\begin{document}")
    a("")

    # TABLE 1
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{Performance of the two-stage pipeline (Decision Tree $\rightarrow$ XGBoost) across no-show cutoff definitions (24h--168h). Mean $\pm$ SD over 10 holdout splits.}")
    a(r"\label{tab:multi_cutoff_results_robust}\small")
    a(r"\begin{tabular}{l c c c c c}\toprule")
    a(r"\textbf{Metric} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\")
    a(r"\midrule")
    a(r"\multicolumn{6}{l}{\textbf{Stage 1: Decision Tree (Cancel vs Not-Cancel)}} \\")
    for mn, cls, fld in [("Precision (Cancel)","Cancel","precision"),("Recall (Cancel)","Cancel","recall"),
        ("F1 (Cancel)","Cancel","f1"),("Precision (Not-Cancel)","Not-Cancel","precision"),
        ("Recall (Not-Cancel)","Not-Cancel","recall"),("F1 (Not-Cancel)","Not-Cancel","f1")]:
        bold = "F1" in mn.split("(")[0]
        vs = [(_pmb if bold else _pm)(t1[c][0][cls][fld]["mean"], t1[c][0][cls][fld]["std"]) for c in CUTOFFS]
        a(f"{mn:25s} & " + " & ".join(vs) + r" \\")
    a(f"{'Macro F1':25s} & " + " & ".join([_pmb(t1[c][0]["macro_f1"]["mean"], t1[c][0]["macro_f1"]["std"]) for c in CUTOFFS]) + r" \\")
    a(f"{'Accuracy':25s} & " + " & ".join([_pm(t1[c][0]["accuracy"]["mean"], t1[c][0]["accuracy"]["std"]) for c in CUTOFFS]) + r" \\")
    a(r"\midrule\multicolumn{6}{l}{\textbf{Stage 2: XGBoost (Show vs No-Show)}} \\")
    for mn, cls, fld in [("Precision (No-Show)","No-Show","precision"),("Recall (No-Show)","No-Show","recall"),
        ("F1 (No-Show)","No-Show","f1"),("Precision (Show)","Show","precision"),
        ("Recall (Show)","Show","recall"),("F1 (Show)","Show","f1")]:
        bold = "F1" in mn.split("(")[0]
        vs = [(_pmb if bold else _pm)(t1[c][1][cls][fld]["mean"], t1[c][1][cls][fld]["std"]) for c in CUTOFFS]
        a(f"{mn:25s} & " + " & ".join(vs) + r" \\")
    a(f"{'Macro F1':25s} & " + " & ".join([_pmb(t1[c][1]["macro_f1"]["mean"], t1[c][1]["macro_f1"]["std"]) for c in CUTOFFS]) + r" \\")
    a(f"{'Accuracy':25s} & " + " & ".join([_pm(t1[c][1]["accuracy"]["mean"], t1[c][1]["accuracy"]["std"]) for c in CUTOFFS]) + r" \\")
    a(r"\bottomrule\end{tabular}")
    a(r"\begin{flushleft}\footnotesize \textit{Note: F1 scores are in \textbf{bold}. Reported as mean $\pm$ SD over 10 holdout splits.}\end{flushleft}\end{table*}")
    a("")

    # TABLE 3 (XGBoost only)
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{XGBoost performance on predicting reasons for no-shows and cancellations. Mean $\pm$ SD over 10 holdout splits.}\small")
    a(r"\begin{tabular}{lcccc}\toprule")
    a(r"\textbf{Cutoff} & \textbf{No-Show Acc} & \textbf{No-Show F1} & \textbf{Cancel Acc} & \textbf{Cancel F1} \\\midrule")
    for c in CUTOFFS:
        na_m, na_s = t3[c]["noshow"]["mean"], t3[c]["noshow"]["std"]
        nf_m, nf_s = t3[c]["noshow_f1"]["mean"], t3[c]["noshow_f1"]["std"]
        ca_m, ca_s = t3[c]["cancel"]["mean"], t3[c]["cancel"]["std"]
        cf_m, cf_s = t3[c]["cancel_f1"]["mean"], t3[c]["cancel_f1"]["std"]
        a(f"{c}h & {_pm(na_m, na_s, 3)} & {_pmb(nf_m, nf_s, 3)} & {_pm(ca_m, ca_s, 3)} & {_pm(cf_m, cf_s, 3)} \\\\")
    a(r"\bottomrule\end{tabular}\label{tab:final_reason_prediction_robust}\end{table*}")
    a("")

    # TABLE 4
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{F1-scores across data splitting strategies. Mean $\pm$ SD over 10 holdout splits.}")
    a(r"\label{tab:split_f1_results_robust}\small")
    a(r"\begin{tabularx}{\textwidth}{l l CCCCC}\toprule")
    a(r"\textbf{Splitting Method} & \textbf{Outcome Class} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\\midrule")
    sps = ["Row-Wise (Random)","Patient-Level","Appointment-Time"]
    for si, sp in enumerate(sps):
        a(f"\\multirow{{3}}{{*}}{{{sp}}} ")
        for nm, ci in [("Cancellations",0),("No-Shows",1),("Shows",2)]:
            vs = [_pm(t4[c][sp][ci]["mean"], t4[c][sp][ci]["std"]) for c in CUTOFFS]
            a(f"& {nm:14s} & " + " & ".join(vs) + r" \\")
        if si < len(sps)-1: a(r"\midrule")
    a(r"\bottomrule\end{tabularx}\end{table*}")
    a("")

    # TABLE 5
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{Counterfactual Analysis: Absolute change in cancellation and no-show rates (pp). Mean $\pm$ SD aggregated via law of total variance over 10 holdout splits $\times$ 100 bootstrap iterations (for stochastic interventions).}")
    a(r"\label{tab:counterfactual_robust}\small")
    a(r"\begin{tabularx}{\textwidth}{l l c c c c c}\toprule")
    a(r"\multirow{2}{*}{\textbf{Intervention}} & \multirow{2}{*}{\textbf{Metric}} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\")
    a(r"\cmidrule(lr){3-7}& & \multicolumn{5}{c}{\textit{(10 holdouts $\times$ 100 bootstrap, aggregated)}} \\\midrule")
    for ii, iv in enumerate(IVS):
        a(f"\\multirow{{2}}{{*}}{{{iv}}} ")
        a("& $\\Delta$ Cancel (\\%) & " + " & ".join([_pm(t5[c][iv]["cancel_mean"], t5[c][iv]["cancel_std"]) for c in CUTOFFS]) + r" \\")
        a("& $\\Delta$ No-show (\\%) & " + " & ".join([_pm(t5[c][iv]["noshow_mean"], t5[c][iv]["noshow_std"]) for c in CUTOFFS]) + r" \\")
        if ii < len(IVS)-1: a(r"\midrule")
    a(r"\bottomrule\end{tabularx}")
    a(r"\begin{flushleft}\footnotesize \textit{Note: $\Delta$ = absolute pp reduction. Positive = improvement. Stochastic interventions bootstrapped 100 times per holdout.}\end{flushleft}\end{table*}")
    a(""); a(r"\end{document}")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L))

# =============================================================================
# PART 9 — MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  HCMS Pipeline — Robust Holdout Experiment")
    print("=" * 70)

    # ── load raw data ────────────────────────────────────────────────────
    step(f"[00:00] Loading df_with_valid_indices_0717.csv...")
    raw = pd.read_csv(os.path.join(DIR, "df_with_valid_indices_0717.csv"))
    step(f"  Loaded {len(raw):,} rows")

    # ── fast feature engineering ─────────────────────────────────────────
    step(f"[00:00] Feature engineering (vectorized)...")
    df_all, ff = fast_feature_engineering(raw)
    step(f"  {len(ff)} features ready")

    # ── results containers (per cutoff, aggregated over holdouts) ────────
    t1_all, t3_all, t4_all, t5_all = {}, {}, {}, {}

    for ci, cutoff in enumerate(CUTOFFS):
        print(f"\n{'─'*70}")
        print(f"  Cutoff {cutoff}h ({ci+1}/{len(CUTOFFS)})")
        print(f"{'─'*70}")
        df = build_labels(df_all.copy(), cutoff)

        # Collect results across 10 holdouts
        t1_holdouts = []  # list of (m1, m2) tuples
        t3_holdouts = []  # list of dicts {"noshow": acc, "noshow_f1": f1, "cancel": acc, "cancel_f1": f1}
        t4_holdouts = {sp: [] for sp in ["Row-Wise (Random)", "Patient-Level", "Appointment-Time"]}
        t5_holdouts = []  # list of dicts from run_cf_bootstrap

        for holdout in range(N_HOLDOUTS):
            seed = RS + holdout
            print(f"  Holdout {holdout+1:2d}/10 | seed={seed}", end="", flush=True)

            # ── Table 1: Row-wise split with threshold tuning ────────────────
            Xtr, Xval, Xte, ytr, yval, yte, Xte_df = holdout_split(df, ff, seed, kind="row")
            
            # Train models
            s1, s2, _, _ = train_pipeline(Xtr, Xval, ytr, yval, seed=seed)
            
            # Tune thresholds on validation set
            s1_thr, s2_thr = tune_thresholds(s1, s2, Xval, yval)
            print(f" → train → tune (S1={s1_thr}, S2={s2_thr})", end="", flush=True)
            
            # Evaluate on test set with tuned thresholds
            _, _, m1, m2 = train_pipeline(Xtr, Xte, ytr, yte, s1_thr=s1_thr, s2_thr=s2_thr, seed=seed)
            t1_holdouts.append((m1, m2))
            print(" → eval T1 ✓", end="", flush=True)

            # ── Table 3: XGBoost reason prediction (only XGBoost) ─────────────
            Xb = df[ff].values
            ns_f = (df["STATUS_CD"].isin(["No Show","Left without seen"])) | (df["cancelled_within"]==1)
            Xns = Xb[ns_f]
            yns = df.loc[ns_f, "noshow_reason_label"].values.astype(float)
            
            ca_f = (df["STATUS_CD"]=="Canceled") & (df["cancelled_within"]==0)
            Xca = Xb[ca_f]
            yca = df.loc[ca_f, "cancellation_reason_label"].values.astype(float)
            ok = ~np.isnan(yca)
            Xca, yca = Xca[ok], yca[ok]
            
            ns_xgb = run_semi_sup(Xns, yns, ff, thr=0.60, seed=seed).get("XGBoost", (0, 0))
            ca_xgb = run_semi_sup(Xca, yca, ff, thr=None, seed=seed).get("XGBoost", (0, 0))
            t3_holdouts.append({
                "noshow": ns_xgb[0],
                "noshow_f1": ns_xgb[1],
                "cancel": ca_xgb[0],
                "cancel_f1": ca_xgb[1],
            })
            print(" T3 ✓", end="", flush=True)

            # ── Table 4: Different split strategies ───────────────────────────
            # Row-wise already done
            t4_holdouts["Row-Wise (Random)"].append((m1["Cancel"]["f1"], m2["No-Show"]["f1"], m2["Show"]["f1"]))
            
            # Patient-level split
            Xtr_p, Xval_p, Xte_p, ytr_p, yval_p, yte_p, _ = holdout_split(df, ff, seed, kind="patient")
            s1_p, s2_p, _, _ = train_pipeline(Xtr_p, Xval_p, ytr_p, yval_p, seed=seed)
            s1_thr_p, s2_thr_p = tune_thresholds(s1_p, s2_p, Xval_p, yval_p)
            _, _, m1_p, m2_p = train_pipeline(Xtr_p, Xte_p, ytr_p, yte_p, s1_thr=s1_thr_p, s2_thr=s2_thr_p, seed=seed)
            t4_holdouts["Patient-Level"].append((m1_p["Cancel"]["f1"], m2_p["No-Show"]["f1"], m2_p["Show"]["f1"]))
            
            # Temporal split
            Xtr_t, Xval_t, Xte_t, ytr_t, yval_t, yte_t, _ = holdout_split(df, ff, seed, kind="temporal")
            s1_t, s2_t, _, _ = train_pipeline(Xtr_t, Xval_t, ytr_t, yval_t, seed=seed)
            s1_thr_t, s2_thr_t = tune_thresholds(s1_t, s2_t, Xval_t, yval_t)
            _, _, m1_t, m2_t = train_pipeline(Xtr_t, Xte_t, ytr_t, yte_t, s1_thr=s1_thr_t, s2_thr=s2_thr_t, seed=seed)
            t4_holdouts["Appointment-Time"].append((m1_t["Cancel"]["f1"], m2_t["No-Show"]["f1"], m2_t["Show"]["f1"]))
            print(" T4 ✓", end="", flush=True)

            # ── Table 5: Counterfactual with bootstrap ───────────────────────
            t5_res = run_cf_bootstrap(Xtr, Xte_df, ytr, yte, ff, s1_thr, s2_thr, seed=seed)
            t5_holdouts.append(t5_res)
            print(" T5 ✓")

        # ── Aggregate across holdouts ─────────────────────────────────────────
        print(f"  Aggregating results...", end="", flush=True)
        
        # Table 1: aggregate m1, m2
        m1_list = [h[0] for h in t1_holdouts]
        m2_list = [h[1] for h in t1_holdouts]
        t1_all[cutoff] = (aggregate_metrics(m1_list), aggregate_metrics(m2_list))
        
        # Table 3: aggregate XGBoost results
        t3_all[cutoff] = {
            "noshow": {"mean": np.mean([h["noshow"] for h in t3_holdouts]),
                       "std": np.std([h["noshow"] for h in t3_holdouts], ddof=1)},
            "noshow_f1": {"mean": np.mean([h["noshow_f1"] for h in t3_holdouts]),
                          "std": np.std([h["noshow_f1"] for h in t3_holdouts], ddof=1)},
            "cancel": {"mean": np.mean([h["cancel"] for h in t3_holdouts]),
                       "std": np.std([h["cancel"] for h in t3_holdouts], ddof=1)},
            "cancel_f1": {"mean": np.mean([h["cancel_f1"] for h in t3_holdouts]),
                          "std": np.std([h["cancel_f1"] for h in t3_holdouts], ddof=1)},
        }
        
        # Table 4: aggregate per split strategy
        t4_all[cutoff] = {}
        for sp in ["Row-Wise (Random)", "Patient-Level", "Appointment-Time"]:
            vals = t4_holdouts[sp]  # list of (cancel_f1, noshow_f1, show_f1) tuples
            t4_all[cutoff][sp] = [
                {"mean": np.mean([v[i] for v in vals]), "std": np.std([v[i] for v in vals], ddof=1)}
                for i in range(3)
            ]
        
        # Table 5: law of total variance aggregation
        t5_all[cutoff] = aggregate_table5(t5_holdouts)
        print(" done")

    # ── export ───────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Exporting LaTeX...")
    out = os.path.join(DIR, "revised_tables_robust.tex")
    write_latex(t1_all, t3_all, t4_all, t5_all, out)
    print(f"  Saved: {out}")
    print(f"\n{'='*70}\n  DONE\n{'='*70}\n")

if __name__ == "__main__":
    main()
