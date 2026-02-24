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
N_SEEDS        = 10
S1_THR         = 0.43
S2_THR         = 0.83
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

def train_pipeline(X_tr, X_te, y_tr, y_te, seed=RS):
    y1tr = (y_tr == 2).astype(int); y1te = (y_te == 2).astype(int)
    s1 = DecisionTreeClassifier(random_state=seed)
    s1.fit(X_tr, y1tr)
    s1p = _thr(s1.predict_proba(X_te)[:, 1], S1_THR)

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
    s2p = _thr(s2.predict_proba(X2te)[:, 1], S2_THR)

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


def combined_f1(s1, s2, X, y):
    X = np.nan_to_num(X.astype(np.float32), nan=0, posinf=0, neginf=0)
    s1p = _thr(s1.predict_proba(X)[:, 1], S1_THR)
    pred = np.full(len(y), -1)
    pred[s1p == 1] = 2
    nc = np.where(s1p == 0)[0]
    if len(nc):
        s2p = _thr(s2.predict_proba(X[nc])[:, 1], S2_THR)
        pred[nc[s2p == 1]] = 0
        pred[nc[s2p == 0]] = 1
    y = np.asarray(y, dtype=int)
    return tuple(f1_score(y, pred, labels=[c], average=None, zero_division=0)[0] for c in [2, 1, 0])

# =============================================================================
# PART 4 — SEMI-SUPERVISED
# =============================================================================

def self_train(mdl, Xtr, ytr, Xu, thr, maxiter=10, seed=RS):
    # Clean inputs
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
    
    # Clean NaN/inf BEFORE scaling
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte, nan=0, posinf=0, neginf=0)
    Xu = np.nan_to_num(Xu, nan=0, posinf=0, neginf=0)
    
    sc = StandardScaler(); ni = [i for i, f in enumerate(ff) if f in NUMERICAL]
    Xtr2 = Xtr.copy(); Xte2 = Xte.copy(); Xu2 = Xu.copy()
    if ni:
        Xtr2[:, ni] = sc.fit_transform(Xtr[:, ni])
        Xte2[:, ni] = sc.transform(Xte[:, ni])
        if len(Xu2): Xu2[:, ni] = sc.transform(Xu[:, ni])
    
    # Clean again after scaling (in case StandardScaler produced NaN/inf)
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

def patient_split(df, ff):
    u = df["PT_ID"].unique()
    tri, tei = train_test_split(u, test_size=0.2, random_state=RS)
    dtr = df[df["PT_ID"].isin(tri)]; dte = df[df["PT_ID"].isin(tei)]
    return dtr[ff].values, dte[ff].values, dtr["primary_head_label"].values, dte["primary_head_label"].values

def temporal_split(df, ff):
    df = df.sort_values(["PT_ID", "ENCOUNTER_DTTM"], ignore_index=True)
    tr, te = [], []
    for _, g in df.groupby("PT_ID", sort=False):
        si = max(1, int(0.8 * len(g)))
        tr.append(g.iloc[:si]); te.append(g.iloc[si:])
    dtr = pd.concat(tr, ignore_index=True); dte = pd.concat(te, ignore_index=True)
    return dtr[ff].values, dte[ff].values, dtr["primary_head_label"].values, dte["primary_head_label"].values

def std_split(df, ff, kind="row"):
    if kind == "row":
        X = df[ff].values; y = df["primary_head_label"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    elif kind == "patient":
        Xtr, Xte, ytr, yte = patient_split(df, ff)
    else:
        Xtr, Xte, ytr, yte = temporal_split(df, ff)
    
    # Clean NaN/inf BEFORE scaling
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte, nan=0, posinf=0, neginf=0)
    
    Xtr = pd.DataFrame(Xtr, columns=ff); Xte = pd.DataFrame(Xte, columns=ff)
    sc = StandardScaler(); np_ = [c for c in NUMERICAL if c in Xtr.columns]
    Xtr[np_] = sc.fit_transform(Xtr[np_]); Xte[np_] = sc.transform(Xte[np_])
    
    # Clean again after scaling
    return (np.nan_to_num(Xtr.values.astype(np.float32), nan=0, posinf=0, neginf=0),
            np.nan_to_num(Xte.values.astype(np.float32), nan=0, posinf=0, neginf=0), ytr, yte, Xte)

# =============================================================================
# PART 6 — COUNTERFACTUAL  (CORRECTED DENOMINATOR)
# =============================================================================

def _s1p(m, X): return _thr(m.predict_proba(X)[:, 1], S1_THR)
def _s2p(m, X, s1): return _thr(m.predict_proba(X[s1 == 0])[:, 1], S2_THR)
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

def run_cf(Xtr, Xte_df, ytr, yte, ff):
    Xtr_np = np.nan_to_num(np.array(Xtr, dtype=np.float32), nan=0, posinf=0, neginf=0)
    res = {iv: {"cancel": [], "noshow": []} for iv in IVS}
    for seed in range(N_SEEDS):
        np.random.seed(seed); random.seed(seed)
        s1 = DecisionTreeClassifier(random_state=seed)
        s1.fit(Xtr_np, (ytr == 2).astype(int))
        i2 = np.where(ytr != 2)[0]
        s2 = XGBClassifier(random_state=seed, **XGB_KW)
        s2.fit(Xtr_np[i2], (ytr[i2] == 0).astype(int))
        orig = np.nan_to_num(Xte_df[ff].values.astype(np.float32), nan=0, posinf=0, neginf=0)
        s1o = _s1p(s1, orig); s2o = _s2p(s2, orig, s1o)
        nt = len(s1o); oc = np.mean(s1o==1); ons = _nsr(s2o, nt)
        for iv in IVS:
            cfdf = apply_iv(Xte_df, iv, seed=seed)
            cfnp = np.nan_to_num(cfdf[ff].values.astype(np.float32), nan=0, posinf=0, neginf=0)
            s1c = _s1p(s1, cfnp); s2c = _s2p(s2, cfnp, s1c)
            res[iv]["cancel"].append((oc - np.mean(s1c==1))*100)
            res[iv]["noshow"].append((ons - _nsr(s2c, nt))*100)
    return res

# =============================================================================
# PART 7 — LATEX WRITER
# =============================================================================

def _f(v, d=2): return f"{v:.{d}f}"
def _fb(v, d=2): return f"\\textbf{{{v:.{d}f}}}"
def _pm(vs):
    m, s = np.mean(vs), np.std(vs)
    return f"${m:.2f} \\pm {s:.2f}$"

def write_latex(t1, t2, t3, t4, t5, path):
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
    a(r"\caption{Performance of the two-stage pipeline (Decision Tree $\rightarrow$ XGBoost) across no-show cutoff definitions (24h--168h).}")
    a(r"\label{tab:multi_cutoff_results_clean_boldf1}\small")
    a(r"\begin{tabular}{l c c c c c}\toprule")
    a(r"\textbf{Metric} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\")
    a(r"\midrule")
    a(r"\multicolumn{6}{l}{\textbf{Stage 1: Decision Tree (Cancel vs Not-Cancel)}} \\")
    for mn, cls, fld in [("Precision (Cancel)","Cancel","precision"),("Recall (Cancel)","Cancel","recall"),
        ("F1 (Cancel)","Cancel","f1"),("Precision (Not-Cancel)","Not-Cancel","precision"),
        ("Recall (Not-Cancel)","Not-Cancel","recall"),("F1 (Not-Cancel)","Not-Cancel","f1")]:
        bold = "F1" in mn.split("(")[0]
        vs = [(_fb if bold else _f)(t1[c][0][cls][fld]) for c in CUTOFFS]
        a(f"{mn:25s} & " + " & ".join(vs) + r" \\")
    a(f"{'Macro F1':25s} & " + " & ".join([_fb(t1[c][0]["macro_f1"]) for c in CUTOFFS]) + r" \\")
    a(f"{'Accuracy':25s} & " + " & ".join([_f(t1[c][0]["accuracy"]) for c in CUTOFFS]) + r" \\")
    a(f"{'Cancel Support':25s} & " + " & ".join([f'{t1[c][0]["Cancel"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(f"{'Not-Cancel Support':25s} & " + " & ".join([f'{t1[c][0]["Not-Cancel"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(f"{'Total':25s} & " + " & ".join([f'{t1[c][0]["Cancel"]["support"]+t1[c][0]["Not-Cancel"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(r"\midrule\multicolumn{6}{l}{\textbf{Stage 2: XGBoost (Show vs No-Show)}} \\")
    for mn, cls, fld in [("Precision (No-Show)","No-Show","precision"),("Recall (No-Show)","No-Show","recall"),
        ("F1 (No-Show)","No-Show","f1"),("Precision (Show)","Show","precision"),
        ("Recall (Show)","Show","recall"),("F1 (Show)","Show","f1")]:
        bold = "F1" in mn.split("(")[0]
        vs = [(_fb if bold else _f)(t1[c][1][cls][fld]) for c in CUTOFFS]
        a(f"{mn:25s} & " + " & ".join(vs) + r" \\")
    a(f"{'Macro F1':25s} & " + " & ".join([_fb(t1[c][1]["macro_f1"]) for c in CUTOFFS]) + r" \\")
    a(f"{'Accuracy':25s} & " + " & ".join([_f(t1[c][1]["accuracy"]) for c in CUTOFFS]) + r" \\")
    a(f"{'No-Show Support':25s} & " + " & ".join([f'{t1[c][1]["No-Show"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(f"{'Show Support':25s} & " + " & ".join([f'{t1[c][1]["Show"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(f"{'Total':25s} & " + " & ".join([f'{t1[c][1]["No-Show"]["support"]+t1[c][1]["Show"]["support"]:,}' for c in CUTOFFS]) + r" \\")
    a(r"\bottomrule\end{tabular}")
    a(r"\begin{flushleft}\footnotesize \textit{Note: F1 scores are in \textbf{bold}.}\end{flushleft}\end{table*}")
    a("")

    # TABLE 2
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{Semi-supervised Learning Sensitivity Analysis: No-Show Reason Prediction.}")
    a(r"\label{tab:reason_prediction_sensitivity_final}\small")
    a(r"\begin{tabularx}{\textwidth}{l l CC CC CC}\toprule")
    a(r"\multirow{2}{*}{\textbf{Cutoff}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{No Pseudo-Labels}} & \multicolumn{2}{c}{\textbf{Pseudo-Labels (0.60)}} & \multicolumn{2}{c}{\textbf{Pseudo-Labels (0.85)}} \\")
    a(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    a(r"& & \textbf{Acc.} & \textbf{F1} & \textbf{Acc.} & \textbf{F1} & \textbf{Acc.} & \textbf{F1} \\\midrule")
    mns = ["DecisionTree","XGBoost","MLP"]
    conds = ["No Pseudo-Labels","Pseudo-Labels (0.60)","Pseudo-Labels (0.85)"]
    for ci, c in enumerate(CUTOFFS):
        af1 = [t2[c].get((m, co), (0,0))[1] for m in mns for co in conds]
        bf1 = max(af1)
        a(f"\\multirow{{3}}{{*}}{{\\textbf{{{c}h}}}} ")
        for m in mns:
            cells = []
            for co in conds:
                ac, f = t2[c].get((m, co), (0, 0))
                cells.append(f"{(_fb if f==bf1 and f>0 else _f)(ac)} & {(_fb if f==bf1 and f>0 else _f)(f)}")
            a(f"& {m:12s} & " + " & ".join(cells) + r" \\")
        if ci < len(CUTOFFS)-1: a(r"\midrule")
    a(r"\bottomrule\end{tabularx}\end{table*}")
    a("")

    # TABLE 3
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{Model performance on predicting reasons for no-shows and cancellations.}\small")
    a(r"\begin{tabular}{lcccc}\toprule")
    a(r"\textbf{Model} & \textbf{No-Show Acc} & \textbf{No-Show F1} & \textbf{Cancel Acc} & \textbf{Cancel F1} \\\midrule")
    for c in CUTOFFS:
        a(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{c}h}}}} \\\\")
        bf = max(t3[c][m][1] for m in mns)
        for m in mns:
            na, nf, ca, cf_ = t3[c][m]
            b = nf == bf and nf > 0
            a(f"{m:12s} & {(_fb if b else _f)(na,3)} & {(_fb if b else _f)(nf,3)} & {(_fb if b else _f)(ca,3)} & {(_fb if b else _f)(cf_,3)} \\\\")
        a(r"\midrule")
    L[-1] = r"\bottomrule"
    a(r"\end{tabular}\label{tab:final_reason_prediction_vertical}\end{table*}")
    a("")

    # TABLE 4
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{F1-scores across data splitting strategies.}")
    a(r"\label{tab:split_f1_results_sensitivity}\small")
    a(r"\begin{tabularx}{\textwidth}{l l CCCCC}\toprule")
    a(r"\textbf{Splitting Method} & \textbf{Outcome Class} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\\midrule")
    sps = ["Row-Wise (Random)","Patient-Level","Appointment-Time"]
    for si, sp in enumerate(sps):
        a(f"\\multirow{{3}}{{*}}{{{sp}}} ")
        for nm, ci in [("Cancellations",0),("No-Shows",1),("Shows",2)]:
            vs = [_f(t4[c][sp][ci]) for c in CUTOFFS]
            a(f"& {nm:14s} & " + " & ".join(vs) + r" \\")
        if si < len(sps)-1: a(r"\midrule")
    a(r"\bottomrule\end{tabularx}\end{table*}")
    a("")

    # TABLE 5
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{Counterfactual Analysis Sensitivity (CORRECTED): Absolute change in cancellation and no-show rates (pp).}")
    a(r"\label{tab:counterfactual_sensitivity_final}\small")
    a(r"\begin{tabularx}{\textwidth}{l l c c c c c}\toprule")
    a(r"\multirow{2}{*}{\textbf{Intervention}} & \multirow{2}{*}{\textbf{Metric}} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\")
    a(r"\cmidrule(lr){3-7}& & \multicolumn{5}{c}{\textit{(Mean $\pm$ SD across 10 model seeds)}} \\\midrule")
    for ii, iv in enumerate(IVS):
        a(f"\\multirow{{2}}{{*}}{{{iv}}} ")
        a("& $\\Delta$ Cancel (\\%) & " + " & ".join([_pm(t5[c][iv]["cancel"]) for c in CUTOFFS]) + r" \\")
        a("& $\\Delta$ No-show (\\%) & " + " & ".join([_pm(t5[c][iv]["noshow"]) for c in CUTOFFS]) + r" \\")
        if ii < len(IVS)-1: a(r"\midrule")
    a(r"\bottomrule\end{tabularx}")
    a(r"\begin{flushleft}\footnotesize \textit{Note: $\Delta$ = absolute pp reduction. Positive = improvement.}\end{flushleft}\end{table*}")
    a(""); a(r"\end{document}")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L))

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 64)
    print("  HCMS Hierarchical ML Pipeline  (single run, corrected)")
    print("=" * 64)

    # ── load raw data ────────────────────────────────────────────────────
    phase("Loading df_with_valid_indices_0717.csv")
    raw = pd.read_csv(os.path.join(DIR, "df_with_valid_indices_0717.csv"))
    step(f"{len(raw):,} rows loaded")

    # ── fast feature engineering ─────────────────────────────────────────
    df_all, ff = fast_feature_engineering(raw)

    # ── results containers ───────────────────────────────────────────────
    t1, t2, t3, t4, t5 = {}, {}, {}, {}, {}

    for ci, cutoff in enumerate(CUTOFFS):
        phase(f"Cutoff {cutoff}h  ({ci+1}/{len(CUTOFFS)})")
        df = build_labels(df_all.copy(), cutoff)

        # Table 1
        step("Table 1  pipeline (DT -> XGB)")
        Xtr, Xte, ytr, yte, Xte_df = std_split(df, ff, "row")
        s1, s2, m1, m2 = train_pipeline(Xtr, Xte, ytr, yte)
        t1[cutoff] = (m1, m2)

        # Table 4
        step("Table 4  split strategies")
        t4[cutoff] = {}
        t4[cutoff]["Row-Wise (Random)"] = combined_f1(s1, s2, Xte, yte)
        Xtrp, Xtep, ytrp, ytep, _ = std_split(df, ff, "patient")
        s1p, s2p, _, _ = train_pipeline(Xtrp, Xtep, ytrp, ytep)
        t4[cutoff]["Patient-Level"] = combined_f1(s1p, s2p, Xtep, ytep)
        Xtrt, Xtet, ytrt, ytet, _ = std_split(df, ff, "temporal")
        s1t, s2t, _, _ = train_pipeline(Xtrt, Xtet, ytrt, ytet)
        t4[cutoff]["Appointment-Time"] = combined_f1(s1t, s2t, Xtet, ytet)

        # Tables 2 & 3
        step("Table 2  semi-supervised")
        Xb = df[ff].values
        ns_f = (df["STATUS_CD"].isin(["No Show","Left without seen"])) | (df["cancelled_within"]==1)
        Xns = Xb[ns_f]; yns = df.loc[ns_f, "noshow_reason_label"].values.astype(float)
        t2[cutoff] = {}
        for cn, tv in [("No Pseudo-Labels",None),("Pseudo-Labels (0.60)",0.60),("Pseudo-Labels (0.85)",0.85)]:
            r = run_semi_sup(Xns, yns, ff, thr=tv)
            for mn, v in r.items(): t2[cutoff][(mn, cn)] = v

        step("Table 3  reason prediction")
        ca_f = (df["STATUS_CD"]=="Canceled") & (df["cancelled_within"]==0)
        Xca = Xb[ca_f]; yca = df.loc[ca_f, "cancellation_reason_label"].values.astype(float)
        ok = ~np.isnan(yca); Xca, yca = Xca[ok], yca[ok]
        t3[cutoff] = {}
        ns_r = run_semi_sup(Xns, yns, ff, thr=0.60)
        ca_r = run_semi_sup(Xca, yca, ff, thr=None)
        for mn in ["DecisionTree","XGBoost","MLP"]:
            na, nf = ns_r.get(mn, (0,0)); ca_, cf_ = ca_r.get(mn, (0,0))
            t3[cutoff][mn] = (na, nf, ca_, cf_)

        # Table 5
        step("Table 5  counterfactual (10 seeds)")
        t5[cutoff] = run_cf(Xtr, Xte_df, ytr, yte, ff)

    # ── export ───────────────────────────────────────────────────────────
    phase("Exporting LaTeX")
    out = os.path.join(DIR, "revised_tables_corrected.tex")
    write_latex(t1, t2, t3, t4, t5, out)
    step(f"Saved: {out}")
    print(f"\n{'='*64}\n  DONE\n{'='*64}\n")

if __name__ == "__main__":
    main()
