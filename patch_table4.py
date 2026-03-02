"""
patch_table4.py — Fix Table 4 using per-stage F1 instead of 3-class F1.
Row-Wise values come from Table 1 (already correct). Patient-Level and
Appointment-Time are retrained here.
"""
import os, sys, time, warnings, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from run_full_pipeline import (
    fast_feature_engineering, build_labels, std_split,
    train_pipeline, CUTOFFS, _f, phase, step
)

DIR = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(DIR, "revised_tables_corrected.tex")

def main():
    phase("Loading + feature engineering")
    raw = pd.read_csv(os.path.join(DIR, "df_with_valid_indices_0717.csv"))
    df_all, ff = fast_feature_engineering(raw)

    # Row-Wise values come straight from Table 1 (already per-stage F1)
    # We only need to retrain Patient-Level and Appointment-Time
    t4 = {}
    for ci, cutoff in enumerate(CUTOFFS):
        phase(f"Cutoff {cutoff}h  ({ci+1}/{len(CUTOFFS)})")
        df = build_labels(df_all.copy(), cutoff)

        # Row-Wise: train to get per-stage metrics
        step("Row-Wise split")
        Xtr, Xte, ytr, yte, _ = std_split(df, ff, "row")
        _, _, m1r, m2r = train_pipeline(Xtr, Xte, ytr, yte)

        step("Patient-Level split")
        Xtrp, Xtep, ytrp, ytep, _ = std_split(df, ff, "patient")
        _, _, m1p, m2p = train_pipeline(Xtrp, Xtep, ytrp, ytep)

        step("Appointment-Time split")
        Xtrt, Xtet, ytrt, ytet, _ = std_split(df, ff, "temporal")
        _, _, m1t, m2t = train_pipeline(Xtrt, Xtet, ytrt, ytet)

        t4[cutoff] = {
            "Row-Wise (Random)":  (m1r["Cancel"]["f1"], m2r["No-Show"]["f1"], m2r["Show"]["f1"]),
            "Patient-Level":      (m1p["Cancel"]["f1"], m2p["No-Show"]["f1"], m2p["Show"]["f1"]),
            "Appointment-Time":   (m1t["Cancel"]["f1"], m2t["No-Show"]["f1"], m2t["Show"]["f1"]),
        }
        for sp in t4[cutoff]:
            c, n, s = t4[cutoff][sp]
            step(f"  {sp:25s}  Cancel={c:.2f}  No-Show={n:.2f}  Show={s:.2f}")

    # Build replacement Table 4 block
    phase("Patching LaTeX")
    lines = []
    a = lines.append
    a(r"\begin{table*}[t]\centering")
    a(r"\caption{F1-scores across data splitting strategies.}")
    a(r"\label{tab:split_f1_results_sensitivity}\small")
    a(r"\begin{tabularx}{\textwidth}{l l CCCCC}\toprule")
    a(r"\textbf{Splitting Method} & \textbf{Outcome Class} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\\midrule")
    sps = ["Row-Wise (Random)", "Patient-Level", "Appointment-Time"]
    for si, sp in enumerate(sps):
        a(f"\\multirow{{3}}{{*}}{{{sp}}} ")
        for nm, ci in [("Cancellations", 0), ("No-Shows", 1), ("Shows", 2)]:
            vs = [_f(t4[c][sp][ci]) for c in CUTOFFS]
            a(f"& {nm:14s} & " + " & ".join(vs) + r" \\")
        if si < len(sps) - 1:
            a(r"\midrule")
    a(r"\bottomrule\end{tabularx}\end{table*}")
    new_table4 = "\n".join(lines)

    # Read existing tex, replace Table 4 block
    with open(TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    pattern = r"\\begin\{table\*\}\[t\]\\centering\n\\caption\{F1-scores across data splitting strategies\.\}.*?\\end\{table\*\}"
    tex_new = re.sub(pattern, new_table4, tex, flags=re.DOTALL)

    with open(TEX, "w", encoding="utf-8") as f:
        f.write(tex_new)

    step(f"Patched {TEX}")
    print(f"\n{'='*64}\n  DONE — Table 4 fixed\n{'='*64}\n")

if __name__ == "__main__":
    main()
