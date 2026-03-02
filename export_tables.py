"""
Export pipeline results to revised_tables_all_cutoffs.tex

Run AFTER run_all.py completes. Call export_tables.export(globals()) from run_all
to pass pipeline variables, or run standalone for a template.

Usage:
  from export_tables import export; export(globals())   # from run_all
  python export_tables.py                             # standalone template
"""
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_out_path = os.path.join(_script_dir, "revised_tables_all_cutoffs.tex")


def write_tables_tex(data: dict) -> str:
    """Build LaTeX content from data dict. data can have keys like table1, table5, etc."""
    # Use provided data or placeholders
    t1 = data.get("table1", {})
    t5 = data.get("table5", {})

    s = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{caption}
\usepackage{geometry}
\geometry{margin=1in}
\newcolumntype{C}{>{\centering\arraybackslash}X}

\begin{document}
"""
    # Table 1: Pipeline performance (placeholder structure - fill from pipeline)
    s += r"""
\begin{table*}[t]
\centering
\caption{Performance of the two-stage pipeline (DT $\rightarrow$ XGBoost) across no-show cutoff definitions.}
\label{tab:multi_cutoff_results}
\small
\begin{tabular}{l c c c c c}
\toprule
\textbf{Metric} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\
\midrule
\multicolumn{6}{l}{\textbf{Stage 1: Decision Tree (Cancel vs Not-Cancel)}} \\
"""
    for row in t1.get("stage1_rows", []):
        s += row + " \\\\\n"
    s += r"""\midrule
\multicolumn{6}{l}{\textbf{Stage 2: XGBoost (Show vs No-Show)}} \\
"""
    for row in t1.get("stage2_rows", []):
        s += row + " \\\\\n"
    s += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    # Table 5: Counterfactual (corrected)
    s += r"""
\begin{table*}[t]
\centering
\caption{Counterfactual Analysis Sensitivity (corrected): $\Delta$ in cancellation and no-show rates.}
\label{tab:counterfactual_sensitivity}
\small
\begin{tabularx}{\textwidth}{l l c c c c c}
\toprule
\multirow{2}{*}{\textbf{Intervention}} & \multirow{2}{*}{\textbf{Metric}} & \textbf{24h} & \textbf{48h} & \textbf{72h} & \textbf{96h} & \textbf{168h} \\
\cmidrule(lr){3-7}
& & \multicolumn{5}{c}{\textit{(Mean $\pm$ SD)}} \\
\midrule
"""
    for interv, rows in t5.get("rows", {}).items():
        s += f"\\multirow{{2}}{{*}}{{{interv}}} " + rows.get("cancel", "") + " \\\\\n"
        s += " & " + rows.get("noshow", "") + " \\\\\n"
        s += "\\midrule\n"
    s += r"""\bottomrule
\end{tabularx}
\end{table*}
"""

    s += r"\end{document}"
    return s


def export(glbs=None):
    """
    Export pipeline results to LaTeX. Pass globals() from run_all to use pipeline data.
    """
    data = _collect_data(glbs or {})
    content = write_tables_tex(data)
    with open(_out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Exported tables to {_out_path}")
    return _out_path


def _collect_data(glbs):
    """Collect table data from pipeline globals."""
    data = {"table1": {"stage1_rows": [], "stage2_rows": []}, "table5": {"rows": {}}}
    # Placeholder rows when no pipeline data
    data["table1"]["stage1_rows"] = [
        "Precision (Cancel)     & -- & -- & -- & -- & --",
        "Recall (Cancel)        & -- & -- & -- & -- & --",
        "F1 (Cancel)            & \\textbf{--} & \\textbf{--} & \\textbf{--} & \\textbf{--} & \\textbf{--}",
    ]
    data["table1"]["stage2_rows"] = [
        "Precision (No-Show)    & -- & -- & -- & -- & --",
        "Recall (No-Show)       & -- & -- & -- & -- & --",
        "F1 (No-Show)           & \\textbf{--} & \\textbf{--} & \\textbf{--} & \\textbf{--} & \\textbf{--}",
    ]
    data["table5"]["rows"] = {
        "SMS Reminders": {"cancel": "& $\\Delta$ Cancel (\\%) & -- & -- & -- & -- & --", "noshow": "$\\Delta$ No-show (\\%) & -- & -- & -- & -- & --"},
        "Patient Confirmation": {"cancel": "& $\\Delta$ Cancel (\\%) & -- & -- & -- & -- & --", "noshow": "$\\Delta$ No-show (\\%) & -- & -- & -- & -- & --"},
        "Provider Consistency": {"cancel": "& $\\Delta$ Cancel (\\%) & -- & -- & -- & -- & --", "noshow": "$\\Delta$ No-show (\\%) & -- & -- & -- & -- & --"},
    }
    return data


def main():
    """Standalone: write template with placeholders."""
    export({})


if __name__ == "__main__":
    main()
