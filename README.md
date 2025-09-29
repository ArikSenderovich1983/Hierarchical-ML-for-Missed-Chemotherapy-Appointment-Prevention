# Patient Appointment Behavior Prediction

This repository contains a complete pipeline for predicting **patient appointment behaviors** (cancellations and no-shows), including feature engineering, semi-supervised learning, explainability, and counterfactual simulation experiments.

---

## Features
- Automated ingestion of multiple `.txt` files from Google Drive  
- Duplicate removal and patient-level history construction  
- Advanced **feature engineering** with temporal and categorical variables  
- Cascaded ML pipeline (Stage 1: Cancellation, Stage 2: Show vs No-show)  
- Semi-supervised **self-training** with pseudo-labels (thresholds 0.60 & 0.85)  
- Explainability: SHAP, SHAPIQ, permutation importance, mutual information  
- **Counterfactual simulations** for interventions:
  - Commitment slot scheduling  
  - Provider/clinic consistency  
  - Confirmation doubling  
  - Scheduling time adjustments  
  - Literature-based strategies  

---

## Repository Structure
- `step1_data_ingestion.py` → Importing `.txt` files, cleaning, duplicate removal, patient history assembly  
- `step2_feature_engineering.py` → Temporal + categorical feature construction, final feature set  
- `step3_hierarchical_pipeline.py` → Hierarchical pipeline combining Stage 1 (cancellation) and Stage 2 (show vs no-show)  
- `step4_semi_supervised.py` → Self-training with Random Forest & XGBoost, threshold sweeps & pseudo-labels  
- `step5_explainability.py` → SHAPIQ, SHAP, permutation importance, mutual information  
- `step6_counterfactual.py` → Simulation of interventions to reduce no-shows  

---

## Requirements
To install dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage
Run each step in order, or integrate them into a pipeline:

```bash
python step1_data_ingestion.py
python step2_feature_engineering.py
python step3_hierarchical_pipeline.py
python step4_semi_supervised.py
python step5_explainability.py
python step6_counterfactual.py
```

---

## Citation
If you use this code in academic work, please cite accordingly.
