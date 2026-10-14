# Customer Churn Intelligence — ANN

Professional Streamlit analytics and prediction dashboard using the Churn Modelling dataset.

## Setup

Use Python 3.11 (recommended) in VS Code.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deployment

Train locally, then commit `models/churn_ann.keras`, `models/preprocessor.joblib`, `models/metrics.json`, `data/Churn_Modelling.csv`, and the source files to a GitHub repository. In Streamlit Community Cloud choose the repository, branch and `app.py` entry point. Select Python 3.11 if available. Ensure that installed TensorFlow has a compatible wheel for the selected Python runtime.

## ML integrity

Stratified 70/15/15 split; preprocessing fit only on training data; early stopping using validation; F1 threshold tuned on validation; final metrics evaluated once on the held-out test set. Model predictions are estimates, not causal explanations.
