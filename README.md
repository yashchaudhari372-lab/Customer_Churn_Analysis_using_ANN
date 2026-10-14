# Customer Churn Analysis using ANN | Deep Learning & Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Live-brightgreen)

An end-to-end **Customer Churn Prediction and Analytics System** powered by an **Artificial Neural Network (ANN)**. The project combines deep learning, customer analytics, interactive visualizations, and predictive modeling to estimate customer churn risk and support data-driven retention strategies.

**[🚀 Try the Live Demo](https://customerchurnanalysisusingann1-aqtdqfhsuti2lcp2mgfgxp.streamlit.app/)** · **[💻 View Repository](https://github.com/yashchaudhari372-lab/Customer_Churn_Analysis_using_ANN)**

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting potential churn helps businesses understand retention challenges and prioritize proactive customer engagement.

This project uses the **Churn Modelling dataset** to develop an ANN-based binary classification model that predicts whether a customer is likely to leave a bank.

The application integrates a professional Streamlit dashboard, customer analytics, interactive Plotly visualizations, churn probability prediction, and model performance monitoring.

## Key Features

- **ANN-based customer churn prediction** using TensorFlow and Keras.
- **Interactive Streamlit dashboard** for customer analytics and predictive insights.
- **Churn probability estimation** based on individual customer information.
- **Customer analytics** covering geography, age, and product ownership.
- **KPI dashboard** showing total customers, churned customers, churn rate, and retention rate.
- **Six professional themes:** Aurora, Ocean, Sunset, Emerald, Royal, and Light.
- **Interactive Plotly charts** for exploring churn patterns.
- **Feature preprocessing** using standard scaling and one-hot encoding.
- **Stratified train/validation/test split** for model evaluation.
- **Early stopping and threshold optimization** using validation data.
- **Saved model and preprocessing artifacts** for reproducible inference.

## Tech Stack

| Area | Technology |
| --- | --- |
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Deep learning | TensorFlow, Keras |
| Machine learning | Scikit-learn |
| Visualization | Plotly |
| Web application | Streamlit |
| Model | Artificial Neural Network |
| Preprocessing | StandardScaler, OneHotEncoder |
| Model storage | Keras, Joblib, JSON |
| Development | VS Code, Git, GitHub |
| Deployment | Streamlit Community Cloud |

## Dataset Information

The project uses the **Churn Modelling dataset**, containing banking customer information and a binary churn target.

| Feature | Description |
| --- | --- |
| CreditScore | Customer credit score |
| Geography | Customer location |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of banking products |
| HasCrCard | Credit card ownership |
| IsActiveMember | Customer activity status |
| EstimatedSalary | Estimated annual salary |
| Exited | Target: 1 = Churned, 0 = Retained |

Customer identifiers are excluded from training.

## Machine Learning Workflow

**Dataset → Feature Selection → Stratified Split → Preprocessing → ANN Training → Threshold Optimization → Test Evaluation → Model Saving → Streamlit Deployment**

The model uses a **70% training, 15% validation, and 15% testing** split. Numeric features are scaled, categorical features are one-hot encoded, and preprocessing is fitted only on training data.

## ANN Model Architecture

| Layer | Configuration |
| --- | --- |
| Input | Preprocessed customer features |
| Hidden Layer 1 | Dense (32 neurons, ReLU) |
| Regularization | Dropout (0.20) |
| Hidden Layer 2 | Dense (16 neurons, ReLU) |
| Output | Dense (1 neuron, Sigmoid) |

**Training Configuration**

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Binary Crossentropy
- Maximum epochs: 100
- Batch size: 32
- Early stopping: Validation loss, patience 12
- Random seed: 42

The Sigmoid output estimates a churn probability between 0 and 1. A classification threshold is selected using validation-set F1-score.

## Dashboard Modules

### 1. Overview

Displays total customers, churned customers, churn rate, retention rate, customer churn distribution, and geographic churn comparison.

### 2. Customer Analytics

Provides geographic filtering, age-based churn analysis, product ownership patterns, and an interactive customer data table.

### 3. Predict Churn

Allows users to enter customer information and generate a churn probability using the trained ANN model. The application displays a higher- or lower-risk assessment based on the selected threshold.

### 4. Model Performance

Displays held-out test accuracy, precision, recall, F1-score, ROC-AUC, test sample count, and decision threshold.

## Repository Structure

```text
Customer_Churn_Analysis_using_ANN/
├── data/
│   └── Churn_Modelling.csv
├── models/
│   ├── churn_ann.keras
│   ├── preprocessor.joblib
│   └── metrics.json
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/yashchaudhari372-lab/Customer_Churn_Analysis_using_ANN.git
cd Customer_Churn_Analysis_using_ANN
```

**2. Create a virtual environment**

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

**
