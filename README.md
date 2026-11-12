# 🧠 Customer Churn Intelligence — ANN | Deep Learning & Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-yellow?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualization-3F4F75?logo=plotly)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

An end-to-end **Customer Churn Analysis and Prediction System** powered by an **Artificial Neural Network (ANN)**. This project combines deep learning, customer analytics, interactive visualizations, and predictive modeling to identify customers at risk of leaving a bank.

The application provides an interactive Streamlit dashboard for exploring customer behavior, analyzing churn patterns, predicting individual customer churn probabilities, and evaluating model performance..

**🚀 Live Demo:** [Customer Churn Intelligence Dashboard](https://customerchurnanalysisusingann1-aqtdqfhsuti2lcp2mgfgxp.streamlit.app/)

**💻 GitHub Repository:** [Customer_Churn_Analysis_using_ANN](https://github.com/yashchaudhari372-lab/Customer_Churn_Analysis_using_ANN)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Dataset Description](#-dataset-description)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [ANN Model Architecture](#-ann-model-architecture)
- [Model Evaluation](#-model-evaluation)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Model Training](#-model-training)
- [Deployment](#-deployment)
- [Business Applications](#-business-applications)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Project Overview

Customer churn occurs when customers discontinue their relationship with a business. In the banking industry, customer churn can result in lost revenue, reduced customer lifetime value, and increased customer acquisition costs.

This project uses an **Artificial Neural Network (ANN)** to estimate the probability that a bank customer will leave based on demographic information and account-related characteristics.

The system integrates a deep learning model with an interactive analytics dashboard, allowing users to explore customer data and generate predictions using customized customer information.

The application is developed using Python, TensorFlow/Keras, Scikit-learn, Pandas, Plotly, and Streamlit.

### What Does This Project Do?

- Analyzes historical bank customer information.
- Calculates overall customer churn and retention rates.
- Visualizes churn patterns across different customer groups.
- Explores relationships between customer age, geography, and banking products.
- Predicts customer churn probability using an ANN model.
- Categorizes predictions into higher or lower churn risk using a validation-selected decision threshold.
- Displays model evaluation metrics to help users understand predictive performance.

The goal is to demonstrate how deep learning and interactive analytics can support data-driven customer retention strategies.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. **Customer Churn Prediction:** Develop an ANN-based binary classification model to predict whether a customer is likely to leave a bank.
2. **Exploratory Customer Analytics:** Analyze customer demographics, account characteristics, and churn distributions.
3. **Data Preprocessing:** Apply appropriate feature scaling and categorical encoding to prepare customer data for neural network training.
4. **Model Optimization:** Use validation data to select a decision threshold that maximizes the F1-score.
5. **Model Evaluation:** Evaluate the trained model using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Interactive Visualization:** Develop a user-friendly Streamlit dashboard for exploring customer data and model results.
7. **Practical Deployment:** Make the application accessible through a live web interface.

---

## ✨ Key Features

### 1. Interactive Overview Dashboard

The overview page provides a summary of the customer dataset, including:

- Total number of customers.
- Number of customers who have churned.
- Overall customer churn rate.
- Overall customer retention rate.
- Customer churn distribution visualization.
- Geographic churn comparison.

These metrics provide an overview of the dataset and help users understand the scale of the churn problem.

### 2. Customer Analytics

The Customer Analytics section enables users to explore customer behavior through interactive visualizations.

Features include:

- Geography-based filtering.
- Customer age distribution analysis.
- Comparison of churn behavior across age groups.
- Analysis of churn by the number of banking products.
- Interactive customer data table.
- Exploration of customer characteristics associated with churn.

Plotly visualizations help users investigate patterns within the available dataset.

### 3. ANN-Based Churn Prediction

The prediction module allows users to enter customer information and estimate the probability of churn.

The application accepts the following inputs:

- Credit score.
- Geography.
- Gender.
- Age.
- Customer tenure.
- Account balance.
- Number of banking products.
- Credit card ownership.
- Active membership status.
- Estimated salary.

After submission, the application processes the customer information using the saved preprocessing pipeline and trained neural network.

The prediction interface displays:

- Estimated churn probability.
- Visual probability indicator.
- Higher predicted churn risk or lower predicted churn risk.
- A risk message based on the selected decision threshold.

**Note:** Predictions are statistical estimates and should not be treated as guarantees that a customer will leave or remain with the bank.

### 4. Model Performance Dashboard

The Model Performance section displays evaluation metrics calculated on the held-out test dataset.

The metrics include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Number of test samples.
- Selected classification threshold.

These metrics provide a broader view of model performance than accuracy alone.

### 5. Customizable Dashboard Themes

The application includes multiple color themes to customize the dashboard's appearance.

Available themes include:

- Aurora.
- Ocean.
- Sunset.
- Emerald.
- Royal.
- Light.

### 6. Automated Model Training

The project includes a dedicated training script that:

- Loads the customer dataset.
- Splits the data into training, validation, and test sets.
- Fits the preprocessing pipeline using training data.
- Trains the ANN model.
- Applies early stopping.
- Selects a decision threshold using validation predictions.
- Evaluates the model on the held-out test set.
- Saves the trained model, preprocessing pipeline, and evaluation metrics.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Deep Learning Framework | TensorFlow / Keras |
| Machine Learning | Scikit-learn |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Plotly Express |
| Web Application | Streamlit |
| Data Preprocessing | StandardScaler, OneHotEncoder, ColumnTransformer |
| Model Persistence | Keras model format, Joblib |
| Model Evaluation | Scikit-learn metrics |
| Version Control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📊 Dataset Description

The project uses the **Churn Modelling dataset**, containing bank customer information and the target variable used to identify customer churn.

The application loads the dataset from:

`data/Churn_Modelling.csv`

### Dataset Features

| Feature | Description |
|---|---|
| RowNumber | Row identifier in the dataset |
| CustomerId | Unique customer identifier |
| Surname | Customer surname |
| CreditScore | Customer credit score |
| Geography | Customer's country or geographical category |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years the customer has been with the bank |
| Balance | Customer's bank account balance |
| NumOfProducts | Number of banking products used |
| HasCrCard | Indicates whether the customer has a credit card |
| IsActiveMember | Indicates whether the customer is an active member |
| EstimatedSalary | Estimated customer salary |
| Exited | Target variable indicating customer churn |

### Target Variable

The model uses `Exited` as the target variable.

| Value | Meaning |
|---|---|
| `0` | Customer did not churn |
| `1` | Customer churned |

### Input Features Used for Training

The model uses the following ten original customer features:

```text
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
```

The `RowNumber`, `CustomerId`, and `Surname` columns are not included in the model's training features.

The categorical features, Geography and Gender, are encoded during preprocessing, while numerical features are standardized.

---

## 🔬 Machine Learning Workflow

The project follows a structured machine learning pipeline.

### Step 1: Data Loading

The customer dataset is loaded using Pandas.

The target variable is separated from the input features before training.

### Step 2: Train, Validation, and Test Split

The dataset is divided into three subsets using stratified sampling.

| Dataset | Proportion | Purpose |
|---|---:|---|
| Training Set | 70% | Train the neural network and fit preprocessing |
| Validation Set | 15% | Monitor training and select the decision threshold |
| Test Set | 15% | Evaluate final model performance |

The split uses a fixed random seed to improve reproducibility.

Stratification helps preserve the target-class proportions across the subsets.

### Step 3: Feature Preprocessing

The project uses Scikit-learn's `ColumnTransformer` to apply different preprocessing operations to different feature types.

**Numerical Features**

`StandardScaler` standardizes numerical features to improve the suitability of the inputs for neural network training.

**Categorical Features**

`OneHotEncoder` converts the categorical Geography and Gender features into numerical representations.

The encoder uses `handle_unknown='ignore'` so that previously unseen categories can be handled during transformation.

**Data Leakage Prevention**

The preprocessing pipeline is fitted only on the training dataset. The validation and test datasets are transformed using the already-fitted pipeline.

This prevents information from the validation and test sets from influencing the preprocessing parameters learned during training.

### Step 4: ANN Model Training

The preprocessed training data is passed to a feedforward Artificial Neural Network.

The network learns relationships between customer characteristics and the target churn variable.

### Step 5: Early Stopping

The model uses an EarlyStopping callback that monitors validation loss.

Training stops when validation loss fails to improve for the configured patience period, and the best-performing weights are restored.

This helps reduce unnecessary training and may limit overfitting.

### Step 6: Decision Threshold Selection

The neural network produces a probability between 0 and 1.

Instead of assuming that a threshold of 0.50 is always optimal, the training script searches candidate thresholds using validation predictions.

The threshold that maximizes the validation F1-score is selected.

This threshold is then used to convert predicted probabilities into binary churn classifications.

### Step 7: Model Evaluation

The final model is evaluated on the held-out test dataset.

The training script calculates accuracy, precision, recall, F1-score, and ROC-AUC.

The resulting metrics and decision threshold are saved for use by the dashboard.

### Step 8: Model Deployment

The trained model and preprocessing pipeline are loaded by the Streamlit application to provide predictions through the web interface.

---

## 🧠 ANN Model Architecture

The project uses a feedforward Artificial Neural Network implemented with TensorFlow/Keras.

### Network Architecture

```text
Customer Input Features
          |
          v
Feature Preprocessing
          |
          v
Input Layer
          |
          v
Dense Layer — 32 Neurons
Activation: ReLU
          |
          v
Dropout Layer — 20%
          |
          v
Dense Layer — 16 Neurons
Activation: ReLU
          |
          v
Output Layer — 1 Neuron
Activation: Sigmoid
          |
          v
Predicted Churn Probability
          |
          v
Validation-Selected Threshold
          |
          v
Higher Risk / Lower Risk
```

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model Type | Feedforward Artificial Neural Network |
| Framework | TensorFlow / Keras |
| First Hidden Layer | 32 neurons |
| First Hidden Activation | ReLU |
| Dropout Rate | 0.20 |
| Second Hidden Layer | 16 neurons |
| Second Hidden Activation | ReLU |
| Output Layer | 1 neuron |
| Output Activation | Sigmoid |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 100 |
| Early Stopping Patience | 12 epochs |
| Model Selection | Validation loss |
| Classification Threshold | Selected using validation F1-score |

### Why Use an ANN?

An Artificial Neural Network can learn nonlinear relationships between customer characteristics and churn outcomes.

For example, customer age, account balance, product usage, and account activity may interact in ways that are not adequately described by simple linear relationships.

The ANN attempts to learn such patterns from historical data.

The sigmoid output layer produces a probability estimate that can be converted into a binary classification using the selected threshold.

---

## 📈 Model Evaluation

The model is evaluated using multiple classification metrics.

### 1. Accuracy

Accuracy measures the proportion of all test observations classified correctly.

It provides a general measure of classification performance but may be insufficient when the churn and non-churn classes are imbalanced.

### 2. Precision

Precision measures the proportion of customers predicted to churn who actually churned.

Higher precision means fewer false-positive churn predictions among customers identified as potential churners.

### 3. Recall

Recall measures the proportion of actual churners correctly identified by the model.

Recall is particularly relevant when the objective is to identify as many at-risk customers as possible.

### 4. F1-Score

The F1-score combines precision and recall into a single metric using their harmonic mean.

It is useful when both false-positive and false-negative predictions matter.

### 5. ROC-AUC

ROC-AUC measures how effectively the model distinguishes between churn and non-churn customers across different classification thresholds.

A higher ROC-AUC generally indicates stronger class discrimination.

### Evaluation Strategy

The project follows these practices:

- Stratified dataset splitting.
- Training-only preprocessing fitting.
- Validation-based early stopping.
- Validation-based decision threshold selection.
- Final evaluation on an independent test set.

The actual performance values are generated by `train_model.py` and saved in `models/metrics.json`.

**The accuracy, precision, recall, F1-score, and ROC-AUC values are not hardcoded in this README.** Run the training script to generate the metrics for your current model and dataset.

---

## 📊 Interactive Dashboard

The application provides four main sections.

| Dashboard Section | Purpose |
|---|---|
| Overview | Displays customer counts, churn rate, retention rate, and churn distribution |
| Customer Analytics | Explores age, geography, and banking product patterns |
| Predict Churn | Estimates the churn probability of an individual customer |
| Model Performance | Displays held-out test metrics and the selected decision threshold |

### Customer Analytics

Users can filter customer data by geography and examine visualizations that compare churn outcomes across customer ages and product counts.

### Predict Churn

Users can adjust customer characteristics and submit them to the trained ANN.

The application transforms the input using the saved preprocessing pipeline before generating the probability estimate.

### Model Performance

The dashboard reads the saved model evaluation metrics and presents them in an easy-to-understand format.

**Try the live application:** [Open Customer Churn Intelligence](https://customerchurnanalysisusingann1-aqtdqfhsuti2lcp2mgfgxp.streamlit.app/)

---

## 📁 Project Structure

```text
Customer_Churn_Analysis_using_ANN/
│
├── data/
│   └── Churn_Modelling.csv
│       # Bank customer dataset
│
├── models/
│   ├── churn_ann.keras
│   │   # Trained ANN model
│   │
│   ├── preprocessor.joblib
│   │   # Fitted feature preprocessing pipeline
│   │
│   └── metrics.json
│       # Test metrics and selected threshold
│
├── app.py
│   # Streamlit analytics and prediction dashboard
│
├── train_model.py
│   # Data preprocessing, training, evaluation,
│   # threshold selection, and artifact generation
│
├── requirements.txt
│   # Python dependencies
│
├── .gitignore
│   # Git exclusion rules
│
└── README.md
    # Project documentation
```

The three files inside `models/` are generated when the training script runs successfully.

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

Install the following:

- Python 3.11 recommended.
- Git.
- A code editor such as Visual Studio Code.

### Step 1: Clone the Repository

```bash
git clone https://github.com/yashchaudhari372-lab/Customer_Churn_Analysis_using_ANN.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Customer_Churn_Analysis_using_ANN
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Generate the Model Artifacts

```bash
python train_model.py
```

This command trains the ANN and generates the saved model, preprocessing pipeline, and evaluation metrics.

### Step 7: Launch the Application

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to access the application.

---

## 🏋️ Model Training

The model can be retrained using the included `train_model.py` script.

### Training Process

1. Load the customer dataset.
2. Select the model input features and target variable.
3. Create stratified training, validation, and test datasets.
4. Fit the preprocessing pipeline on the training dataset.
5. Transform the validation and test datasets.
6. Construct the ANN architecture.
7. Train the model using Adam and binary cross-entropy.
8. Apply early stopping based on validation loss.
9. Select the classification threshold using validation F1-score.
10. Evaluate the final predictions on the test dataset.
11. Save the model, preprocessor, and evaluation metrics.

### Generated Artifacts

| File | Purpose |
|---|---|
| `models/churn_ann.keras` | Stores the trained neural network |
| `models/preprocessor.joblib` | Stores the fitted feature preprocessing pipeline |
| `models/metrics.json` | Stores test evaluation metrics and the selected classification threshold |

To retrain the model, run:

```bash
python train_model.py
```

After training completes, restart or refresh the Streamlit application to use the updated artifacts.

---

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:** [Customer Churn Intelligence Dashboard](https://customerchurnanalysisusingann1-aqtdqfhsuti2lcp2mgfgxp.streamlit.app/)

### Deployment Workflow

1. Push the project source code to GitHub.
2. Ensure the required dependencies are listed in `requirements.txt`.
3. Generate the model artifacts using the training script.
4. Ensure the deployment repository contains the required dataset and model artifacts.
5. Open Streamlit Community Cloud.
6. Connect your GitHub account.
7. Select the project repository and branch.
8. Set `app.py` as the application entry point.
9. Configure a compatible Python version.
10. Deploy the application.

### Important Deployment Requirements

The application expects the following files to exist:

```text
data/Churn_Modelling.csv
models/churn_ann.keras
models/preprocessor.joblib
models/metrics.json
```

If the model artifacts are missing, run the training script before launching the application.

Ensure that the Python version and installed TensorFlow dependencies are compatible with the deployment environment.

---

## 💼 Business Applications

Customer churn prediction can support several business activities.

### 1. Customer Retention

Identify customers with higher predicted churn risk so that retention teams can prioritize follow-up activities.

### 2. Customer Segmentation

Analyze churn patterns across customer demographics, geography, account balances, and banking product usage.

### 3. Targeted Marketing

Use churn predictions as one input when designing customer engagement and retention campaigns.

### 4. Banking Strategy

Explore the characteristics associated with churn and use the findings to guide further business analysis.

### 5. Data-Driven Decision-Making

Combine predictive probabilities with customer analytics to support more informed retention decisions.

**Important:** A predicted churn risk does not establish why a customer leaves. Business decisions should consider additional customer information, model limitations, and appropriate human review.

---

## ⚠️ Limitations

- The model is trained on the available historical dataset and may not generalize to every bank or customer population.
- The quality of predictions depends on the dataset, feature quality, and model training process.
- Customer behavior can change over time, potentially reducing predictive performance.
- Accuracy alone cannot fully describe model performance, particularly when churn classes are imbalanced.
- A high predicted churn probability does not guarantee that a customer will leave.
- Model predictions identify statistical patterns, not causal explanations.
- The application is an educational and analytical project and should not be treated as a fully validated production banking risk system.

---

## 🔮 Future Improvements

Potential enhancements include:

- Compare ANN performance with Logistic Regression, Random Forest, and Gradient Boosting models.
- Add confusion matrix and ROC curve visualizations.
- Improve model interpretability using SHAP or other explainability methods.
- Introduce additional customer behavior features when reliable data becomes available.
- Add automated model monitoring and retraining.
- Evaluate probability calibration.
- Analyze model performance across different customer groups.
- Improve the dashboard with downloadable reports.
- Add customer retention recommendations based on carefully validated business rules.
- Establish ongoing evaluation using more recent customer data.

---

## 📚 Learning Outcomes

This project demonstrates practical experience with:

- Python-based data analysis.
- Exploratory data analysis and interactive visualization.
- Categorical encoding and numerical feature scaling.
- Artificial Neural Network development using TensorFlow/Keras.
- Binary classification.
- Stratified train-validation-test splitting.
- Early stopping and model optimization.
- Decision threshold tuning.
- Classification metrics and model evaluation.
- Saving and loading trained machine learning artifacts.
- Building interactive applications with Streamlit.
- Deploying a machine learning application through a web platform.

---

## 👨‍💻 Author

**Yash Chaudhari**

Explore the source code and project implementation:

- **GitHub Repository:** [Customer_Churn_Analysis_using_ANN](https://github.com/yashchaudhari372-lab/Customer_Churn_Analysis_using_ANN)
- **Live Project:** [Customer Churn Intelligence Dashboard](https://customerchurnanalysisusingann1-aqtdqfhsuti2lcp2mgfgxp.streamlit.app/)

---

## ⭐ Support

If you find this project useful for learning about deep learning, customer analytics, or churn prediction, consider giving the repository a star on GitHub.

Thank you for visiting this project!
