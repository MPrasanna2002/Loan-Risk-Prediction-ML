# Loan-Risk-Prediction-ML
Machine Learning model to predict loan risk using classification algorithms
# 🏦 Loan Risk Prediction - Machine Learning Project

A end-to-end Machine Learning project that predicts whether a loan application should be **✅ Approved** or **❌ Rejected** based on applicant's financial and personal details.

---

## 📖 Project Description

Financial institutions face the critical challenge of identifying high-risk loan applicants to minimize defaults and financial losses. This project addresses that challenge by building a robust binary classification model using real-world credit risk data.

The project covers the **complete ML pipeline** — from raw data cleaning to model deployment-ready prediction function — making it a production-ready solution for automated loan approval decisions.

### 🎯 Problem Statement
> Given an applicant's personal and financial details, predict whether they are a **loan default risk (Rejected)** or a **safe borrower (Approved)**.

---

## 🔄 Project Workflow

### 1️⃣ Data Collection & Loading
- Loaded `cr_loan2.csv` containing applicant financial records
- Explored dataset structure using `head()` and `isnull().sum()`

### 2️⃣ Data Cleaning & Preprocessing
- Removed outliers: applicants with age > 70 and employment length > 45 years
- Handled missing values in `loan_int_rate` using **median imputation**
- Dropped irrelevant column `loan_grade`

### 3️⃣ Feature Engineering
- Applied **One-Hot Encoding** on:
  - `person_home_ownership` (OTHER, OWN, RENT)
  - `loan_intent` (EDUCATION, HOMEIMPROVEMENT, MEDICAL, PERSONAL, VENTURE)
- Converted `cb_person_default_on_file` to binary (Y → 1, N → 0)

### 4️⃣ Feature Scaling
- Applied **StandardScaler** on numerical features
- Combined scaled numerical + encoded categorical features into final dataset

### 5️⃣ Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset
- Ensured model doesn't bias toward the majority class

### 6️⃣ Model Training & Comparison
- Split data: **80% Training / 20% Testing**
- Trained and compared **5 Machine Learning models**:

| Model | Type |
|-------|------|
| Logistic Regression | Linear Classifier |
| Random Forest | Ensemble (Bagging) |
| XGBoost | Ensemble (Boosting) |
| Gradient Boosting | Ensemble (Boosting) |
| Neural Network (MLP) | Deep Learning |

- Visualized model accuracy scores using a **bar chart**

### 7️⃣ Best Model Selection & Saving
- ✅ **Random Forest Classifier** selected as the best model
- Saved using `joblib`:
  - `rf_model.pkl` → Trained model
  - `scaler.pkl` → StandardScaler for preprocessing

### 8️⃣ Prediction Function
- Built a `simple_predict()` function that:
  - Accepts 16 input features
  - Scales numerical features using saved scaler
  - Returns **"Loan Approved"** or **"Loan Rejected"**

---

## 📁 Files in this Repository

| File | Description |
|------|-------------|
| `cr_loan2.csv` | Raw dataset used for training and testing |
| `credit_risk_modeling_Project.ipynb` | Complete ML pipeline notebook |
| `scaler.pkl` | Saved StandardScaler for preprocessing |
| `rf_model.pkl` | Trained Random Forest Model (available on request) |

---

## 🔍 Input Features

| Feature | Description |
|---------|-------------|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_emp_length` | Employment length (years) |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan as % of income |
| `cb_person_cred_hist_length` | Credit history length |
| `person_home_ownership` | OWN / RENT / OTHER |
| `loan_intent` | Purpose of loan |
| `cb_person_default_on_file` | Previous default history |

---

## 🛠 Tools & Technologies

- **Language:** Python
- **Platform:** Google Colab
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Joblib

---

## ▶️ How to Run

1. Clone this repository
2. Open `credit_risk_modeling_Project.ipynb` in Google Colab or Jupyter
3. Upload `cr_loan2.csv` to your Colab environment
4. Run all cells to train the model and generate predictions

---

## 👤 Author

**Prasanna M** — GitHub Profile - (https://github.com/MPrasanna2002)
