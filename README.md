# 🎯 HR Employee Attrition & Retention Dashboard

Welcome to the **HR Employee Attrition Dashboard**! This interactive web application leverages a full-stack Machine Learning pipeline to predict and deeply understand employee turnover, allowing HR professionals to proactively manage organizational retention.

![App Dashboard Screenshot](https://img.shields.io/badge/UI-Glassmorphism%20Dark%20Mode-8b5cf6?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Python-Streamlit-3b82f6?style=for-the-badge&logo=python)

## 📌 Project Overview
In today's competitive landscape, employee retention is critical. Replacing an employee often costs anywhere from 50% to 200% of their annual salary. This dashboard is built on top of the renowned **IBM HR Analytics Employee Attrition** dataset to visually expose trends and forecast individual "Flight Risks" using a tailored Random Forest algorithm.

### Core Features
- **🔮 Real-Time Attrition Prediction:** Instantly flag high-performing individuals who might be considering leaving using a custom threshold tailored for severe class imbalances.
- **📊 Exploratory Data Analysis (EDA):** Interactive, reactive visualizations (Donuts, Boxplots, Histograms) depicting age distributions, income curves, and attrition per job role.
- **⚙️ Model Transparency:** Real-time confusion matrix, accuracy, precision/recall metrics, and 2D Principal Component Analysis (PCA) projection of the active predictive algorithm.
- **🌌 Premium UI/UX:** A stunning Glassmorphism dark mode utilizing deep neon accents and floating metric cards.

---

## 🛠️ Technology Stack
This project runs entirely on Python's robust data science stack:
- **Frontend & App Framework:** `Streamlit`
- **Data Manipulation:** `Pandas`, `NumPy`
- **Machine Learning Engine:** `Scikit-learn`, `Joblib`
- **Data Visualization:** `Plotly Express`, `Plotly Figure Factory`

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### 1. Prerequisites
Ensure you have Python 3.9+ installed and clone or download this repository.

### 2. Setting up the Virtual Environment
All dependencies for this project are listed in the `requirements.txt` file. We strongly recommend using the created `.venv` virtual environment to prevent global package conflicts.

Activate the virtual environment:
**Windows (PowerShell or Command Prompt):**
```powershell
.venv\Scripts\activate
```

*(If setting up from scratch, you can install the dependencies via `pip install -r requirements.txt`)*

### 3. Launching the App
Run the following command to boot up the dashboard locally:
```powershell
streamlit run app.py
```
This will automatically compile the CSS engine, deserialize the ML models, and open the interface at `http://localhost:8501`.

---

## 🧠 Under the Hood (Machine Learning)
1. **Data Preprocessing:** Categorical features undergo One-Hot Encoding via Dummy Variable Trap prevention (`drop_first=True`). Unnecessary noise variables (Zero-variance columns) are surgically dropped.
2. **Model Formulation:** Utilizing `RandomForestClassifier(class_weight='balanced', n_estimators=100)` to combat the severe 84:16 class imbalance present in HR populations.
3. **Optimized Thresholding:** Implemented a granular `0.30` classification boundary directly within `app.py` allowing HR to detect emerging attrition risks significantly earlier than basic defaults.

---
*Developed by Priyanshu Yadav | IBM HR Attrition Capstone Project*
