
---

# 🔥 **PROJECT — CREDIT RISK MODEL**

```markdown
# 📊 Credit Risk Prediction – End-to-End ML Application

# 🔥 PROJECT — CREDIT RISK MODEL

## 📊 Credit Risk Prediction – End-to-End ML Application


---

## 🚀 Overview

This project presents an **end-to-end machine learning pipeline** for predicting **customer default risk** using socioeconomic data.

It combines **automated model selection (PyCaret)** with an **interactive web application**, demonstrating how ML models can support real-world financial decision-making.

---

## 🎯 Business Problem

Financial institutions need to answer:

👉 **“Should we approve this customer’s credit?”**

This project builds a system to:
- Predict default probability  
- Support risk-based decision-making  
- Improve credit approval strategies  

---

## 💡 Solution

- Automated model selection and optimization using PyCaret  
- Built a **deployable Streamlit application**  
- Delivered interpretable outputs for decision support  

---

## 🧠 Machine Learning Pipeline

### Data
- ~50,000 customer records  
- Socioeconomic and behavioral variables  

### Process
- Data preprocessing (PyCaret `setup()`)
- Feature engineering  
- Model comparison (`compare_models`)
- Hyperparameter tuning (`tune_model`)  

### Model
- LightGBM (best-performing model)  
- Evaluated using AUC, confusion matrix, and feature importance  

---

## 📊 Features

- 📈 Exploratory Data Analysis (EDA)
- 🔍 Feature correlation analysis  
- 🤖 Automated model selection  
- ⚙️ Hyperparameter tuning  
- 📊 Model evaluation (AUC, confusion matrix)
- 📉 Feature importance visualization  
- 💾 Model saving/loading  
- 🖥️ Interactive UI with Streamlit  

---

## 🏗️ Architecture

- Data → Preprocessing → Model training → Evaluation → Deployment (Streamlit)

---

## 🚀 Run Locally

```bash
git clone https://github.com/Samuel-Oliveira-saturno/Cred_Score_mod38.git
cd Cred_Score_mod38
pip install -r requirements.txt
streamlit run credit_score.py
