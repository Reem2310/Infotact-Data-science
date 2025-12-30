<div align="center">

# ✈️ Predictive Maintenance with Explainable AI (XAI)
### Remaining Useful Life (RUL) Prediction on NASA CMAPSS Data

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff0055?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Project_Complete-success?style=for-the-badge)

<p align="center">
  <strong>An end-to-end industrial AI pipeline to predict jet engine failures <br>
  and provide transparent, sensor-level explanations using SHAP values.</strong>
</p>

</div>

---

## 📖 Project Overview

This repository implements a three-stage pipeline for **Predictive Maintenance**. By analyzing the **NASA CMAPSS Jet Engine dataset**, the system predicts imminent failures and utilizes **Explainable AI (XAI)** to solve the "black box" problem, allowing maintenance engineers to understand which sensor patterns lead to a high-risk alert.

## 🛠️ Multi-Week Pipeline

### **Week 1: Data Engineering & Baselines**
* **Ingestion**: Automated loading of `FD001` through `FD004` datasets.
* **Cleaning**: Drops non-informative constant sensors and handles missing values via forward-fill.
* **Feature Engineering**: Creates temporal context using **Lag Features** (1-2 cycles) and **Rolling Statistics** (Mean/Std Dev) to capture engine degradation over time.

### **Week 2: High-Performance Modeling**
* **Advanced Architecture**: Transitioned from Logistic Regression to a tuned **XGBoost Classifier**.
* **Class Imbalance**: Utilized `scale_pos_weight` to handle the rare nature of failure events in industrial data.
* **Optimization**: Implemented `RandomizedSearchCV` focusing on **F1-Score** and **Recall** to minimize dangerous false negatives.

### **Week 3: Explainable AI (SHAP)**
* **Global Interpretability**: Summary plots identify the most critical sensors across the entire fleet.
* **Local Interpretability**: Force plots explain *specific* individual predictions, showing which sensor values increased or decreased the risk of failure.
* **Logic Validation**: Cross-references top SHAP features with Pearson correlation to ensure model reliability.



---

## 📂 Directory Structure

```text
├── CMaps/                     # 📥 Input: Raw NASA .txt files
├── output/                    # 📤 Output: Models, CSVs, and Visualizations
│   ├── correlation_matrix.png
│   ├── confusion_matrix_week2.png
│   ├── shap_summary_plot.png
│   ├── shap_force_plot_sample.png
│   ├── xgboost_week2_final.joblib    # 🧠 Saved High-Performance Model
│   └── week1_feature_engineered_dataset.csv
├── Predictive Maintenance...py# 📜 Week 1: Data & Features
├── week2_modeling.py          # 📜 Week 2: XGBoost & Tuning
├── week3_xai.py               # 📜 Week 3: SHAP Explanations
├── requirements.txt           # 📦 Project Dependencies
└── README.md                  # 📄 You are here
