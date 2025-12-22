<div align="center">

# ✈️ Predictive Maintenance with Explainable AI (XAI)
### Remaining Useful Life (RUL) Prediction on NASA CMAPSS Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff0055?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Week_1_Complete-success?style=for-the-badge)

<p align="center">
  <strong>A data engineering and machine learning pipeline to predict jet engine failures <br>
  and explain <i>why</i> they occur using SHAP values.</strong>
</p>

</div>

---

## 📖 Project Overview

This repository contains an end-to-end pipeline for **Predictive Maintenance**. Using the **NASA CMAPSS Jet Engine dataset**, the project aims to predict whether an engine will fail within a specific time window (24 cycles) and uses Explainable AI (XAI) to interpret the root causes of failure predictions.

Current Status: **Week 1 (Data Engineering & Baseline Modeling)**.

## ⚙️ Key Features

* **🔄 Multi-Stream Ingestion**: Automates loading of multiple CMAPSS data files (`FD001` - `FD004`).
* **🧹 Smart Data Cleaning**:
    * Identifies and drops sensors with zero variance (noise reduction).
    * Handles missing data via forward-filling.
* **⏱️ Temporal Feature Engineering**:
    * **Lag Features**: Captures trends using 1-step and 2-step lags.
    * **Rolling Statistics**: Calculates Rolling Mean and Std Dev (Window=5).
* **🎯 Target Generation**:
    * Computes **RUL** (Remaining Useful Life).
    * Generates binary failure labels (`failure_24h`) for classification.
* **🧠 Baseline Model**: Logistic Regression with time-aware splitting.

---

## 📂 Directory Structure

Ensure your project folder is organized as follows:

```text
├── CMaps/                     # 📥 Input Data (Download from NASA)
│   ├── train_FD001.txt
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   └── train_FD004.txt
├── output/                    # 📤 Generated Artifacts
│   ├── correlation_matrix.png
│   ├── shap_summary_plot.png
│   └── week1_feature_engineered_dataset.csv
├── Predictive Maintenance with Explainable AI.py   # 📜 Main Script
├── requirements.txt           # 📦 Dependencies
└── README.md                  # 📄 This file
