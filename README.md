# Credit Card Fraud Detection: Model & Sampling Comparison

This project is a **Master's degree assignment** focused on evaluating various machine learning strategies to handle highly imbalanced data. 

### Project Overview
The core objective is to compare different methodologies for achieving optimal fraud detection results, specifically focusing on:
* **Model Comparison:** Evaluating Logistic Regression, Random Forest, XGBoost, and Neural Networks.
* **Imbalance Handling:** Comparative analysis of **Random Under-Sampling (RUS)** vs. **SMOTE (Synthetic Minority Over-sampling Technique)**.
* **Anomaly Detection:** Integrating an **Isolation Forest** as a feature engineering step.

### Dataset
The analysis is based on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013.

### Interactive Notebook
Click the badge below to open the project directly in Google Colab and run the experiments:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wisuy/Credit-Card-Fraud/blob/main/Credit_Card_Fraud.ipynb)

### Setup & Reproducibility
To replicate the results of this study, ensure your environment meets the following specifications:

* **Python Version:** `3.12.12`
* **Dependencies:** All necessary libraries (Scikit-Learn, XGBoost, Imbalanced-Learn, etc.) are managed via the `requirements.txt` file.

To install the environment locally:

```bash
# Clone the repository
git clone https://github.com/Wisuy/Credit-Card-Fraud.git
cd Credit-Card-Fraud

# Install required packages
pip install -r requirements.txt
```

---

### Technical Notes & Corrections
* **Focal Loss (Implementation vs. Documentation):** Although Focal Loss is discussed in the project presentation and documentation as the best strategy, it was excluded from the final code implementation. The computational overhead (nearly doubling training time) did not justify the marginal gains in AUPRC compared to the current methods.
* **Ablation Study (Isolation Forest):** Due to a procedural oversight, the final comparison lacks a "baseline-only" test (data without Isolation Forest features). However, preliminary iterations suggested that the impact of the IF-generated features on the final performance was minimal.
* **Temporal Features:** The `Time` column from the original dataset was excluded from the feature set. Initial exploratory data analysis indicated that its inclusion did not significantly improve the models' ability to distinguish between fraud and legitimate transactions.
