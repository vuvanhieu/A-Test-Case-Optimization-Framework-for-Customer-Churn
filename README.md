# 📊 Customer Churn Prediction with Test Case Optimization and Explainable AI

This project implements a complete pipeline for **multi-task customer churn prediction** using deep learning and explainable AI (XAI). It includes automated test case configuration, model training, performance evaluation, rule-based explanation, and a churn risk alert system.

---

## 📦 Key Features

- 🔄 **Multi-task learning**: Jointly predicts churn, credit score class, and high-balance risk.
- 🧪 **Test case optimization**: Systematic evaluation of different batch sizes, epochs, loss weights, and focal loss combinations.
- 📈 **Performance visualization**: Generates confusion matrices, ROC, PR curves, and AUC scores.
- ⚖️ **Label balancing**: Uses SMOTE for primary task and nearest-neighbor matching for auxiliary tasks.
- 🧠 **Explainability**: Rule-based churn reason extraction at both global and individual levels.
- 🚨 **Churn risk alert system**: Assigns risk levels and explanations to high-risk customers.

---

## 🧰 Requirements

- Python >= 3.7
- TensorFlow >= 2.4
- scikit-learn
- imbalanced-learn
- pandas, numpy, matplotlib, seaborn

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Core Functions

### 🔹 `train_multitask_model()`
Trains the main deep learning model with options for focal loss and class weights.

### 🔹 `build_multitask_dnn(input_dim)`
Builds the shared DNN architecture with 3 output heads.

### 🔹 `generate_test_cases()`
Creates a grid of test configurations (`TC_000`, `TC_001`, ...).

### 🔹 `apply_balancing_all_tasks()`
Balances churn using SMOTE and aligns auxiliary labels.

### 🔹 `plot_all()`
Visualizes confusion matrix, ROC, PR curves, and AUC.

### 🔹 `generate_churn_reason_statistics()`
Extracts and visualizes the most common churn reasons.

### 🔹 `generate_focal_comparison_report()` / `generate_focal_comparison_allTask_report()`
Summarizes performance comparison between focal and non-focal settings.

### 🔹 `run_baseline_models()`
Trains baseline models: Logistic Regression, SVM, Naive Bayes, MLP.

### 🔹 `run_churn_risk_alert_pipeline()`
Predicts churn risk scores and levels for each customer.

### 🔹 `save_predictions_with_explanations()`
Outputs CSV with predicted labels and rule-based explanations.

---

## 📁 Output Directory Structure

```
project_root/
├── Multitask DNN/
│   ├── TC_000_ep150_bs32/
│   ├── test_case_list.csv
│   └── gridsearch_multitask_results.csv
├── Baselines/
│   └── baseline_comparison.csv
├── churn_risk_alert/
│   ├── churn_risk_alert_log.csv
│   └── high_risk_customers_with_reasons.csv
├── reports/
│   ├── explanation_results.csv
│   ├── focal_vs_nonfocal_comparison_report.txt
│   └── best_model_summary.txt
```

---

## 🚀 Running the Pipeline

Ensure your dataset is in the same folder as the script (e.g., `Bank_Customer_Churn.csv`). Then run:

```bash
python your_script.py
```

All final reports and plots will be saved in the `reports/` directory.

---

## 📬 Contact

For questions or collaboration: **Van Hieu Vu** (vvhieu@ioit.ac.vn)

---