# Breast Cancer Classification Using Support Vector Machine (SVM)

## 📌 Project Overview

This project explores whether breast tumors can be classified as **Benign (non-cancerous) or Malignant (cancerous)** using diagnostic measurements and a Machine Learning algorithm called Support Vector Machine (SVM).

The project was completed as part of a college Machine Learning mini-project.

## 🎯 Problem Statement

**Can breast tumors be classified as benign or malignant using diagnostic measurements?**

The goal is to train and evaluate an SVM classification model using the Breast Cancer Wisconsin (Diagnostic) dataset.

## 📊 Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Total records:** 569
- **Diagnostic features:** 30 numerical features
- **Target variable:** `diagnosis`
- **Classes:**
  - `B` — Benign
  - `M` — Malignant

The dataset contains diagnostic measurements computed from digitized images of breast mass samples.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## ⚙️ Project Workflow

1. **Data Loading and Inspection** — examined the dataset shape, column names, data types, missing values, and duplicate records.
2. **Data Preprocessing** — removed irrelevant identifier and empty columns, separated the target variable, and prepared the features.
3. **Exploratory Data Analysis (EDA)** — explored class distribution, feature distributions, outliers, and relationships between variables.
4. **Statistical Analysis** — calculated descriptive statistics, variability, correlations, and potential outliers.
5. **Feature Scaling** — used `StandardScaler` to standardize numerical features.
6. **Model Training** — trained a Support Vector Machine classifier.
7. **Hyperparameter Tuning** — used cross-validation and grid search to select model parameters.
8. **Model Evaluation** — evaluated predictions using classification metrics and visualizations.
9. **Interpretation** — discussed the model's performance, limitations, and possible improvements.

## 🤖 Algorithm: Support Vector Machine

Support Vector Machine is a supervised Machine Learning algorithm used for classification. It attempts to find a decision boundary that separates classes while maximizing the margin between them.

This project evaluates SVM configurations and uses feature scaling because diagnostic measurements can have different numerical ranges.

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Visualizations include class distribution, feature analysis, a correlation heatmap, a confusion matrix, an ROC curve, and a learning curve.

## 🧰 How to Run the Project

### Option 1: Google Colab

1. Open your notebook in Google Colab.
2. Upload the dataset file if it is not already accessible.
3. Run the notebook cells in order.
4. Review the EDA plots, model predictions, and evaluation results.

### Option 2: Run Locally

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Open the project folder:

   ```bash
   cd YOUR_PROJECT_FOLDER
   ```

3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Place the dataset where the code expects it, or update the dataset path in the code.
5. Run the Python script or open the notebook.

## 🔍 Key Learning Outcomes

- Understanding the end-to-end Machine Learning workflow.
- Performing data cleaning and exploratory data analysis.
- Applying feature scaling before training an SVM model.
- Using cross-validation and hyperparameter tuning.
- Evaluating classification models with multiple metrics.
- Interpreting false positives, false negatives, and model limitations.

## ⚠️ Disclaimer

This project is for educational purposes only. It is not a clinically validated medical diagnostic system and must not be used to make medical decisions.

## 👨‍💻 Project Information

- **Group:** 9
- **Project:** Breast Cancer Classification
- **Algorithm:** Support Vector Machine (SVM)

## 🤝 Feedback

Suggestions and constructive feedback are welcome!
