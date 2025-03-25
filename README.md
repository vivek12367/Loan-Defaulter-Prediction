## 📂 Loan Default Prediction

### 💡 Overview
This project focuses on building a machine learning model to predict the likelihood of a customer defaulting on a loan, using historical data of repeated loans from current clients. It demonstrates end-to-end development from data exploration to model evaluation, with a focus on handling class imbalance and providing explainability for business use.

---

### 🚀 Goals
- Predict the probability of loan default (`bad_flag`) using available features.
- Compare multiple models and select the best one.
- Handle class imbalance using **SMOTE**.
- Explain the model’s behavior using **SHAP**.
- Provide business-friendly insights and visualizations.

---

### 🧰 Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP (explainability)
- Matplotlib, Seaborn (visualization)

---

### 📊 Exploratory Data Analysis (EDA)
- Investigated feature distributions and correlations.
- Identified key features: `close_loans_cnt`, `score_1`, `past_billings_cnt`, etc.
- Visualized default rates by gender, region, credit score, and more.

---

### ⚙️ Modeling Approach
- Split the data into training and test sets using stratified sampling.
- Applied **SMOTE** to oversample the minority class (defaults).
- Trained and compared multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - **XGBoost** (Best performer)

---

### 🏆 Best Model – XGBoost (with SMOTE)
| Metric                | Value |
|-----------------------|-------|
| Accuracy              | 85%   |
| Recall (Default Class)| 25%   |
| Precision (Default)   | 24%   |
| ROC AUC               | ~0.70 |

- Balanced performance between risk detection and approval volume.
- Interpretable using **SHAP** feature importance.

---

### 📈 Visualizations Included
- Correlation heatmap
- Default rate by categorical features
- Score distribution by class
- Loan relationship vs default
- ROC and Precision-Recall curves
- SHAP summary plot

---

### 🔍 Key Insights
- Closed loan history and score values are strong indicators of default.
- Imbalanced data requires thoughtful modeling (SMOTE + metric choice).
- Business can tune thresholds to control approval risk tradeoffs.

---

### 📌 How to Run
1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook: `Loan_Default_Prediction.ipynb`

---

### ✅ Future Work
- Improve feature engineering with repayment patterns
- Deploy model via Flask or Streamlit
- Add dashboard visualizations
- Include cost-sensitive evaluation (profit/loss per prediction)

---
