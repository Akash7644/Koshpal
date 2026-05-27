# Employee Financial Health Predictor

A Machine Learning web application that predicts whether an employee is financially healthy based on salary, savings, expenditures, EMI burden, and other financial indicators.

## 🚀 Live Demo

🔗 Streamlit App:  
https://koshpal-njnr6zrwlucf6svpsucakw.streamlit.app/

---

# 📌 Built Using

- Python
- Logistic Regression
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Seaborn
- Matplotlib

---

# 🎯 Project Objective

The objective of this project is to analyze employee financial behavior and predict financial wellness using machine learning techniques.

The system evaluates:
- salary strength
- savings behavior
- expenditure patterns
- EMI burden
- investment patterns

to classify employees as:

✅ Financially Healthy  
❌ Financially Unhealthy

---

# 📊 Dataset Information

The dataset contains employee financial information such as:

- Monthly Salary
- Net Salary
- Income Tax
- PF Contribution
- Savings Amount
- Investments
- EMI or Loan Payments
- Rent Expense
- Grocery Expense
- Total Expenditure
- Profit

The dataset was enhanced with:
- synthetic employee records
- employee names and IDs
- engineered financial indicators

### Final Dataset Size
- 10,000 employee records

---

# 📈 Exploratory Data Analysis (EDA)

Several visualizations were performed to understand financial relationships and employee behavior.

---

## 🔥 Correlation Heatmap

Correlation heatmaps revealed strong relationships between:

- Monthly Salary and Profit
- Monthly Salary and Net Salary
- EMI Burden and Financial Health
- Expense Ratio and Financial Health

### Key Insight

Higher expenditures and EMI burdens negatively affect employee financial wellness.

---

## 📦 Boxplots

Boxplots were used to detect outliers in:

- Monthly Salary
- Profit
- Savings Amount
- Total Expenditure

### Observation

The dataset showed balanced financial distributions with minimal extreme outliers.

---

## 📉 Scatterplots

Scatterplots helped analyze relationships between financial variables.

### Monthly Salary vs Profit

A strong positive correlation was observed:
- higher salary generally leads to higher profit.

### Monthly Salary vs Savings Amount

Employees with higher salaries tend to maintain higher savings.

### Total Expenditure vs Profit

A negative relationship was observed:
- higher expenditures reduce employee profit and financial stability.

---

# ⚙️ Feature Engineering

Several meaningful financial indicators were engineered from raw financial data.

---

## Savings Ratio

```text
Savings Ratio = Savings Amount / Monthly Salary
```

---

## Expense Ratio

```text
Expense Ratio = Total Expenditure / Monthly Salary
```

---

## EMI Burden

```text
EMI Burden = EMI or Loan Payment / Monthly Salary
```

---

## Financial Health Score

A custom financial health score was created using:

- profit
- savings
- investments
- EMI burden
- expenditure

This score was later converted into a binary classification target variable:

- 1 → Financially Healthy
- 0 → Financially Unhealthy

---

# 🤖 Machine Learning Model

## Model Used

- Logistic Regression

---

## Preprocessing Steps

- Feature Engineering
- Feature Selection
- Train-Test Split
- Feature Scaling using StandardScaler

---

# 📊 Model Performance

## ✅ Accuracy

```text
91.65%
```

---

## 📋 Classification Report

| Class | Precision | Recall | F1-Score |
|------|------------|---------|----------|
| 0 | 0.92 | 0.92 | 0.92 |
| 1 | 0.91 | 0.92 | 0.91 |

---

## 🔲 Confusion Matrix

```text
[[939  86]
 [ 81 894]]
```

---

## 📌 Interpretation

- The model demonstrates strong predictive performance.
- Both financially healthy and unhealthy classes are predicted with balanced precision and recall.
- The model generalizes well on unseen employee financial data.

---

# 🌐 Streamlit Web Application

A Streamlit interface was developed to:

- collect employee financial inputs
- calculate financial indicators automatically
- predict employee financial health
- display confidence score and financial summary

---

## 🧾 User Inputs

- Monthly Salary
- Savings
- Rent Expense
- Grocery Expense
- EMI Burden
- Other Expenses

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# ▶️ How to Run the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```text
employee-financial-health-predictor/
│
├── app.py
├── Employee_data.csv
├── EmployeeFinancialHealthAnalysis.ipynb
├── requirements.txt
└── README.md
```

---

# 🚀 Future Improvements

Possible future enhancements:

- XGBoost / Random Forest models
- SHAP explainability
- Interactive dashboards
- Cloud database integration
- PDF financial reports
- User authentication
- Deployment using Docker and AWS

---

# 🏁 Conclusion

This project successfully demonstrates how machine learning can be applied to analyze employee financial wellness using salary, expenditure, savings, and EMI-related indicators.

The Logistic Regression model achieved strong predictive performance with approximately **92% accuracy** and provides a practical framework for employee financial health assessment.
