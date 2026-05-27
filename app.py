import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Financial Health Predictor",
    layout="centered"
)

st.title("Employee Financial Health Predictor")

st.write(
    "Predict employee financial wellness using salary, savings, expenses, and EMI burden."
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("Employee_data.csv")

    # Remove unnecessary columns
    remove_cols = [
        'EmployeeID',
        'EmployeeName',
        'Name'
    ]

    existing = [c for c in remove_cols if c in df.columns]

    df.drop(columns=existing, inplace=True)

    # ---------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------

    df['Total_Expenditure'] = (
        df['Rent_Expense']
        + df['Grocery_Expense']
        + df['EMI_or_Loan_Payment']
        + df['Entertainment_Expense']
        + df['Other_Expenses']
    )

    df['Profit'] = (
        df['Net_Salary']
        - df['Total_Expenditure']
    )

    df['SavingsRatio'] = (
        df['Savings_Amount']
        / df['Monthly_Salary']
    )

    df['ExpenseRatio'] = (
        df['Total_Expenditure']
        / df['Monthly_Salary']
    )

    df['EMIBurden'] = (
        df['EMI_or_Loan_Payment']
        / df['Monthly_Salary']
    )

    # ---------------------------------------------------
    # TARGET VARIABLE
    # ---------------------------------------------------

    df['FinancialHealthScore'] = (
        (df['Profit'] * 0.4)
        + (df['Savings_Amount'] * 0.3)
        - (df['EMI_or_Loan_Payment'] * 0.15)
        - (df['Total_Expenditure'] * 0.15)
    )

    threshold = df['FinancialHealthScore'].median()

    df['FinancialHealth'] = (
        df['FinancialHealthScore'] > threshold
    ).astype(int)

    return df


# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

@st.cache_resource
def train_model(df):

    important_features = [
        'Monthly_Salary',
        'Savings_Amount',
        'Total_Expenditure',
        'EMI_or_Loan_Payment',
        'Profit',
        'SavingsRatio',
        'ExpenseRatio',
        'EMIBurden'
    ]

    X = df[important_features]

    y = df['FinancialHealth']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy


# ---------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------

df = load_data()

model, scaler, accuracy = train_model(df)

# ---------------------------------------------------
# MODEL ACCURACY
# ---------------------------------------------------

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.header("Enter Employee Financial Details")

monthly_salary = st.number_input(
    "Monthly Salary",
    min_value=10000,
    value=70000
)

savings = st.number_input(
    "Savings Amount",
    min_value=0,
    value=10000
)

rent_expense = st.number_input(
    "Rent Expense",
    min_value=0,
    value=12000
)

grocery_expense = st.number_input(
    "Grocery Expense",
    min_value=0,
    value=5000
)

emi = st.number_input(
    "EMI or Loan Payment",
    min_value=0,
    value=4000
)

other_expenses = st.number_input(
    "Other Expenses",
    min_value=0,
    value=3000
)

# ---------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------

total_expenditure = (
    rent_expense
    + grocery_expense
    + emi
    + other_expenses
)

profit = (
    monthly_salary
    - total_expenditure
)

savings_ratio = (
    savings / monthly_salary
)

expense_ratio = (
    total_expenditure / monthly_salary
)

emi_burden = (
    emi / monthly_salary
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict Financial Health"):

    input_data = pd.DataFrame([{
        'Monthly_Salary': monthly_salary,
        'Savings_Amount': savings,
        'Total_Expenditure': total_expenditure,
        'EMI_or_Loan_Payment': emi,
        'Profit': profit,
        'SavingsRatio': savings_ratio,
        'ExpenseRatio': expense_ratio,
        'EMIBurden': emi_burden
    }])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "Employee is Financially Healthy"
        )

    else:

        st.error(
            "Employee is Financially Unhealthy"
        )

    st.write(
        f"Confidence Score: {probability:.2f}"
    )

    # ---------------------------------------------------
    # FINANCIAL SUMMARY
    # ---------------------------------------------------

    st.subheader("Financial Summary")

    st.write(f"Total Expenditure: {total_expenditure}")

    st.write(f"Profit: {profit}")

    st.write(f"Savings Ratio: {savings_ratio:.2f}")

    st.write(f"Expense Ratio: {expense_ratio:.2f}")

    st.write(f"EMI Burden: {emi_burden:.2f}")