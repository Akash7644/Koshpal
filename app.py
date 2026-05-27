import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Financial Health Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #4CAF50;
}

.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("💰 Employee Financial Health Predictor")

st.markdown("""
Analyze employee financial wellness using Machine Learning.

The system evaluates:
- Salary
- Savings
- Expenditures
- EMI Burden
- Profitability
""")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("Employee_data.csv")

    remove_cols = [
        'EmployeeID',
        'EmployeeName',
        'Name'
    ]

    existing = [c for c in remove_cols if c in df.columns]

    df.drop(columns=existing, inplace=True)

    # Feature Engineering

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

    # Target

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
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.header("📥 Employee Inputs")

monthly_salary = st.sidebar.number_input(
    "Monthly Salary",
    min_value=10000,
    value=70000
)

savings = st.sidebar.number_input(
    "Savings Amount",
    min_value=0,
    value=10000
)

rent_expense = st.sidebar.number_input(
    "Rent Expense",
    min_value=0,
    value=12000
)

grocery_expense = st.sidebar.number_input(
    "Grocery Expense",
    min_value=0,
    value=5000
)

emi = st.sidebar.number_input(
    "EMI / Loan Payment",
    min_value=0,
    value=4000
)

other_expenses = st.sidebar.number_input(
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
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Prediction Dashboard",
    "📈 Analytics",
    "💡 Financial Suggestions"
])

# ---------------------------------------------------
# TAB 1
# ---------------------------------------------------

with tab1:

    st.subheader("📌 Financial Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💵 Salary", f"{monthly_salary}")
    col2.metric("💸 Expenses", f"{total_expenditure}")
    col3.metric("📈 Profit", f"{profit}")
    col4.metric("🏦 Savings", f"{savings}")

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

        st.subheader("🧠 Prediction Result")

        if prediction == 1:

            st.success(
                f"✅ Employee is Financially Healthy"
            )

        else:

            st.error(
                f"❌ Employee is Financially Unhealthy"
            )

        st.metric(
            "Confidence Score",
            f"{probability:.2f}"
        )

        # Financial Score Bar

        st.progress(float(probability))

# ---------------------------------------------------
# TAB 2
# ---------------------------------------------------

with tab2:

    st.subheader("📊 Expense Breakdown")

    expense_data = {
        'Rent': rent_expense,
        'Grocery': grocery_expense,
        'EMI': emi,
        'Other': other_expenses
    }

    fig1, ax1 = plt.subplots()

    ax1.pie(
        expense_data.values(),
        labels=expense_data.keys(),
        autopct='%1.1f%%'
    )

    st.pyplot(fig1)

    # Savings vs Expenses

    st.subheader("📈 Savings vs Expenses")

    compare_df = pd.DataFrame({
        'Category': ['Savings', 'Expenses'],
        'Amount': [savings, total_expenditure]
    })

    st.bar_chart(
        compare_df.set_index('Category')
    )

    # Ratios

    st.subheader("📌 Financial Ratios")

    ratio_col1, ratio_col2, ratio_col3 = st.columns(3)

    ratio_col1.metric(
        "Savings Ratio",
        f"{savings_ratio:.2f}"
    )

    ratio_col2.metric(
        "Expense Ratio",
        f"{expense_ratio:.2f}"
    )

    ratio_col3.metric(
        "EMI Burden",
        f"{emi_burden:.2f}"
    )

# ---------------------------------------------------
# TAB 3
# ---------------------------------------------------

with tab3:

    st.subheader("💡 Financial Improvement Suggestions")

    suggestions = []

    if savings_ratio < 0.20:
        suggestions.append(
            "Increase monthly savings to at least 20% of salary."
        )

    if expense_ratio > 0.60:
        suggestions.append(
            "Reduce total expenditure. Current expenses are very high."
        )

    if emi_burden > 0.30:
        suggestions.append(
            "Try reducing loan burden or refinancing EMI."
        )

    if rent_expense > (0.35 * monthly_salary):
        suggestions.append(
            "Rent expense is too high compared to salary."
        )

    if grocery_expense > 10000:
        suggestions.append(
            "Optimize grocery and lifestyle spending."
        )

    if profit < 0:
        suggestions.append(
            "You are operating at a financial loss. Reduce expenses immediately."
        )

    if len(suggestions) == 0:

        st.success(
            "Excellent financial health! Keep maintaining balanced spending and savings habits."
        )

    else:

        for s in suggestions:
            st.warning(s)

    # Financial Summary

    st.subheader("📋 Financial Summary")

    st.write(f"Monthly Salary: {monthly_salary}")
    st.write(f"Total Expenditure: {total_expenditure}")
    st.write(f"Profit: {profit}")
    st.write(f"Savings Amount: {savings}")
    st.write(f"Savings Ratio: {savings_ratio:.2f}")
    st.write(f"Expense Ratio: {expense_ratio:.2f}")
    st.write(f"EMI Burden: {emi_burden:.2f}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Built using Streamlit, Scikit-learn, Pandas, and Logistic Regression"
)
