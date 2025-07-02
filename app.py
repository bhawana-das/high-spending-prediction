import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Banking Transaction Analysis", layout="wide")

st.title("📚 Bank Transaction Analysis & Prediction App")


# --- 1. Load default data or user file ---
st.sidebar.header("Upload or Use Sample Data")

uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded")
else:
    # Sample default data
    df = pd.DataFrame({
        "transaction_amount (NPR)": np.random.randint(20000, 100000, 100),
        "balance_after_transaction (NPR)": np.random.randint(1500, 8000, 100),
        
    })

# --- 2. Manual Entry ---
st.sidebar.header("➕ Add New Entry")

with st.sidebar.form("manual_form"):
    txn_amt = st.number_input("transaction_amount (NPR)", 1000, 200000, 30000)
    blnc_txn_amt = st.number_input("balance_after_transaction (NPR)", 500, 15000, 2500)

    
    submitted = st.form_submit_button("Add to Dataset")
    if submitted:
        new_row = {
            "transaction_amount (NPR)": txn_amt,
            "balance_after_transaction (NPR)": blnc_txn_amt,

        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.sidebar.success("✅ Entry Added to Data")

# --- 3. Data Preview ---
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- 4. Model Training & Prediction ---
st.subheader("📈 Predict transaction_amount")

features = ['balance_after_transaction (NPR)']
target = 'transaction_amount (NPR)'

# Ensure no NaN in training
df.dropna(inplace=True)

X = df[features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.markdown(f"*Model Accuracy (R² Score)*: {score:.2f}")

    st.markdown("### 🔮 Try Custom Prediction")

    col1, col2 = st.columns(2)
    with col1:
        input_monthly = st.number_input("balance_after_transaction", 1000, 20000, 3000)
        
    if st.button("Predict Amount"):
        prediction = model.predict([[input_monthly]])
        st.success(f"Predicted transaction_amount: NPR {prediction[0]:,.2f}")

except Exception as e:
    st.error("❌ Model training failed. Ensure enough valid data and no missing values.")
    st.exception(e)

# --- 5. Visualizations ---
st.subheader("📉 Data Visualizations")

# 1. Histogram of Annual Tuition Fee
st.markdown("#### Histogram of Transaction Amount")
fig1, ax1 = plt.subplots()
sns.histplot(df['transaction_amount (NPR)'], bins=30, kde=True, ax=ax1)
ax1.set_title("Distribution of transaction_amount")
st.pyplot(fig1)

# 2. Boxplot of Monthly Fee
st.markdown("#### Boxplot of balance_after_transaction")
fig2, ax2 = plt.subplots()
sns.boxplot(x=df['balance_after_transaction (NPR)'], ax=ax2)
ax2.set_title("Boxplot of balance_after_transaction")
st.pyplot(fig2)

# 3. Correlation Heatmap
st.markdown("#### Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Feature Correlation")
st.pyplot(fig3)



st.success("✅ App Ready")


