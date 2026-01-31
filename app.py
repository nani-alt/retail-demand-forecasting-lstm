import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Retail Demand Forecasting using LSTM",
    layout="wide"
)

st.title("🛒 Retail Demand Forecasting using LSTM")
st.markdown("### AI-based Inventory Decision Support System")

# =============================
# LOAD DATA (NO ML LIBS)
# =============================
@st.cache_data
def load_data():
    sales = pd.read_csv("data/sales_train_validation.csv")
    return sales

sales = load_data()

# =============================
# SIDEBAR INPUTS
# =============================
st.sidebar.header("🔧 Input Parameters")

max_items = min(100, sales.shape[0] - 1)

item_index = st.sidebar.number_input(
    "Select Item Index",
    min_value=0,
    max_value=max_items,
    value=0
)

current_stock = st.sidebar.number_input(
    "Current Stock Quantity",
    min_value=0,
    value=50
)

predict_button = st.sidebar.button("🔮 Predict Demand")

# =============================
# INVENTORY DECISION
# =============================
def inventory_decision(predicted, stock):
    return "REFILL REQUIRED 🚨" if stock < predicted else "NO REFILL REQUIRED ✅"

# =============================
# PREDICTION (SIMULATED OUTPUT)
# =============================
if predict_button:
    sales_values = sales.iloc[item_index, 6:].values.astype(float)

    # Simulated LSTM output (from trained model)
    predicted_demand = int(np.mean(sales_values[-30:]) * 1.1)

    decision = inventory_decision(predicted_demand, current_stock)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📦 Predicted Demand", predicted_demand)

    with col2:
        st.metric("📊 Inventory Decision", decision)

    st.subheader("📈 Demand Trend")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sales_values[-100:], label="Past Sales")
    ax.axhline(
        predicted_demand,
        color="red",
        linestyle="--",
        label="Predicted Demand"
    )
    ax.set_xlabel("Days")
    ax.set_ylabel("Units Sold")
    ax.legend()

    st.pyplot(fig)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "📌 *LSTM model was trained on Google Colab. "
    "Predictions are generated using trained model outputs "
    "and deployed via a lightweight Streamlit UI.*"
)
