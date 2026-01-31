import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model
from utils import load_data, prepare_data, inventory_decision

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("🛒 Retail Demand Forecasting using LSTM")
st.markdown("### AI-based Inventory Decision Support System")

# Load model
model = load_model("model/lstm_model.h5")

# Load data
sales, calendar = load_data()

# Sidebar
st.sidebar.header("🔧 Input Parameters")
item_index = st.sidebar.number_input(
    "Select Item Index", min_value=0, max_value=100, value=0
)

current_stock = st.sidebar.number_input(
    "Current Stock Quantity", min_value=0, value=50
)

if st.sidebar.button("🔮 Predict Demand"):
    X, scaler, actual_sales = prepare_data(sales, calendar, item_index)

    prediction = model.predict(X)

    dummy = np.zeros((1, 3))
    dummy[0, 0] = prediction[0][0]
    predicted_demand = scaler.inverse_transform(dummy)[0][0]

    decision = inventory_decision(predicted_demand, current_stock)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📦 Predicted Demand", int(predicted_demand))

    with col2:
        st.metric("📊 Inventory Decision", decision)

    st.subheader("📈 Demand Trend")
    plt.figure(figsize=(8,4))
    plt.plot(actual_sales[-100:], label="Past Sales")
    plt.axhline(predicted_demand, color="red", linestyle="--", label="Predicted")
    plt.legend()
    st.pyplot(plt)
