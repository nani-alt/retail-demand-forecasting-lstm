import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

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
# SAFE PATH SETUP
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "lstm_model.h5")
DATA_DIR = os.path.join(BASE_DIR, "data")

# =============================
# LOAD LSTM MODEL (SAFE)
# =============================
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH, compile=False)

model = load_lstm_model()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    sales = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
    calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    return sales, calendar

sales, calendar = load_data()

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
# FEATURE ENGINEERING
# =============================
def prepare_input_data(item_index):
    sales_values = sales.iloc[item_index, 6:].values.astype(float)

    cal = calendar.copy()
    cal["is_weekend"] = cal["weekday"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    festival = cal["event_name_1"].notnull().astype(int)

    festival = festival[:len(sales_values)]
    weekend = cal["is_weekend"][:len(sales_values)]

    df = pd.DataFrame({
        "sales": sales_values,
        "festival": festival.values,
        "weekend": weekend.values
    })

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Last 30 days for prediction
    X = np.array([scaled_data[-30:]])

    return X, scaler, df["sales"]

# =============================
# INVENTORY DECISION LOGIC
# =============================
def inventory_decision(predicted, stock):
    if stock < predicted:
        return "REFILL REQUIRED 🚨"
    else:
        return "NO REFILL REQUIRED ✅"

# =============================
# PREDICTION & OUTPUT
# =============================
if predict_button:
    X, scaler, actual_sales = prepare_input_data(item_index)

    prediction = model.predict(X, verbose=0)

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

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(actual_sales[-100:], label="Past Sales")
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
    "📌 *LSTM model trained on historical retail sales data with festival and weekend effects. "
    "Model training performed on Google Colab and deployed using Streamlit Cloud.*"
)
