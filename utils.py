import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def load_data():
    sales = pd.read_csv("data/sales_train_validation.csv")
    calendar = pd.read_csv("data/calendar.csv")
    return sales, calendar

def prepare_data(sales, calendar, item_index=0):
    sales_values = sales.iloc[item_index, 6:].values.astype(float)

    calendar["is_weekend"] = calendar["weekday"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    festival = calendar["event_name_1"].notnull().astype(int)
    festival = festival[:len(sales_values)]
    weekend = calendar["is_weekend"][:len(sales_values)]

    df = pd.DataFrame({
        "sales": sales_values,
        "festival": festival.values,
        "weekend": weekend.values
    })

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = np.array([scaled[-30:]])
    return X, scaler, df["sales"]

def inventory_decision(predicted, stock):
    return "REFILL REQUIRED 🚨" if stock < predicted else "NO REFILL ✅"
