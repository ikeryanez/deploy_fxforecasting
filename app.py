import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import matplotlib.pyplot as plt

# Diccionario para mapear los nombres de los pares de divisas
currency_pairs = {
    "EUR/USD": "eurusd",
    "GBP/USD": "gbpusd",
    "JPY/USD": "jpyusd"
}

# Función para cargar el modelo correspondiente
def load_model(currency_pair_key):
    model_path = f"xgb_{currency_pairs[currency_pair_key]}.pkl"
    return joblib.load(model_path)

# Función para obtener datos recientes del mercado forex
def get_forex_data(currency_pair_key):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15)  # Obtener datos de los últimos 15 días para asegurar tener al menos 5 días completos de cierre
    ticker = f"{currency_pairs[currency_pair_key]}=X".replace('/', '')
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data = data[['Close']].rename(columns={'Close': 'close'})
    data = data.dropna()
    return data

# Calcular la variación porcentual y los lags
def prepare_features(data):
    data['pct_change'] = data['close'].pct_change()  # No multiplicar por 100
    for lag in range(1, 6):
        data[f'pct_change_lag_{lag}'] = data['pct_change'].shift(lag)
    data = data.dropna()
    return data

# Realizar la predicción para los próximos 5 días
def predict_next_days(model, data, days=5):
    predictions = []
    last_row = data.iloc[-1].copy()

    for _ in range(days):
        features = last_row[['pct_change_lag_1', 'pct_change_lag_2', 'pct_change_lag_3',
                             'pct_change_lag_4', 'pct_change_lag_5']].values.reshape(1, -1)
        prediction = model.predict(features)[0]
        predictions.append(prediction)

        # Actualizar los lags
        new_lag_row = pd.Series({
            'pct_change': prediction,
            'pct_change_lag_1': last_row['pct_change'],
            'pct_change_lag_2': last_row['pct_change_lag_1'],
            'pct_change_lag_3': last_row['pct_change_lag_2'],
            'pct_change_lag_4': last_row['pct_change_lag_3'],
            'pct_change_lag_5': last_row['pct_change_lag_4']
        })
        last_row = new_lag_row

    return predictions

# Interfaz de Streamlit
st.title("Forex Prediction with XGBRegressor")

currency_pair_key = st.selectbox("Select Currency Pair", list(currency_pairs.keys()))
model = load_model(currency_pair_key)

if st.button('Predict'):
    data = get_forex_data(currency_pair_key)
    data = prepare_features(data)
    predictions = predict_next_days(model, data, days=5)
    
    # Calcular la predicción acumulada de variación porcentual y los nuevos precios de cierre
    predictions_pct = [p * 100 for p in predictions]
    last_close = data['close'].iloc[-1]
    predicted_closes = [last_close * (1 + predictions[0])]
    for i in range(1, len(predictions)):
        predicted_closes.append(predicted_closes[-1] * (1 + predictions[i]))

    # Mostrar la predicción acumulada de variación porcentual en un tamaño grande y con color
    total_prediction_pct = sum(predictions_pct)
    color = 'green' if total_prediction_pct >= 0 else 'red'
    st.markdown(f"<h1>5-Day Prediction: <span style='color: {color};'>{total_prediction_pct:.6f}%</span></h1>", unsafe_allow_html=True)
    
    # Crear un gráfico con los precios de cierre de los últimos 5 días y las predicciones de los siguientes 5 días
    data_last_5_days = data[-5:]
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 6)]
    plt.figure(figsize=(10, 6))
    plt.plot(data_last_5_days['close'], marker='o', label='Closing Price')
    plt.plot(future_dates, predicted_closes, marker='o', color='orange', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Closing Price of {currency_pair_key} and Predictions for the Next 5 Days')
    plt.legend()
    st.pyplot(plt)
else:
    st.write("Press the button to update data and make a prediction")
