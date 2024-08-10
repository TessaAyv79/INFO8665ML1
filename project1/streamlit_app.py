import streamlit as st
import requests

st.title("Stock Price Prediction")

selected_stock = st.text_input("Enter stock symbol (e.g., AAPL):")
start_date = st.date_input("Start date")
end_date = st.date_input("End date")

if st.button("Get Predictions"):
    response = requests.post('http://localhost:8502/api/predict', json={
        'ticker': selected_stock,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    })
    data = response.json()
    st.write(data)  # Display predictions here