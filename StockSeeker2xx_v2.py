# Tessa Ayvazoglu
# 13/06/2024
# ML programming project

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas_ta as ta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import timedelta, datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
import logging
import requests
from flask import Flask, request, jsonify
# Flask uygulamasını başlatın
app = Flask(__name__)

# Define popular tickers globally
popular_tickers = ['NFLX', 'JPM', 'NVDA', 'AAPL', 'META', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF']

# Define TODAY as the current date
TODAY = datetime.today()

def configure_logging(log_dir):
    # Get the current date and format it
    today_date = datetime.now().strftime('%Y%m%d')
    log_file_name = f'access_log_{today_date}.log'
    log_file_path = os.path.join(log_dir, log_file_name)

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f'Created log directory: {log_dir}')

    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

    # Create the log file if it does not exist
    if not os.path.exists(log_file_path):
        open(log_file_path, 'w').close()
        print(f'Created log file: {log_file_path}')
    
    print(f'Log file path: {log_file_path}')


@st.cache_resource
def load_and_prepare_data(stock_tickers, start_date, end_date):
    logging.info('Application started')
    logging.info(f'Loading data for tickers: {stock_tickers} from {start_date} to {end_date}')
    data = pd.DataFrame()
    for ticker in stock_tickers:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            cleaned_data = clean_stock_data(df)
            prepared_data = prepare_data_for_analysis(cleaned_data)
            data = pd.concat([data, prepared_data], axis=0)
        except Exception as e:
            logging.error(f'Error processing ticker {ticker}: {e}')
            st.error(f"Failed to process data for ticker {ticker}.")
    return data

def get_stock_data(stock_ticker, start_date, end_date):
    try:
        return yf.download(stock_ticker, start=start_date, end=end_date)
    except Exception as e:
        logging.error(f'Error fetching stock data for {stock_ticker}: {e}')
        st.error(f"Failed to fetch stock data for {stock_ticker}.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def clean_stock_data(data):
    try:
        data = data.fillna(method='ffill')
        data = data.dropna(subset=['Close'])
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        logging.error(f'Error cleaning stock data: {e}')
        st.error("Failed to clean stock data.")
        return pd.DataFrame()

def prepare_data_for_analysis(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data['Daily Return'] = data['Close'].pct_change()
        return data
    except Exception as e:
        logging.error(f'Error preparing data for analysis: {e}')
        st.error("Failed to prepare data for analysis.")
        return pd.DataFrame()

def get_company_name(selected_stock):
    try:
        ticker = yf.Ticker(selected_stock)
        company_name = ticker.info.get('longName', 'Unknown Company')
        return company_name
    except Exception as e:
        logging.error(f'Failed to get company name for {selected_stock}: {e}')
        st.error(f"Failed to get company name for {selected_stock}. Error: {str(e)}")
        return 'Unknown Company'

def setup_streamlit():
    st.header("STOCK SEEKER WEB APP")
    st.subheader("Overview Video")
    st.video(r"C:\Users\Admin\Documents\MLAI\INFO8665ML1\project3\ClipforStocSeekerWeb-highquaalit.mp4")
    st.subheader("Stock Analysis Section")
    st.subheader("About Stock Seeker")
    st.write("""
    Stock Seeker is a powerful web application aimed at helping amateur investors reduce financial losses 
    by providing real-time market data, advanced analytics, and machine learning predictions. 
    We focus on making sophisticated tools accessible and easy to use, so that anyone can make 
    informed investment decisions.
    """)
    st.write("---")

    st.sidebar.title('Connect with Me')
    st.sidebar.markdown("""
    - [LinkedIn](https://www.linkedin.com/in/tessa-nejla-ayvazoglu/)
    - [X (Twitter)](https://x.com/Kristal48712726)
    - [Instagram](https://www.instagram.com/tessa_ayv/)
    - [Facebook](https://www.facebook.com/tessa.ayv)
    - [GitHub](https://github.com/TessaAyv79)
    - [YouTube](https://www.youtube.com/@tessanejlaayv.4561)
    """)

    st.sidebar.subheader("STOCK SEEKER WEB APP")

    selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers, key="stock_tickers")

    if not selected_stocks:
        st.warning("Please select at least one stock ticker.")
        st.stop()  # Akışı durdurur

    logging.info(f'selected_stocks={selected_stocks}')

    try:
        n_years = st.sidebar.slider('Years of prediction for forecast with Prophet model:', 1, 10, key="forecast_period")
        valid_prd = f'{n_years}y'
        valid_periods = [f'{i}y' for i in range(1, 101)] + ['max']
        validate_period(valid_prd, valid_periods)

    except Exception as e:
        logging.error(f"Error in setting valid_prd: {e}")

    logging.info(f'valid_prd={valid_prd}')
    logging.info(f'n_years={n_years}')
    
    period = n_years * 365
    start_datex = TODAY - timedelta(days=period)
    logging.info("start_datex = %s", start_datex)

    start_date = st.sidebar.date_input("Start Date", start_datex, key="start_date")
    end_date = st.sidebar.date_input("End Date", datetime.today(), key="end_date")

    data_load_state = st.text('Loading data...')
    
    data = load_and_prepare_data(selected_stocks, start_date, end_date)

    if data is not None:
        data_load_state.text('Loading data... done!')
        st.subheader('Data')
        st.write(data.tail())
        plot_data(data)
    else:
        st.error("Failed to load data.")

    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Buy/Sell Signals"])
    if analysis_type:
        st.subheader(f"{analysis_type} Analysis")
        handle_analysis(analysis_type, data)
    
    st.subheader('Predictions using Prophet model')
    st.write("**Stock Market Predictions**")
    
    try:
        # Make prediction using local model
        url = 'http://localhost:8501/predict'
        response = requests.post(url, json={'tickers': selected_stocks, 'start_date': start_date, 'end_date': end_date})
        predictions = response.json()
        st.write(predictions)
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        st.error("Failed to get predictions.")
    
def plot_data(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.update_layout(title='Stock Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    except Exception as e:
        logging.error(f'Error plotting data: {e}')
        st.error("Failed to plot data.")

def plot_raw_data(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.update_layout(title='Stock Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    except Exception as e:
        logging.error(f'Error plotting raw data: {e}')
        st.error("Failed to plot raw data.")

def handle_analysis(analysis_type, data):
    try:
        if analysis_type == "Closing Prices":
            plot_data(data)
        elif analysis_type == "Volume":
            plot_volume(data)
        elif analysis_type == "Moving Averages":
            plot_moving_averages(data)
        elif analysis_type == "Buy/Sell Signals":
            plot_buy_sell_signals(data)
    except Exception as e:
        logging.error(f'Error handling analysis {analysis_type}: {e}')
        st.error(f"Failed to handle analysis: {analysis_type}")

def plot_volume(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
        fig.update_layout(title='Volume', xaxis_title='Date', yaxis_title='Volume')
        st.plotly_chart(fig)
    except Exception as e:
        logging.error(f'Error plotting volume: {e}')
        st.error("Failed to plot volume.")

def plot_moving_averages(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=50).mean(), mode='lines', name='50-Day MA'))
        fig.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    except Exception as e:
        logging.error(f'Error plotting moving averages: {e}')
        st.error("Failed to plot moving averages.")

def plot_buy_sell_signals(data):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.update_layout(title='Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
    except Exception as e:
        logging.error(f'Error plotting buy/sell signals: {e}')
        st.error("Failed to plot buy/sell signals.")

def validate_period(valid_prd, valid_periods):
    if valid_prd not in valid_periods:
        raise ValueError(f"Invalid period '{valid_prd}'")

if __name__ == "__main__":
    # Example usage
    log_dir = 'C:/Users/Admin/Documents/MLAI/INFO8665ML1/project2/Fake-Apache-Log-Generator'
    #log_dir = 'logs'  # Ensure this directory exists or change it to a valid directory
    configure_logging(log_dir)
    setup_streamlit()
        
 #

def handle_analysis(selected_stock, analysis_type, start_date, end_date):
    selected_stock = selected_stock[0].upper() if isinstance(selected_stock, list) else selected_stock.upper() 
    #elected_stock = selected_stock[0].upper()
    if analysis_type != "Predicted Prices":
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # df = get_stock_data(selected_stock, start_date, end_date)
        df = data
        if df.empty:
           st.warning(f"No data available for {selected_stock} between {start_date} and {end_date}.")
           return
     
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, df)
    else:
        # Loglama yapılandırması
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

                # Başlangıç logu
                logging.info(f"Option selected: Predicted Prices")
                logging.info(f"Selected stock: {selected_stock}")
                logging.info(f"Date range: {df.index.min()} to {df.index.max()}")
                df = data
                try:
                    # API çağrısı ve yanıtı
                    logging.info(f"Sending request to API for {selected_stock}.")
                    response = requests.post('http://localhost:8501/predict', json={
                        'selected_stock': selected_stock,
                        'start_date': str(df.index.min()),
                        'end_date': str(df.index.max())
                    })

                    if response.status_code == 200:
                        logging.info("Received successful response from API.")
                        result = response.json()

                        # Yanıttan verileri çıkar
                        prediction_dates = pd.to_datetime(result['dates'])
                        predictions = result['predictions']
                        metrics = result['metrics']  # Performans metrikleri
                        plot_url = result.get('plot_url', '')  # Opsiyonel: Kaydedilen grafik URL'si

                        # Metrikleri loglama
                        logging.info(f"Metrics received: MSE: {metrics['MSE']}, MAE: {metrics['MAE']}, R2: {metrics['R2']}, MAPE: {metrics['MAPE']}, Training Time: {metrics['Training Time']} seconds.")

                        # Metrikleri ekranda gösterme
                        st.write(f"Mean Squared Error: {metrics['MSE']}")
                        st.write(f"Mean Absolute Error: {metrics['MAE']}")
                        st.write(f"R^2 Score: {metrics['R2']}")
                        st.write(f"Mean Absolute Percentage Error: {metrics['MAPE']}")
                        st.write(f"Training Time: {metrics['Training Time']} seconds")

                        # Verileri çizme
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines', name='Predicted Price'))
                        fig.update_layout(title=f'{selected_stock} Predicted Prices',
                                        xaxis_title='Date',
                                        yaxis_title='Price')
                        st.plotly_chart(fig)


                        if plot_url:
                            st.markdown(f"[View full plot here]({plot_url})")
                            logging.info(f"Plot available at: {plot_url}")

                        # Diğer model değerlendirmelerini ve karşılaştırmaları yapma
                        logging.info(f"Training and evaluating additional models...")
                        mse_rf, mae_rf, r2_rf, mape_rf, time_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr = train_and_evaluate_model(selected_stock, df, selected_stock)
                        
                        logging.info(f"Collecting model performance data...")
                        # Performans verilerini toplama
                        df_performance = collect_model_performance(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )

                        # Performans tablosunu gösterme
                        logging.info(f"Displaying performance table...")
                        display_performance_table(df_performance)

                        # Performans karşılaştırmasını çizme
                        logging.info(f"Plotting performance comparison...")
                        plot_performance_comparison(df_performance)

                        # Model skorlarını hesaplama ve değerlendirme
                        logging.info(f"Calculating and evaluating model scores...")
                        calculate_score(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )
                        evaluate_model_score(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )
                    else:
                        logging.error(f'Failed to retrieve data, status code: {response.status_code}')
                        st.error(f'Failed to retrieve data, status code: {response.status_code}')
                except KeyError as e:
                    logging.error(f'Missing key in the response: {e}')
                    st.error(f'Missing key in the response: {e}')
                except ValueError as e:
                    logging.error(f'Error processing the response: {e}')
                    st.error(f'Error processing the response: {e}')
                except requests.RequestException as e:
                    logging.error(f'Request failed: {e}')
                    st.error(f'Request failed: {e}')

def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        st.plotly_chart(fig)
        
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        st.plotly_chart(fig)
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        st.plotly_chart(fig)

# Function to display additional information
def display_additional_information(selected_stock, df):
    for option, checked in selected_options.items():
        if checked:
            st.subheader(f"{selected_stock} - {option}")
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    st.write(display_action)
                else:
                    st.write("No data available")
            elif option == "Quarterly Financials":
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    st.write(display_financials)
                else:
                    st.write("No data available")
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    st.write(display_shareholders)
                else:
                    st.write("No data available")
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
                if not display_balancesheet.empty:
                    st.write(display_balancesheet)
                else:
                    st.write("No data available")
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    st.write(display_cashflow)
                else:
                    st.write("No data available")
            elif option == "Analysts Recommendation":
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    st.write(display_analyst_rec)
                else:
                    st.write("No data available")
            elif option == "Predicted Prices":
                # Loglama yapılandırması
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

                # Başlangıç logu
                logging.info(f"Option selected: Predicted Prices")
                logging.info(f"Selected stock: {selected_stock}")
                logging.info(f"Date range: {df.index.min()} to {df.index.max()}")

                try:
                    # API çağrısı ve yanıtı
                    logging.info(f"Sending request to API for {selected_stock}.")
                    response = requests.post('http://localhost:8501/predict', json={
                        'selected_stock': selected_stock,
                        'start_date': str(df.index.min()),
                        'end_date': str(df.index.max())
                    })

                    if response.status_code == 200:
                        logging.info("Received successful response from API.")
                        result = response.json()

                        # Yanıttan verileri çıkar
                        prediction_dates = pd.to_datetime(result['dates'])
                        predictions = result['predictions']
                        metrics = result['metrics']  # Performans metrikleri
                        plot_url = result.get('plot_url', '')  # Opsiyonel: Kaydedilen grafik URL'si

                        # Metrikleri loglama
                        logging.info(f"Metrics received: MSE: {metrics['MSE']}, MAE: {metrics['MAE']}, R2: {metrics['R2']}, MAPE: {metrics['MAPE']}, Training Time: {metrics['Training Time']} seconds.")

                        # Metrikleri ekranda gösterme
                        st.write(f"Mean Squared Error: {metrics['MSE']}")
                        st.write(f"Mean Absolute Error: {metrics['MAE']}")
                        st.write(f"R^2 Score: {metrics['R2']}")
                        st.write(f"Mean Absolute Percentage Error: {metrics['MAPE']}")
                        st.write(f"Training Time: {metrics['Training Time']} seconds")

                        # Verileri çizme
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines', name='Predicted Price'))
                        fig.update_layout(title=f'{selected_stock} Predicted Prices',
                                        xaxis_title='Date',
                                        yaxis_title='Price')
                        st.plotly_chart(fig)

                        if plot_url:
                            st.markdown(f"[View full plot here]({plot_url})")
                            logging.info(f"Plot available at: {plot_url}")

                        # Diğer model değerlendirmelerini ve karşılaştırmaları yapma
                        logging.info(f"Training and evaluating additional models...")
                        mse_rf, mae_rf, r2_rf, mape_rf, time_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr = train_and_evaluate_model(selected_stock, df, selected_stock)
                        
                        logging.info(f"Collecting model performance data...")
                        # Performans verilerini toplama
                        df_performance = collect_model_performance(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )

                        # Performans tablosunu gösterme
                        logging.info(f"Displaying performance table...")
                        display_performance_table(df_performance)

                        # Performans karşılaştırmasını çizme
                        logging.info(f"Plotting performance comparison...")
                        plot_performance_comparison(df_performance)

                        # Model skorlarını hesaplama ve değerlendirme
                        logging.info(f"Calculating and evaluating model scores...")
                        calculate_score(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )
                        evaluate_model_score(
                            mse_lstm=metrics['MSE'], mae_lstm=metrics['MAE'], r2_lstm=metrics['R2'], mape_lstm=metrics['MAPE'], time_lstm=metrics['Training Time'],
                            mse_rf=mse_rf, mae_rf=mae_rf, r2_rf=r2_rf, mape_rf=mape_rf, time_rf=time_rf,
                            mse_gbr=mse_gbr, mae_gbr=mae_gbr, r2_gbr=r2_gbr, mape_gbr=mape_gbr, time_gbr=time_gbr
                        )
                    else:
                        logging.error(f'Failed to retrieve data, status code: {response.status_code}')
                        st.error(f'Failed to retrieve data, status code: {response.status_code}')
                except KeyError as e:
                    logging.error(f'Missing key in the response: {e}')
                    st.error(f'Missing key in the response: {e}')
                except ValueError as e:
                    logging.error(f'Error processing the response: {e}')
                    st.error(f'Error processing the response: {e}')
                except requests.RequestException as e:
                    logging.error(f'Request failed: {e}')
                    st.error(f'Request failed: {e}')
                # # Diğer model metriklerini burada çağırıp döndürebilirsiniz
                # mse_rf, mae_rf, r2_rf, mape_rf, time_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr = train_and_evaluate_model(selected_stock, df, selected_stock)
                # # plot_model_performance(mse_lstm, mae_lstm, r2_lstm, mape_lstm, mse_rf, mae_rf, r2_rf, mape_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr)
                # # Collect performance data
                # # calculate_lstm_evaluation(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm)
                # df_performance = collect_model_performance(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
                #                                            mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
                #                                            mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
                # # Display the performance table
                # display_performance_table(df_performance)
        
                # # Plot the performance comparison
                # plot_performance_comparison(df_performance)
                
                # calculate_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
                #                                            mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
                #                                            mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
                # evaluate_model_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
                #                                            mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
                #                                            mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
                     
        
            # elif option == "Predicted Prices":
            #     mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm = display_predicted_prices(selected_stock, start_date, end_date)
            #     # Diğer model metriklerini burada çağırıp döndürebilirsiniz
            #     mse_rf, mae_rf, r2_rf, mape_rf, time_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr = train_and_evaluate_model(selected_stock, df, selected_stock)
            #     # plot_model_performance(mse_lstm, mae_lstm, r2_lstm, mape_lstm, mse_rf, mae_rf, r2_rf, mape_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr)
            #     # Collect performance data
            #     df_performance = collect_model_performance(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
            #                                                mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
            #                                                mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
            #     # Display the performance table
            #     display_performance_table(df_performance)
        
            #     # Plot the performance comparison
            #     plot_performance_comparison(df_performance)
                
            #     calculate_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
            #                                                mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
            #                                                mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
            #     evaluate_model_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
            #                                                mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
            #                                                mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr)
                 
def calculate_lstm_evaluation(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm):
    # Gerçek fiyat verilerini alalım (bu kısım varsayımsal, sizin veri kaynağınıza göre uyarlayın)
    real_prices = df.loc[prediction_dates]['Close'].values  # Gerçek fiyatların 'Close' kolonundan alınması

    # MSE, MAE, R2, MAPE hesaplamaları
    mse_lstm = mean_squared_error(real_prices, predictions)
    mae_lstm = mean_absolute_error(real_prices, predictions)
    r2_lstm = r2_score(real_prices, predictions)
    mape_lstm = np.mean(np.abs((real_prices - predictions) / real_prices)) * 100
    time_lstm = result.get('time', None)  # Zaman bilgisi tahmin API'sinden alınmış olabilir

    # Sonuçları göster
    st.write(f"MSE: {mse_lstm}")
    st.write(f"MAE: {mae_lstm}")
    st.write(f"R2: {r2_lstm}")
    st.write(f"MAPE: {mape_lstm}")
    if time_lstm:
       st.write(f"Time: {time_lstm} seconds")
    
# Performans hesaplaması
def calculate_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
                mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
                mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr):
    # Ağırlıklar
    weights = {
        'MSE': 15,
        'MAE': 25,
        'R2': 25,
        'MAPE': 25,
        'Time': 10
    }

    # Minimum puanlar
    minimum_scores = {
            'MSE': 7,
            'MAE': 7,
            'R2': 6,
            'MAPE': 6,
        'Time': 3
                    }            
# Normalize metrics
    mse_score = max(0, (1 - mse_lstm / minimum_scores['MSE']) * weights['MSE'])
    mae_score = max(0, (1 - mae_lstm / minimum_scores['MAE']) * weights['MAE'])
    r2_score = max(0, (r2_lstm / minimum_scores['R2']) * weights['R2'])
    mape_score = max(0, (1 - mape_lstm / minimum_scores['MAPE']) * weights['MAPE'])
    time_score = max(0, (1 - time_lstm / 100) * weights['Time'] / minimum_scores['Time'])

    total_score = mse_score + mae_score + r2_score + mape_score + time_score
    return total_score
 
def calculate_bollinger_recommendation(selected_stock, start_date, end_date):
    # Fetch stock data
    stock_data = yf.Ticker(selected_stock)
    stock_history = stock_data.history(period='1d', start=start_date, end=end_date)

    # Check if 'Adj Close' column is present, otherwise use 'Close'
    if 'Adj Close' in stock_history.columns:
        stock_data = stock_history[['Adj Close']].copy()
        stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
    elif 'Close' in stock_history.columns:
        stock_data = stock_history[['Close']].copy()
    else:
        return "Error: Neither 'Adj Close' nor 'Close' columns are available in the data."

    # Parameters
    period = 20

    # Calculate Bollinger Bands
    stock_data['SMA'] = stock_data['Close'].rolling(window=period).mean()
    stock_data['STD'] = stock_data['Close'].rolling(window=period).std()
    stock_data['Upper'] = stock_data['SMA'] + (stock_data['STD'] * 2)
    stock_data['Lower'] = stock_data['SMA'] - (stock_data['STD'] * 2)

    # Prepare new DataFrame for signals
    new_stock_data = stock_data[period-1:].copy()

    # Function to get buy and sell signals
    def get_signal_bb(data):
        buy_signal = []
        sell_signal = []
        
        # Vectorized operations for performance improvement
        buy_signal = np.where(data['Close'] < data['Lower'], data['Close'], np.nan)
        sell_signal = np.where(data['Close'] > data['Upper'], data['Close'], np.nan)
        
        return buy_signal, sell_signal

    # Add buy and sell signals to DataFrame
    new_stock_data['Buy'], new_stock_data['Sell'] = get_signal_bb(new_stock_data)

    # Determine the most recent signal
    latest_data = new_stock_data.iloc[-1]
    if latest_data['Close'] > latest_data['Upper']:
        return "Sell"
    elif latest_data['Close'] < latest_data['Lower']:
        return "Buy"
    else:
        return "Hold"
     
def evaluate_model_score(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm,
                                                           mse_rf, mae_rf, r2_rf, mape_rf, time_rf,
                                                           mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr):
# Her model için puanı hesaplayın
   # Model sonuçları ve hesaplama sürelerini girin
   data_evaluate = {
    'Model': ['Random Forest', 'LSTM', 'Gradient Boosting'],
    'MSE': [mse_rf, mse_lstm, mse_gbr],
    'MAE': [mae_rf, mae_lstm, mae_gbr],
    'R2': [r2_rf, r2_lstm, r2_gbr],
    'MAPE': [mape_rf, mape_lstm, mape_gbr],
    'Time': [time_rf, time_lstm, time_gbr]
    # 'Time': [end_time_rf - start_time_rf, end_time_lstm - start_time_lstm, end_time_gbr - start_time_gbr]
    }


   scores = []
   for i in range(len(data_evaluate['Model'])):
        score = calculate_score(
            data_evaluate['MSE'][i], data_evaluate['MAE'][i], data_evaluate['R2'][i],
            data_evaluate['MAPE'][i], data_evaluate['Time'][i],
            mse_rf if data_evaluate['Model'][i] == 'Random Forest' else 0,
            mae_rf if data_evaluate['Model'][i] == 'Random Forest' else 0,
            r2_rf if data_evaluate['Model'][i] == 'Random Forest' else 0,
            mape_rf if data_evaluate['Model'][i] == 'Random Forest' else 0,
            time_rf if data_evaluate['Model'][i] == 'Random Forest' else 0,
            mse_gbr if data_evaluate['Model'][i] == 'Gradient Boosting' else 0,
            mae_gbr if data_evaluate['Model'][i] == 'Gradient Boosting' else 0,
            r2_gbr if data_evaluate['Model'][i] == 'Gradient Boosting' else 0,
            mape_gbr if data_evaluate['Model'][i] == 'Gradient Boosting' else 0,
            time_gbr if data_evaluate['Model'][i] == 'Gradient Boosting' else 0
        )
        scores.append(score)
     
# Puanları ekrana yazdırın
   for i, model in enumerate(data_evaluate['Model']):
    print(f"{model} Score: {scores[i]}") 
   # Skorları grafiklerde gösterin
   plot_scores(data_evaluate['Model'], scores)     
     
def plot_scores(models, scores):
    # Bar chart
    fig1, ax1 = plt.subplots()
    ax1.bar(models, scores, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Model Scores')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    st.pyplot(fig1)

    # Pie chart
    fig2, ax2 = plt.subplots()
    ax2.pie(scores, labels=models, autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title('Model Score Distribution')  # Title added here
    st.pyplot(fig2)    
                    
def train_and_evaluate_model(selected_stock, df, company_name):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Close']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Regressor
    rf = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    
    start_time_rf = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time_rf = time.time()
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test_scaled)

    # Train Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    
    start_time_gbr = time.time()
    gbr.fit(X_train_scaled, y_train)
    end_time_gbr = time.time()
    
    y_pred_gbr = gbr.predict(X_test_scaled)

    # Calculate performance metrics
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

    mse_gbr = mean_squared_error(y_test, y_pred_gbr)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    r2_gbr = r2_score(y_test, y_pred_gbr)
    mape_gbr = mean_absolute_percentage_error(y_test, y_pred_gbr)
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Actual vs Predicted Prices (Random Forest)
    axs[0].plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', color='blue', alpha=0.7)
    axs[0].plot(df.index[-len(y_test):], y_pred_rf, label='Predicted Prices (RF)', color='green', linestyle='--', alpha=0.7)
    axs[0].legend()
    axs[0].set_title(f'Actual vs Predicted Stock Prices (Random Forest) for {company_name}')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[0].tick_params(axis='x', rotation=45)
    
    # Plot Actual vs Predicted Prices (Gradient Boosting)
    axs[1].plot(df.index[-len(y_test):], y_test.values, label='Actual Prices', color='blue', alpha=0.7)
    axs[1].plot(df.index[-len(y_test):], y_pred_gbr, label='Predicted Prices (GBR)', color='red', linestyle='--', alpha=0.7)
    axs[1].legend()
    axs[1].set_title(f'Actual vs Predicted Stock Prices (Gradient Boosting) for {company_name}')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].tick_params(axis='x', rotation=45)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    time_rf = end_time_rf - start_time_rf 
    time_gbr = end_time_gbr - start_time_gbr
    # Print performance metrics
    # st.subheader(f"Model Performance Metrics for {company_name}")
    # st.write(f"Random Forest Regressor Metrics:")
    # st.write(f"Mean Squared Error (MSE): {mse_rf:.4f}")
    # st.write(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
    # st.write(f"R2 Score: {r2_rf:.4f}")
    # st.write(f"Mean Absolute Percentage Error (MAPE): {mape_rf:.4f}")
    # st.write(f"Training Time: {end_time_rf - start_time_rf:.2f} seconds")
    
    # st.write(f"Gradient Boosting Regressor Metrics:")
    # st.write(f"Mean Squared Error (MSE): {mse_gbr:.4f}")
    # st.write(f"Mean Absolute Error (MAE): {mae_gbr:.4f}")
    # st.write(f"R2 Score: {r2_gbr:.4f}")
    # st.write(f"Mean Absolute Percentage Error (MAPE): {mape_gbr:.4f}")
    # st.write(f"Training Time: {end_time_gbr - start_time_gbr:.2f} seconds")
    # return mse_rf, mae_rf, r2_rf, mape_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr
    return mse_rf, mae_rf, r2_rf, mape_rf, time_rf, mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr 
             
# def display_predicted_LSTM(selected_stock, start_date, end_date):
#     logging.info(f"Starting LSTM prediction for {selected_stock} from {start_date} to {end_date}.")
#     st.subheader(f"{selected_stock} - Predicted Prices")
    
#     # Download historical data
#     logging.info("Downloading historical data.")
#     df = yf.download(selected_stock, start=start_date, end=end_date)
    
#     if df.empty:
#         st.warning(f"No data found for {selected_stock}.")
#         logging.warning(f"No data found for {selected_stock}.")
#         return

#     # Preprocess Data
#     logging.info("Starting data preprocessing.")
#     df = preprocess_data(df)
    
#     logging.info("Performing Exploratory Data Analysis (EDA).")
#     perform_eda(df)
    
#     logging.info("Starting feature engineering.")
#     df = feature_engineering(df)

#     # Initialize the pipeline
#     logging.info("Creating ML pipeline.")
#     pipeline = create_pipeline()
    
#     # Prepare the data
#     logging.info("Preparing training and testing data.")
#     data = df[['Close']].values
#     training_data_len = int(np.ceil(len(data) * .95))
#     train_data = data[:training_data_len]
#     test_data = data[training_data_len - 60:]

#     # Fit and transform training data
#     logging.info("Fitting and transforming training data.")
#     X_train, y_train = pipeline.named_steps['sequence_gen'].transform(
#         pipeline.named_steps['data_prep'].fit_transform(train_data)
#     )

#     # Transform test data
#     logging.info("Transforming test data.")
#     X_test, y_test = pipeline.named_steps['sequence_gen'].transform(
#         pipeline.named_steps['data_prep'].transform(test_data)
#     )
    
#     # Train and evaluate the LSTM model
#     logging.info("Training and evaluating the LSTM model.")
#     mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm = pipeline.named_steps['lstm_model'].fit(X_train, y_train, X_test, y_test)
    
#     # Log and display metrics
#     logging.info(f"LSTM Model Performance - MSE: {mse_lstm}, MAE: {mae_lstm}, R2: {r2_lstm}, MAPE: {mape_lstm}, Time: {time_lstm}s")
#     st.write(f"Mean Squared Error: {mse_lstm}")
#     st.write(f"Mean Absolute Error: {mae_lstm}")
#     st.write(f"R^2 Score: {r2_lstm}")
#     st.write(f"Mean Absolute Percentage Error: {mape_lstm}")
#     st.write(f"Training Time: {time_lstm} seconds")
    
#     # Predict the prices
#     logging.info("Making predictions with the LSTM model.")
#     predictions = pipeline.named_steps['lstm_model'].model.predict(X_test)
#     predictions = pipeline.named_steps['data_prep'].inverse_transform(predictions)
    
#     # Calculate the prediction dates
#     logging.info("Calculating prediction dates.")
#     prediction_dates = pd.date_range(start=df.index[-len(predictions)], periods=len(predictions), freq='B')
    
#     # Plot the data
#     logging.info("Plotting the actual and predicted prices.")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
#     fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
#     fig.update_layout(title=f'{selected_stock} Predicted Prices',
#                       xaxis_title='Date',
#                       yaxis_title='Price')
#     st.plotly_chart(fig)
    
#     logging.info("LSTM prediction and plotting completed.")

# def preprocess_data(df):
#     logging.info("Starting data preprocessing.")
#     df['Date'] = pd.to_datetime(df.index)
#     df.set_index('Date', inplace=True)
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['Weekday'] = df.index.to_series().dt.weekday  # Haftanın günü
    
#     logging.info("Data preprocessing completed.")
#     return df
 
# def perform_eda(df):
#     logging.info("Starting Exploratory Data Analysis (EDA).")
#     st.subheader("Exploratory Data Analysis")
    
#     st.write("Descriptive Statistics")
#     st.write(df.describe())
    
#     st.write("Price vs Date")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
#     fig.update_layout(title="Close Price Over Time",
#                       xaxis_title='Date',
#                       yaxis_title='Close Price')
#     st.plotly_chart(fig)
#     logging.info("Exploratory Data Analysis (EDA) completed.")

# def feature_engineering(df):
#     logging.info("Starting feature engineering.")
#     df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
#     df['Rolling_Std'] = df['Close'].rolling(window=20).std()
#     logging.info("Feature engineering completed.")
    
#     return df

# def create_pipeline():
#     logging.info("Creating ML pipeline.")
#     return Pipeline(steps=[
#         ('data_prep', ColumnTransformer(
#             transformers=[
#                 ('num', StandardScaler(), ['Close'])
#             ],
#             remainder='passthrough'
#         )),
#         ('sequence_gen', SequenceGenerator()),
#         ('lstm_model', LSTMModel())
#     ])

# class SequenceGenerator:
#     def transform(self, data):
#         logging.info("Generating sequences for LSTM model.")
#         X, y = [], []
#         for i in range(len(data) - 60):
#             X.append(data[i:i+60])
#             y.append(data[i+60])
#         logging.info("Sequence generation completed.")
#         return np.array(X), np.array(y)

# class LSTMModel:
#     def __init__(self):
#         logging.info("Initializing LSTM model.")
#         self.model = None
        
#     def fit(self, X_train, y_train, X_test, y_test):
#         logging.info("Building and fitting LSTM model.")
        
#         # Modelin oluşturulması
#         self.model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#             LSTM(50),
#             Dense(1)
#         ])
#         self.model.compile(optimizer='adam', loss='mean_squared_error')
        
#         # Eğitim süresinin ölçülmesi
#         start_time_lstm = time.time()
#         self.model.fit(X_train, y_train, batch_size=1, epochs=1)
#         end_time_lstm = time.time()
#         time_lstm = end_time_lstm - start_time_lstm
        
#         # Tahminlerin yapılması
#         logging.info("Making predictions with the LSTM model.")
#         predictions = self.model.predict(X_test)
        
#         # Performans metriklerinin hesaplanması
#         logging.info("Calculating performance metrics.")
#         mse_lstm = mean_squared_error(y_test, predictions)
#         mae_lstm = mean_absolute_error(y_test, predictions)
#         r2_lstm = r2_score(y_test, predictions)
#         mape_lstm = mean_absolute_percentage_error(y_test, predictions)
        
#         logging.info("LSTM model training and evaluation completed.")
        
#         return mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm


def plot_pie_chart(metric_name, values, model_names):
    # Check for negative or NaN values and handle them
    if any(val < 0 for val in values):
        raise ValueError("Wedge sizes 'x' must be non-negative values")
    
    values = [max(val, 0) for val in values]  # Replace negative values with zero
    if np.any(np.isnan(values)):
        values = [0 if np.isnan(val) else val for val in values]  # Replace NaN values with zero
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot pie chart
    ax.pie(values, labels=model_names, autopct='%1.1f%%', startangle=140, colors=['blue', 'green', 'red'])
    ax.set_title(f'{metric_name} Distribution')
    
    # Use Streamlit's st.pyplot() to render the figure
    st.pyplot(fig)
     
# Example usage
def plot_all_metrics_pie_charts(mse_lstm, mse_rf, mse_gbr,
                                mae_lstm, mae_rf, mae_gbr,
                                r2_lstm, r2_rf, r2_gbr,
                                mape_lstm, mape_rf, mape_gbr):
    # Metrics and their values
    metrics = {
        'Mean Squared Error (MSE)': [mse_lstm, mse_rf, mse_gbr],
        'Mean Absolute Error (MAE)': [mae_lstm, mae_rf, mae_gbr],
        'R2 Score': [r2_lstm, r2_rf, r2_gbr],
        'Mean Absolute Percentage Error (MAPE)': [mape_lstm, mape_rf, mape_gbr]
    }
    
    model_names = ['LSTM', 'Random Forest', 'Gradient Boosting']
    
    for metric_name, values in metrics.items():
        plot_pie_chart(metric_name, values, model_names)
         
def collect_model_performance(mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm, 
                               mse_rf, mae_rf, r2_rf, mape_rf, time_rf, 
                               mse_gbr, mae_gbr, r2_gbr, mape_gbr, time_gbr):
    performance_data = {
        'Model': ['LSTM', 'Random Forest', 'Gradient Boosting'],
        'MSE': [mse_lstm, mse_rf, mse_gbr],
        'MAE': [mae_lstm, mae_rf, mae_gbr],
        'R2 Score': [r2_lstm, r2_rf, r2_gbr],
        'MAPE': [mape_lstm, mape_rf, mape_gbr]
    }
    df_performance = pd.DataFrame(performance_data)
    return df_performance    
    
def display_performance_table(df_performance):
    st.subheader("Overall Model Performance")
    st.dataframe(df_performance)    
    
def plot_performance_comparison(df_performance):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars for each metric
    metrics = ['MSE', 'MAE', 'R2 Score', 'MAPE']
    bar_width = 0.2
    index = range(len(df_performance))

    for i, metric in enumerate(metrics):
        ax.bar(
            [x + i * bar_width for x in index], 
            df_performance[metric],
            bar_width,
            label=metric
        )
    
    # Add labels, title, and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks([x + bar_width for x in index])
    ax.set_xticklabels(df_performance['Model'])
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)  
     
     
       
# Function to detect pivot points
def isPivot(candle, window, df):
    """
    Function that detects if a candle is a pivot/fractal point
    Args:
        candle: Candle index (datetime object)
        window: Number of days before and after the candle to test if pivot
        df: DataFrame containing the stock data
    Returns:
        1 if pivot high, 2 if pivot low, 3 if both, and 0 default
    """
    # Assuming candle is a datetime object
    candle_timestamp = pd.Timestamp(candle)
    if candle_timestamp - timedelta(days=window) < df.index[0] or candle_timestamp + timedelta(days=window) >= df.index[-1]:
        return False
        return 0

    pivotHigh = 1
    pivotLow = 2
    start_index = candle_timestamp - timedelta(days=window)
    end_index = candle_timestamp + timedelta(days=window)
    for i in range((end_index - start_index).days + 1):
        current_date = start_index + timedelta(days=i)
    
        if 'low' in df.columns and df.loc[candle_timestamp, 'low'] > df.loc[current_date, 'low']:
            pivotLow = 0
        if 'high' in df.columns and df.loc[candle_timestamp, 'high'] < df.loc[current_date, 'high']:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

# Function to detect pivot points
def isPivot(candle, window, df):
    """
    Function that detects if a candle is a pivot/fractal point
    Args:
        candle: Candle index (datetime object)
        window: Number of days before and after the candle to test if pivot
        df: DataFrame containing the stock data
    Returns:
        1 if pivot high, 2 if pivot low, 3 if both, and 0 default
    """
    # Assuming candle is a datetime object
    candle_timestamp = pd.Timestamp(candle)
    if candle_timestamp - timedelta(days=window) < df.index[0] or candle_timestamp + timedelta(days=window) >= df.index[-1]:
        return False
        return 0

    pivotHigh = 1
    pivotLow = 2
    start_index = candle_timestamp - timedelta(days=window)
    end_index = candle_timestamp + timedelta(days=window)
    for i in range((end_index - start_index).days + 1):
        current_date = start_index + timedelta(days=i)
    
        if 'low' in df.columns and df.loc[candle_timestamp, 'low'] > df.loc[current_date, 'low']:
            pivotLow = 0
        if 'high' in df.columns and df.loc[candle_timestamp, 'high'] < df.loc[current_date, 'high']:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0
# Function to calculate Chaikin Oscillator
def calculate_chaikin_oscillator(data):
    """
    Calculate Chaikin Oscillator using pandas_ta.
    """
    data['ADL'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])
    data['Chaikin_Oscillator'] = ta.ema(data['ADL'], length=3) - ta.ema(data['ADL'], length=10)
    return data

# Define the calculate_stochastic_oscillator function
def calculate_stochastic_oscillator(df, period=14):
    """
    Calculate Stochastic Oscillator (%K and %D).
    """
    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max() 
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    return df

def chart_stochastic_oscillator_and_price(ticker, df):
    """
    Plots the stock's closing price with its 50-day and 200-day moving averages,
    and the Stochastic Oscillator (%K and %D) below the price chart.
    """
    plt.figure(figsize=[16, 8])
    plt.style.use('default')
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(16, 8))
    fig.suptitle(ticker, fontsize=16)

    # Plotting the closing price and moving averages on the first subplot
    ax[0].plot(df['Close'], color='black', linewidth=1, label='Close')
    ax[0].plot(df['ma50'], color='blue', linewidth=1, linestyle='--', label='50-day MA')
    ax[0].plot(df['ma200'], color='red', linewidth=1, linestyle='--', label='200-day MA')
    ax[0].set_ylabel('Price [\$]')
    ax[0].grid(True)
    ax[0].legend(loc='upper left')
    ax[0].axes.get_xaxis().set_visible(False)  # Hide X axis labels for the price plot

    # Plotting the Stochastic Oscillator on the second subplot
    ax[1].plot(df.index, df['%K'], color='orange', linewidth=1, label='%K')
    ax[1].plot(df.index, df['%D'], color='grey', linewidth=1, label='%D')
    ax[1].grid(True)
    ax[1].set_ylabel('Stochastic Oscillator')
    ax[1].set_ylim(0, 100)
    ax[1].axhline(y=80, color='b', linestyle='-')  # Overbought threshold
    ax[1].axhline(y=20, color='r', linestyle='-')  # Oversold threshold
    ax[1].legend(loc='upper left')

    # Formatting the date labels on the X-axis
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Adjust space between the plots

    st.pyplot(fig)  # Display the plot in Streamlit
    return data

def display_technical_summary(selected_stock, start_date, end_date):
    st.subheader(f"{selected_stock} - Technical Summary")
    
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    # Calculate Chaikin Oscillator
    stock_df = calculate_chaikin_oscillator(stock_df)
    stock_df = calculate_stochastic_oscillator(stock_df)

    # Detect pivot points
    window = 5
    stock_df['isPivot'] = stock_df.apply(lambda x: isPivot(x.name, window, stock_df), axis=1)
    stock_df['pointpos'] = stock_df.apply(lambda row: row['Low'] - 1e-3 if row['isPivot'] == 2 else (row['High'] + 1e-3 if row['isPivot'] == 1 else np.nan), axis=1)

    # Plot candlestick with pivots
    fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                                         open=stock_df['Open'],
                                         high=stock_df['High'],
                                         low=stock_df['Low'],
                                         close=stock_df['Close'],
                                         name='Candlestick')])
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['pointpos'], mode='markers',
                             marker=dict(size=5, color="MediumPurple"),
                             name="Pivot"))
    
    fig.update_layout(title=f'{selected_stock} Candlestick Chart with Pivots',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Plot Chaikin Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Chaikin_Oscillator'], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title=f'{selected_stock} Chaikin Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Chaikin Oscillator Value')
    st.plotly_chart(fig)
    # Plot Stochastic Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%K'], mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['%D'], mode='lines', name='%D'))
    fig.update_layout(title=f'{selected_stock} Stochastic Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Stochastic Oscillator Value')
    st.plotly_chart(fig)
# Define the display_advanced_analysis function
def display_advanced_analysis(selected_stock, start_date, end_date):
    
    st.subheader(f"Advanced Analysis for {selected_stock}")

    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)

    # Add Moving Average Convergence Divergence (MACD)
    df['12 Day EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26 Day EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12 Day EMA'] - df['26 Day EMA']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # MACD Buy/Sell Signals
    df['MACD_Buy_Signal'] = np.where(df['MACD'] > df['Signal Line'], df['MACD'], np.nan)
    df['MACD_Sell_Signal'] = np.where(df['MACD'] < df['Signal Line'], df['MACD'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax.plot(df.index, df['Signal Line'], label='Signal Line', color='red')
    ax.scatter(df.index, df['MACD_Buy_Signal'], marker='^', color='g', label='MACD Buy Signal')
    ax.scatter(df.index, df['MACD_Sell_Signal'], marker='v', color='r', label='MACD Sell Signal')
    ax.set_title(f'MACD for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format       
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)

    # Add Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # RSI Buy/Sell Signals
    df['RSI_Buy_Signal'] = np.where(df['RSI'] < 30, df['RSI'], np.nan)
    df['RSI_Sell_Signal'] = np.where(df['RSI'] > 70, df['RSI'], np.nan)

    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax.axhline(70, linestyle='--', alpha=0.5, color='green')
    ax.scatter(df.index, df['RSI_Buy_Signal'], marker='^', color='g', label='RSI Buy Signal')
    ax.scatter(df.index, df['RSI_Sell_Signal'], marker='v', color='r', label='RSI Sell Signal')
    ax.set_title(f'RSI for {selected_stock}')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # Format the date
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    st.pyplot(fig)
def stochastic_calculator(selected_stock, start_date, end_date):
    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Stochastic Oscillator (%K and %D)
    high14 = df['High'].rolling(window=14).max()
    low14 = df['Low'].rolling(window=14).min()
    df['%K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['%D'] = df['%K'].rolling(window=3).mean()

    fig, axs = plt.subplots(2, figsize=(12, 8), sharex=True)

    # Plotting the closing prices and moving averages
    axs[0].plot(df.index, df['Close'], label='Closing Price', color='blue')
    axs[0].plot(df.index, df['MA50'], label='50-day MA', color='red')
    axs[0].plot(df.index, df['MA200'], label='200-day MA', color='green')
    axs[0].set_ylabel('Price')
    axs[0].legend(loc='upper left')
    axs[0].set_title(f'{selected_stock} - Closing Prices and Moving Averages')

    # Plotting the Stochastic Oscillator
    axs[1].plot(df.index, df['%K'], label='%K', color='blue')
    axs[1].plot(df.index, df['%D'], label='%D', color='red')
    axs[1].axhline(y=20, color='gray', linestyle='--')
    axs[1].axhline(y=80, color='gray', linestyle='--')
    axs[1].set_ylabel('Oscillator')
    axs[1].set_title('Stochastic Oscillator')
    axs[1].legend(loc='upper left')

    date_format = DateFormatter("%Y-%m-%d")
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Define the function for buy/sell signals based on EMAs
def buy_sell_ewma3(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False
    print(type(data))
    print(data.columns)
    for i in range(len(data)):
        if data['Middle'].iloc[i] < data['Long'].iloc[i] and data['Short'].iloc[i] < data['Middle'].iloc[i] and not flag_long and not flag_short:
            buy_list.append(data['Adj Close'].iloc[i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short and data['Short'].iloc[i] > data['Middle'].iloc[i]:
            sell_list.append(data['Adj Close'].iloc[i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'].iloc[i] > data['Long'].iloc[i] and data['Short'].iloc[i] > data['Middle'].iloc[i] and not flag_long and not flag_short:
            buy_list.append(data['Adj Close'].iloc[i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long and data['Short'].iloc[i] < data['Middle'].iloc[i]:
            sell_list.append(data['Adj Close'].iloc[i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return buy_list, sell_list


# Define the function for plotting buy/sell signals and EMAs
def buy_sell_ewma3_plot(data, ticker, title_txt, label_txt):
    print(type(data))
    print(data.columns)
    sns.set(rc={'figure.figsize':(18, 10)})
    plt.plot(data['Adj Close'], label=f"{label_txt}", color='blue', alpha=0.35)
    plt.plot(data['Short'], label='Short/Fast EMA', color='red', alpha=0.35)
    plt.plot(data['Middle'], label='Middle/Medium EMA', color='orange', alpha=0.35)
    plt.plot(data['Long'], label='Long/Slow EMA', color='green', alpha=0.35)
    plt.scatter(data.index, data['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(data.index, data['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title(f"{title_txt}", color='black', fontsize=20)
    plt.xlabel('Date', color='black', fontsize=15)
    plt.ylabel('Stock Price (p)', color='black', fontsize=15)
    plt.legend()
    st.pyplot(plt)

# Streamlit app
def ema_buy_sel_grph(selected_stock, start_date, end_date):
    st.title("Stock Trading Signals and EMAs")
    
    # Fetch stock data
    stock_data = yf.Ticker(selected_stock)
    stock_history = stock_data.history(period='1d', start=start_date, end=end_date)
    
    # Check if 'Adj Close' column is present, otherwise use 'Close'
    if 'Adj Close' in stock_history.columns:
        amzn_adj_6mo = stock_history[['Adj Close']]
    elif 'Close' in stock_history.columns:
        amzn_adj_6mo = stock_history[['Close']]
    else:
        st.error("Neither 'Adj Close' nor 'Close' columns are available in the data.")
        return

    # Rename 'Close' to 'Adj Close' if needed
    if 'Close' in amzn_adj_6mo.columns:
        amzn_adj_6mo.rename(columns={'Close': 'Adj Close'}, inplace=True)
    
    # Calculate EMAs
    amzn_adj_6mo['Short'] = amzn_adj_6mo['Adj Close'].ewm(span=5, adjust=False).mean()
    amzn_adj_6mo['Middle'] = amzn_adj_6mo['Adj Close'].ewm(span=21, adjust=False).mean()
    amzn_adj_6mo['Long'] = amzn_adj_6mo['Adj Close'].ewm(span=63, adjust=False).mean()

    # Generate buy/sell signals
    buy_signals, sell_signals = buy_sell_ewma3(amzn_adj_6mo)
    amzn_adj_6mo['Buy'] = buy_signals
    amzn_adj_6mo['Sell'] = sell_signals

    # Set ticker and title
    ticker = selected_stock
    title_txt = f"Trading signals for {ticker} stock"
    label_txt = f"{ticker} Adj Close"

    # Plot using Streamlit
    buy_sell_ewma3_plot(amzn_adj_6mo, ticker, title_txt, label_txt)
# Define a function to calculate and plot Bollinger Bands with buy/sell signals
# V11 - usecase2 pipeline's example
def bollinger_bands_plot(selected_stock, start_date, end_date):
    # Existing code ...

    # Get recommendation from the new function
    recommendation = calculate_bollinger_recommendation(selected_stock, start_date, end_date)

    # Display recommendation
    st.title("📈 Trading Recommendation for NFLX 📉")
    st.markdown(f"""
    <div style="background-color: #e9f5f5; border-radius: 10px; padding: 20px; text-align: center;">
        <h2 style="color: #1f77b4;">
            <span style="font-size: 50px;">⚡</span> Current Recommendation <span style="font-size: 50px;">⚡</span>
        </h2>
        <h1 style="font-size: 72px; color: #d62728; font-weight: bold;">{recommendation.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

# Example Streamlit application to call the function
def bolinger_main(selected_stock, start_date, end_date):
    st.title("Bollinger Bands and Trading Signals")
    bollinger_bands_plot(selected_stock, start_date, end_date)

# Execute technical summary when summary button is clicked
if summary_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            display_technical_summary(selected_stock, start_date, end_date)
            display_advanced_analysis(selected_stock, start_date, end_date)   
            stochastic_calculator(selected_stock, start_date, end_date)
            ema_buy_sel_grph(selected_stock, start_date, end_date)
            bolinger_main(selected_stock, start_date, end_date)
    else:
        st.sidebar.warning("Please select at least one stock ticker.")
# def bollinger_bands_plot(selected_stock, start_date, end_date):
#     # Fetch stock data
#     stock_data = yf.Ticker(selected_stock)
#     stock_history = stock_data.history(period='1d', start=start_date, end=end_date)

#     # Check if 'Adj Close' column is present, otherwise use 'Close'
#     if 'Adj Close' in stock_history.columns:
#         stock_data = stock_history[['Adj Close']].copy()
#         stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
#     elif 'Close' in stock_history.columns:
#         stock_data = stock_history[['Close']].copy()
#     else:
#         st.error("Neither 'Adj Close' nor 'Close' columns are available in the data.")
#         return

#     # Parameters
#     period = 20

#     # Calculate Bollinger Bands
#     stock_data['SMA'] = stock_data['Close'].rolling(window=period).mean()
#     stock_data['STD'] = stock_data['Close'].rolling(window=period).std()
#     stock_data['Upper'] = stock_data['SMA'] + (stock_data['STD'] * 2)
#     stock_data['Lower'] = stock_data['SMA'] - (stock_data['STD'] * 2)

#     # Prepare new DataFrame for signals
#     new_stock_data = stock_data[period-1:].copy()

#     # Function to get buy and sell signals
#     # Function to get buy and sell signals
#     def get_signal_bb(data):
#         buy_signal = []
#         sell_signal = []

#         for i in range(len(data['Close'])):
#             if data['Close'][i] > data['Upper'][i]:
#                 buy_signal.append(np.nan)
#                 sell_signal.append(data['Close'][i])
#             elif data['Close'][i] < data['Lower'][i]:
#                 sell_signal.append(np.nan)
#                 buy_signal.append(data['Close'][i])
#             else:
#                 buy_signal.append(np.nan)
#                 sell_signal.append(np.nan)

#         return buy_signal, sell_signal

#     # Add buy and sell signals to DataFrame
#     new_stock_data['Buy'] = get_signal_bb(new_stock_data)[0]
#     new_stock_data['Sell'] = get_signal_bb(new_stock_data)[1]

#     # Determine the most recent signal
#     latest_data = new_stock_data.iloc[-1]
#     recommendation = ""
#     if latest_data['Close'] > latest_data['Upper']:
#         recommendation = "Sell"
#     elif latest_data['Close'] < latest_data['Lower']:
#         recommendation = "Buy"
#     else:
#         recommendation = "Hold"

#     # Plot Bollinger Bands
#     def plot_bollinger_bands():
#         fig, ax = plt.subplots(figsize=(20, 10))
#         x_axis = new_stock_data.index
#         ax.fill_between(x_axis, new_stock_data['Upper'], new_stock_data['Lower'], color='grey', alpha=0.3)
#         ax.plot(x_axis, new_stock_data['Close'], color='gold', lw=2, label='Close Price')
#         ax.plot(x_axis, new_stock_data['SMA'], color='blue', lw=2, label='Simple Moving Average')
#         ax.set_title(f'Bollinger Band For {selected_stock}', color='black', fontsize=20)
#         ax.set_xlabel('Date', color='black', fontsize=15)
#         ax.set_ylabel('Close Price', color='black', fontsize=15)
#         plt.xticks(rotation=45)
#         ax.legend()
#         st.pyplot(fig)

#     # Plot Bollinger Bands with buy/sell signals
#     def plot_bollinger_bands_with_signals():
#         fig, ax = plt.subplots(figsize=(20, 10))
#         x_axis = new_stock_data.index
#         ax.fill_between(x_axis, new_stock_data['Upper'], new_stock_data['Lower'], color='grey', alpha=0.3)
#         ax.plot(x_axis, new_stock_data['Close'], color='gold', lw=2, label='Close Price', alpha=0.5)
#         ax.plot(x_axis, new_stock_data['SMA'], color='blue', lw=2, label='Moving Average', alpha=0.5)
#         ax.scatter(x_axis, new_stock_data['Buy'], color='green', lw=3, label='Buy', marker='^', alpha=1)
#         ax.scatter(x_axis, new_stock_data['Sell'], color='red', lw=3, label='Sell', marker='v', alpha=1)
#         ax.set_title(f'Bollinger Band, Close Price, MA and Trading Signals for {selected_stock}', color='black', fontsize=20)
#         ax.set_xlabel('Date', color='black', fontsize=15)
#         ax.set_ylabel('Close Price', color='black', fontsize=15)
#         plt.xticks(rotation=45)
#         ax.legend()
#         st.pyplot(fig)

#     # Plot the data
#     plot_bollinger_bands()
#     plot_bollinger_bands_with_signals()

#     # # Display recommendation
#     # st.subheader(f"Trading Recommendation for {selected_stock}")
#     # st.write(f"**Current Recommendation:** {recommendation}") 
    
#     # Başlık
     
#     st.title("📈 Trading Recommendation for NFLX 📉")

#     # Büyük formatta öneri ve simge
 
#     st.markdown(f"""
#     <div style="background-color: #e9f5f5; border-radius: 10px; padding: 20px; text-align: center;">
#         <h2 style="color: #1f77b4;">
#             <span style="font-size: 50px;">⚡</span> Current Recommendation <span style="font-size: 50px;">⚡</span>
#         </h2>
#         <h1 style="font-size: 72px; color: #d62728; font-weight: bold;">{recommendation.upper()}</h1>
#     </div>
#     """, unsafe_allow_html=True)
    


# Example Streamlit application to call the function
def bolinger_main(selected_stock, start_date, end_date):
    st.title("Bollinger Bands and Trading Signals")
    bollinger_bands_plot(selected_stock, start_date, end_date)

# # Streamlit button event handling
# if summary_clicked:
#     if selected_stocks:
#         for selected_stock in selected_stocks:
#             display_technical_summary(selected_stock, start_date, end_date)
#             display_advanced_analysis(selected_stock, start_date, end_date)   
#             stochastic_calculator(selected_stock, start_date, end_date)
#             ema_buy_sel_grph(selected_stock, start_date, end_date)
#             buy_sell_ewma3(selected_stock)
#     else:
#         st.sidebar.warning("Please select at least one stock ticker.")
# Validate period before analysis
# valid_prd1 = validate_period(valid_prd, valid_periods)

# if valid_prd1:
#     # Execute analysis when button is clicked
#     if button_clicked:
#         if selected_stocks:
#             for selected_stock in selected_stocks:
#                 handle_analysis(selected_stock, analysis_type, start_date, end_date, valid_prd)
#         else:
#             st.sidebar.warning("Please select at least one stock ticker.")



# # Execute technical summary when summary button is clicked
if button_clicked:
        if selected_stocks:
            for selected_stock in selected_stocks:
                handle_analysis(selected_stock, analysis_type, start_date, end_date)
        else:
            st.sidebar.warning("Please select at least one stock ticker.")
             
if summary_clicked:
    if selected_stocks:
        for selected_stock in selected_stocks:
            display_technical_summary(selected_stock, start_date, end_date)
            display_advanced_analysis(selected_stock, start_date, end_date)   
            stochastic_calculator(selected_stock, start_date, end_date)
            ema_buy_sel_grph(selected_stock, start_date, end_date)
            bolinger_main(selected_stock, start_date, end_date)
# Define the stochastic_calculator function
    else:
        st.sidebar.warning("Please select at least one stock ticker.")

# Feedback Mechanism
st.sidebar.title('Feedback & Support')
feedback_form = st.sidebar.form(key='feedback_form')
feedback_text = feedback_form.text_area('Share your feedback or report an issue:', height=150)
email_input = feedback_form.text_input('Your email (optional):')
submit_button = feedback_form.form_submit_button(label='Submit Feedback')

# print the first few rows and the types of the columns
if submit_button:
    if feedback_text.strip() == '':
        st.sidebar.error("Feedback cannot be empty. Please provide your feedback.")
    else:
        # Send feedback to a designated email address or store it in a database
        st.sidebar.success("Thank you for your feedback! We'll review it and take necessary actions.")
        # if conn:
        #   save_data_to_db2()
         
        if email_input.strip() != '':
            # Send email notification if email is provided
            st.sidebar.write(f"A confirmation email has been sent to {email_input}.")

