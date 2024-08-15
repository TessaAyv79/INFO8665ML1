import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_test():
    stock_symbol = 'NVDA'
    start_date = '2023-08-16'
    end_date = '2024-08-15'
    
    logging.info(f"Sending request to API for stock: {stock_symbol}")
    logging.info(f"Start Date: {start_date}")
    logging.info(f"End Date: {end_date}")

log_test()