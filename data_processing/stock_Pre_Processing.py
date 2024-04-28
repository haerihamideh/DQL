import pandas as pd
import datetime
import os

def stock_Pre_Processing(ticker, start, end, interval):
    '''
    Getting data from a local CSV file
    ticker: Symbol of the stock (not used for local CSV)
    start: start of the range string type
    end: end of the range string type
    interval: not used for local CSV
    ''' 
        # Load data from the CSV file

    filepath = f'datasets/FA_StockPrices/{ticker}.csv'
    # Check if the file exists
    if os.path.exists(filepath):
        # Read the file
        df = pd.read_csv(filepath)
        
        # Convert Date column to datetime format
        df['Date'] = pd.to_datetime(df['<DTYYYYMMDD>'], format='%Y%m%d')
        
        # Filter data based on the specified date range
        start_date = datetime.datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        # Rename columns to match the expected format
        df.rename(columns={'<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close', '<VOL>': 'Volume'}, inplace=True)
        
        # Extract relevant columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    else:
        print(f'File not found: {filename}')

 

