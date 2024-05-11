import pandas as pd

def calculate_reward(ticker, data):
    df = data.copy()
    
    # Load daily average polarity data
    polarity_data = pd.read_csv(f'datasets/FA_DataSet_XML/{ticker}/daily_average_polarity.csv')
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert 'Date' column in polarity_data to match the format in df
    polarity_data['Date'] = pd.to_datetime(polarity_data['Date'], format='%Y%m%d')
    
    # Merge dataframes based on 'Date' column
    merged_data = pd.merge(df, polarity_data, on='Date', how='left')
    
    # Fill missing values with 0
    merged_data['Average_Polarity'].fillna(0, inplace=True)
    
    # Calculate reward for each row
    merged_data['Reward'] = (merged_data['Close'] - merged_data['Open']) / merged_data['Open'] + merged_data['Average_Polarity']
    
    # Set action to 1 if reward is non-zero, else 0
    merged_data['Action'] = (merged_data['Reward'] > 0).astype(int)
    
    # Analyze if 'Average_Polarity' column in any row is non-zero and print that row
    for index, row in merged_data.iterrows():
        if row['Average_Polarity'] != 0:
            print(row)
    
    return merged_data

# Example usage
# calculate_reward('AAPL', data)
