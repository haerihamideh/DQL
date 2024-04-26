import pandas as pd

def calculate_reward(ticker,data):
    df = data.copy()
    
    # Load daily average polarity data
   
    polarity_data = pd.read_csv( f'datasets/FA_DataSet_XML/{ticker}/daily_average_polarity.csv')
    
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
    return merged_data

    '''
    df = data.copy()
    reward = []
    action = []
    for i in range(len(df)):
        reward_day = (df["Close"][i] - df["Open"][i]) / df["Open"][i]
        reward_day = round(reward_day, 2)
        action_day = 1
        if reward_day < 0:
            reward_day = 0
            action_day = 0
        reward.append(reward_day)
        action.append(action_day)
    df["Reward"] = reward
    df["Action"] = action
    
    return df
    '''