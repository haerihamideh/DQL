import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, begin, end, test_size, validation_size, train_size):
            self.begin = begin
            self.end = end
            self.test_size = test_size
            self.validation_size = validation_size
            self.train_size = train_size

    def load_data(self, company):
        print(f"Processing data for {company}...")

        PATH = "datasets/FA_StockPrices/"

        # Load and combine data from different time frames
        data_files = [f"{PATH}{company}_{time_frame}.csv" for time_frame in ['day', 'hour', 'week']]
        combined_data = pd.concat([pd.read_csv(file) for file in data_files], ignore_index=True)

        # Filter data based on the provided date range
        combined_data = combined_data[(combined_data['Date'] >= self.begin) & (combined_data['Date'] <= self.end)]

        # Define features (X) and target (y)
        X = combined_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Average_Polarity', 'Reward']]
        y = combined_data['Action']

        # Calculate RoR for each stock based on historical data
        combined_data['RoR'] = self.calculate_ror(combined_data['Open'].iloc[0], combined_data['Close'].iloc[-1])

        # Convert y to 1-dimensional numpy array
        y = y.values

        # Split data into train, test, and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size + self.validation_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_size / (self.train_size + self.validation_size), random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test, combined_data['RoR'].iloc[0]

    def calculate_ror(self, first_price, last_price):
        # Calculate Rate of Return (RoR) for the entire dataset
        ror = (last_price - first_price) / first_price
        return ror
