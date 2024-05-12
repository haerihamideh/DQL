import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, concatenate, Input
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

class StockPriceModelTrainer:
    def __init__(self, directory, outdirectory):
        self.directory = directory
        self.outdirectory = outdirectory

    def preprocess_data(self, df):
        # Drop any rows with missing values
        df.dropna(inplace=True)
        
        # Normalize the features
        scaler = StandardScaler()
        df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
        
        return df

    def train_model_lstm(self, X_train, y_train):
        # Define the LSTM model
        model = Sequential([
            LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(64, activation='relu'),
            Dense(y_train.shape[1])  
        ])
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='mse')
        
        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate MAPE
        n = len(y_test) 
        mape = (1/n) * np.sum(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        
        return mse, rmse, mae, mape

    def calculate_metrics(self, y_test):
        # Calculate ROE (Return on Equity)
        Vf = np.sum(y_test)
        Vi = np.sum(y_test[0])
        ROE = (Vf - Vi) / Vi
        
        # Calculate Drawdown (DD)
        max_dd = np.max(np.maximum.accumulate(y_test) - y_test)
        
        # Calculate Sharpe ratio
        ROR = np.mean(y_test)
        Ra = 0.05  # Assumed annual return without risk
        sigma_a = np.std(y_test)
        sharpe_ratio = (ROR - Ra) / sigma_a
        
        return ROE, max_dd, sharpe_ratio

    def process_file(self, filename):
        # Load the CSV file
        df = pd.read_csv(os.path.join(self.directory, filename))
        
        # Filter columns containing "Return_Rate"
        y_columns = [col for col in df.columns if 'Return_Rate' in col]
        
        # Split the data into features and labels
        X = df.drop(columns=y_columns).values
        y = df[y_columns].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape the input data for LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Train the Deep Q-Learning model using LSTM
        model_lstm = self.train_model_lstm(X_train_reshaped, y_train)
        
        # Evaluate the LSTM model
        mse_lstm, rmse_lstm, mae_lstm, mape_lstm = self.evaluate_model(model_lstm, X_test_reshaped, y_test)
        ROE_lstm, max_dd_lstm, sharpe_ratio_lstm = self.calculate_metrics(y_test)
        
        # Print the metrics for LSTM
        print("\nLSTM Metrics:")
        print(f"MSE for {filename}: {mse_lstm}")
        print(f"RMSE for {filename}: {rmse_lstm}")
        print(f"MAE for {filename}: {mae_lstm}")
        print(f"MAPE for {filename}: {mape_lstm}")
        print(f"ROR for {filename}: {ROE_lstm}")
        print(f"Max Drawdown for {filename}: {max_dd_lstm}")
        print(f"Sharpe Ratio for {filename}: {sharpe_ratio_lstm}")
        
        # Save the LSTM model
        model_lstm.save(os.path.join(self.outdirectory, f'model_lstm_{os.path.splitext(filename)[0]}.h5'))
        
    