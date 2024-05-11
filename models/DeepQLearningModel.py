import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

class DeepQLearningModel:
    def __init__(self, directory, outdirectory):
        self.directory = directory
        self.outdirectory = outdirectory

    def preprocess_data(self, df):
        # Drop the 'Date' column
        df = df.drop('Date', axis=1)
        
        # Check the data type of the 'DTYYYYMMDD' column
        if df['DTYYYYMMDD'].dtype == 'object':
            # Convert the 'DTYYYYMMDD' column to a numeric format
            df['DTYYYYMMDD'] = df['DTYYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d').toordinal())

        
        return df

    def generate_state_action(self, df):
        states = []
        actions = []
        rewards = []

        for i in range(len(df) - 1):
            state = df.iloc[i].values
            next_state = df.iloc[i+1].values

            open_price = state[6]  # 7th column (0-indexed)
            close_price = next_state[4]  # 5th column (0-indexed)
            average_polarity = state[-1]  # Last column

            reward = ((close_price - open_price) / open_price) + average_polarity
            rewards.append(reward)

            if next_state[-1] > state[-1]:
                action = 1  # Buy
            elif next_state[-1] < state[-1]:
                action = -1  # Sell
            else:
                action = 0  # Do nothing

            states.append(state)
            actions.append(action)

        # Ensure the length of states, actions, and rewards matches the length of the DataFrame
        if len(states) < len(df):
            states.append(df.iloc[-1].values)
            actions.append(0)  # Assume the last action is 'do nothing'
            rewards.append(0)  # Assume the last reward is 0

        return np.array(states), np.array(actions), np.array(rewards)

    def train_model(self, X, y):
        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], 1)))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        
        return model

    def process_file(self, filename):
        # Load the CSV file
        df = pd.read_csv(os.path.join(self.directory, filename))
        
        # Preprocess the data
        df = self.preprocess_data(df)
        
        # Generate the state-action pairs
        states, actions, rewards = self.generate_state_action(df)
        
        # Train the Deep Q-Learning model with LSTM
        model = self.train_model(states, actions)
        
        # Save the output file
        predictions = model.predict(states).squeeze()
        predictions[predictions > 0] = 1
        predictions[predictions < 0] = -1
        df['Action'] = predictions
        df.to_csv(os.path.join(self.outdirectory, f'output_{filename}'), index=False)
        
        print(f'Processed file: {filename}')
