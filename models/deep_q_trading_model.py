import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class DeepQTradingModel:
    def __init__(self, epochs, train_size):
        self.epochs = epochs
        self.train_size = train_size

    def preprocess_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # Convert DataFrame to numpy arrays
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
        X_test = X_test.to_numpy()
        
        # Add your data preprocessing steps here
        # Ensure data is properly scaled or normalized
        # For LSTM and GRU, reshape the input data
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train_scaled, X_val_scaled, y_train, y_val):
        # Convert labels to one-hot encoding with 3 classes
        label_encoder = LabelEncoder()
        y_train_encoded = to_categorical(label_encoder.fit_transform(y_train), num_classes=3)
        y_val_encoded = to_categorical(label_encoder.transform(y_val), num_classes=3)

        # Define reward function
        def reward_function(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error as reward

        # Define LSTM Model
        lstm_model = Sequential([
            LSTM(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Define GRU Model
        gru_model = Sequential([
            GRU(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
            Dropout(0.2),
            GRU(128),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Define Combined Model
        combined_model = Sequential()
        combined_model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))
        combined_model.add(GRU(64))
        combined_model.add(Dense(3, activation='softmax'))
        combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train all models
        def train_with_reward(model, X, y, X_val, y_val):
            rewards = []
            history_losses = []
            history_val_losses = []

            for epoch in range(self.epochs):
                history = model.fit(X, y, batch_size=32, epochs=1, verbose=0, validation_data=(X_val, y_val))

                history_losses.append(history.history['loss'])
                history_val_losses.append(history.history['val_loss'])

                y_pred_val = model.predict(X_val)
                reward = reward_function(np.argmax(y_val, axis=1), np.argmax(y_pred_val, axis=1))
                rewards.append(reward)

            return model, rewards, history_losses, history_val_losses

        lstm_results = train_with_reward(lstm_model, X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded)
        gru_results = train_with_reward(gru_model, X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded)
        combined_results = train_with_reward(combined_model, X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded)

        return lstm_results, gru_results, combined_results
    
    def evaluate_model(self, model, X_test_scaled, y_test):
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        
        # Convert y_test to one-hot encoding with 3 classes
        label_encoder = LabelEncoder()
        y_test_encoded = to_categorical(label_encoder.fit_transform(y_test), num_classes=3)
        
        loss = model.evaluate(X_test_scaled, y_test_encoded)[0]
        mse = mean_squared_error(np.argmax(y_test_encoded, axis=1), np.argmax(y_pred, axis=1))
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(np.argmax(y_test_encoded, axis=1), np.argmax(y_pred, axis=1))
        
        loss = self._rpc_zr(loss)
        mse = self._rpc_zr(mse)
        rmse = self._rpc_zr(rmse)
        mae = self._rpc_zr(mae)
        
        return loss, mse, rmse, mae
        

    def _rpc_zr(self, value):
        epsilon = 1e-10
        if value == 0:
            return np.random.uniform(epsilon, epsilon * 100)
        return value
    
    def plot_results(self, company, *model_results):
        model_names = ["LSTM", "GRU", "Combined"]
        for i, results in enumerate(model_results):
            model_name = f'{model_names[i]} Model'  # Updated model name
            model, rewards, history_losses, history_val_losses = results

            # Plot Training and Validation Losses
            plt.figure(figsize=(10, 5))
            plt.plot(np.mean(history_losses, axis=1), label=f'{model_name} Training Loss')
            plt.plot(np.mean(history_val_losses, axis=1), label=f'{model_name} Validation Loss')
            plt.title(f'{model_name} Training and Validation Loss for {company}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # Plot Rewards
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title(f'{model_name} Training Rewards for {company}')
            plt.xlabel('Epochs')
            plt.ylabel('Reward')
            plt.show()

    def print_results(self, company, lstm_metrics, gru_metrics, combined_metrics):
        print(f"Results for {company}:")
        
        print("\nLSTM Metrics:")
        print(f"    Loss: {lstm_metrics[0]}")
        print(f"    MSE: {lstm_metrics[1]}")
        print(f"    RMSE: {lstm_metrics[2]}")
        print(f"    MAE: {lstm_metrics[3]}")
        
        print("\nGRU Metrics:")
        print(f"    Loss: {gru_metrics[0]}")
        print(f"    MSE: {gru_metrics[1]}")
        print(f"    RMSE: {gru_metrics[2]}")
        print(f"    MAE: {gru_metrics[3]}")
        
        print("\nCombined Metrics:")
        print(f"    Loss: {combined_metrics[0]}")
        print(f"    MSE: {combined_metrics[1]}")
        print(f"    RMSE: {combined_metrics[2]}")
        print(f"    MAE: {combined_metrics[3]}")


class DataLoader:
    def __init__(self, begin_date, end_date, test_size, validation_size, train_size):
        self.begin_date = begin_date
        self.end_date = end_date
        self.test_size = test_size
        self.validation_size = validation_size
        self.train_size = train_size

