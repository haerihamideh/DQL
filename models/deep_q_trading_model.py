import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class DeepQTradingModel:
    def __init__(self, epochs, train_size):
        self.epochs = epochs
        self.train_size = train_size

    def train_model(self, X_train_scaled, X_val_scaled, y_train, y_val):
            # Reward function
            def reward_function(y_true, y_pred):
                return np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error as reward

            # LSTM Model
            lstm_model = Sequential([
                LSTM(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
                Dropout(0.2),  # Add Dropout layer to reduce overfitting
                LSTM(128),
                Dropout(0.2),  # Add Dropout layer to reduce overfitting
                Dense(3, activation='softmax')  # Softmax activation for multi-class classification (3 actions: Hold, Long, Short)
            ])
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # GRU Model
            gru_model = Sequential([
                GRU(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
                Dropout(0.2),  # Add Dropout layer to reduce overfitting
                GRU(128),
                Dropout(0.2),  # Add Dropout layer to reduce overfitting
                Dense(3, activation='softmax')  # Softmax activation for multi-class classification (3 actions: Hold, Long, Short)
            ])
            gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Combined Model
            combined_model = Sequential()
            combined_model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))
            combined_model.add(GRU(64))
            combined_model.add(Dense(3, activation='softmax'))
            combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train all models
            def train_with_reward(model, X, y, X_val, y_val):
                rewards = []
                history_losses = []  # To store training loss history
                history_val_losses = []  # To store validation loss history

                for epoch in range(self.epochs):
                    history = model.fit(X, y, batch_size=32, epochs=1, verbose=0, validation_data=(X_val, y_val))

                    # Store training and validation loss for each epoch
                    history_losses.append(history.history['loss'])
                    history_val_losses.append(history.history['val_loss'])

                    # Calculate reward
                    y_pred_val = model.predict(X_val)
                    reward = reward_function(y_val, np.argmax(y_pred_val, axis=1))
                    rewards.append(reward)

                return model, rewards, history_losses, history_val_losses

            lstm_results = train_with_reward(lstm_model, X_train_scaled, y_train, X_val_scaled, y_val)
            gru_results = train_with_reward(gru_model, X_train_scaled, y_train, X_val_scaled, y_val)
            combined_results = train_with_reward(combined_model, X_train_scaled, y_train, X_val_scaled, y_val)

            return lstm_results, gru_results, combined_results


    def evaluate_model(self, model, X_test_scaled, y_test):
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_scaled, y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

        model.summary()

        return loss, accuracy

    def run(self, company, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, RoR):
        results = self.train_model(X_train_scaled, X_val_scaled, y_train, y_val)
        return results

    def plot_results(self, company, *model_results):
        for i, results in enumerate(model_results):
            model_name = f'Model {i+1}'
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


