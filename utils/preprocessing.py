import numpy as np
from sklearn.preprocessing import RobustScaler

def preprocess_data(X_train, X_test, X_val):
    # Replace infinity or very large values with the maximum finite value
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.max(), inplace=True)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_train.max(), inplace=True)
    X_test_scaled = scaler.transform(X_test)

    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.fillna(X_train.max(), inplace=True)
    X_val_scaled = scaler.transform(X_val)

    # Reshape the data for LSTM (samples, time steps, features)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    return X_train_scaled, X_val_scaled, X_test_scaled
