import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between y_true and y_pred.
    
    Parameters:
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns:
    mape : float
        Mean absolute percentage error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
