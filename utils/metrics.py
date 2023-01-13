import numpy as np

def MAE(y_true, y_pred):
    """ Mean Absolute Error """
    return np.mean(np.abs((y_true - y_pred)))

def MSE(y_true, y_pred):
    """ Mean Squared Error """ 
    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((y_true-y_pred)**2))

def MAPE(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    return np.mean(np.abs((y_true-y_pred) / y_true)) * 100