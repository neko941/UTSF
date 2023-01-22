import numpy as np
from sklearn.metrics import r2_score

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

def R2(y_true, y_pred):
    # return 1 - (np.sum(np.power(y - yhat, 2)) / np.sum(np.power(y - np.mean(y), 2)))
    return r2_score(y_true, y_pred)