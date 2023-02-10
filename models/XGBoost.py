import os
from xgboost import XGBRegressor
from models.Base import MachineLearningModel

class ExtremeGradientBoosting(MachineLearningModel):
    def __init__(self, **kwargs):
        self.model = XGBRegressor()
        self.name = 'XGBoost'