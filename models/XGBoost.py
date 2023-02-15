from xgboost import XGBRegressor
from utils.general import yaml_load
from models.Base import MachineLearningModel

class ExtremeGradientBoostingRegression(MachineLearningModel):
    def __init__(self, config_path, **kwargs):
        self.model = XGBRegressor(**yaml_load(config_path))