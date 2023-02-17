from xgboost import XGBRegressor
from utils.general import yaml_load
from models.Base import MachineLearningModel

class ExtremeGradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = XGBRegressor(**yaml_load(self.config_path))