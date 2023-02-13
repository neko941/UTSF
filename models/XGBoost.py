import os
import pickle
from xgboost import XGBRegressor
from models.Base import MachineLearningModel
from utils.general import yaml_load
from utils.general import abs_path

class ExtremeGradientBoosting(MachineLearningModel):
    def __init__(self, config_path, **kwargs):
        self.model = XGBRegressor(**yaml_load(config_path))
        self.name = 'XGBoost'

    def save(self, save_dir:str='.', file_name:str='XGBoost', extension:str='.pkl'):
        os.makedirs(name=save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name+extension)
        pickle.dump(self.model, open(abs_path(file_path), "wb"))
        return file_path

    def load(self, weight):
        if not os.path.exists(weight): pass
        self.model = pickle.load(open(weight, "rb"))