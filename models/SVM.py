from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from utils.general import yaml_load
from models.Base import MachineLearningModel

class SupportVectorMachinesRegression(MachineLearningModel):
    def __init__(self, config_path, **kwargs):
        self.model = SVR(**yaml_load(config_path))

class LinearSupportVectorMachinesRegression(MachineLearningModel):
    def __init__(self, config_path, **kwargs):
        self.model = LinearSVR(**yaml_load(config_path))