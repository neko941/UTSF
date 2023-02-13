from sklearn.svm import SVR
from utils.general import yaml_load
from models.Base import MachineLearningModel

class SupportVectorMachinesRegression(MachineLearningModel):
    def __init__(self, config_path, **kwargs):
        self.model = SVR(**yaml_load(config_path))
        self.name = 'SVM'