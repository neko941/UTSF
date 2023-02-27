from utils.general import yaml_load
from models.Base import MachineLearningModel

from xgboost import XGBRegressor
class ExtremeGradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = XGBRegressor(**yaml_load(self.config_path))

from sklearn.linear_model import LinearRegression
class LinearRegressionWrapper(MachineLearningModel):
    def build(self):
        self.model = LinearRegression(**yaml_load(self.config_path))

from sklearn.linear_model import Lasso
class LassoWrapper(MachineLearningModel):
    def build(self):
        self.model = Lasso(**yaml_load(self.config_path))

from sklearn.svm import SVC
class SupportVectorMachinesClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = SVC(**yaml_load(self.config_path))

from sklearn.svm import SVR
class SupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = SVR(**yaml_load(self.config_path))

from sklearn.svm import NuSVR
class NuSupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = NuSVR(**yaml_load(self.config_path))

from sklearn.svm import LinearSVR
class LinearSupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = LinearSVR(**yaml_load(self.config_path))

from catboost import CatBoostRegressor
class CatBoostRegression(MachineLearningModel):
    def build(self):
        self.model = CatBoostRegressor(**yaml_load(self.config_path))