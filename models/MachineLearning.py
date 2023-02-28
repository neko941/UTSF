from utils.general import yaml_load
from models.Base import MachineLearningModel
from xgboost import XGBRegressor

class ExtremeGradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = XGBRegressor(**yaml_load(self.config_path))

from sklearn.linear_model import LinearRegression
class LinearRegression_(MachineLearningModel):
    def build(self):
        self.model = LinearRegression(**yaml_load(self.config_path))

from sklearn.linear_model import Lasso
class Lasso_(MachineLearningModel):
    def build(self):
        self.model = Lasso(**yaml_load(self.config_path))

from sklearn.linear_model import LassoCV
class LassoCrossValidation(MachineLearningModel):
    def build(self):
        self.model = LassoCV(**yaml_load(self.config_path))

# from sklearn.linear_model import MultiTaskLasso
# class MultiTaskLasso_(MachineLearningModel):
#     def build(self):
#         self.model = MultiTaskLasso(**yaml_load(self.config_path))

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

from lightgbm import LGBMRegressor
class LightGBM(MachineLearningModel):
    def build(self):
        self.model = LGBMRegressor(**yaml_load(self.config_path))

from sklearn.linear_model import Ridge
class Ridge_(MachineLearningModel):
    def build(self):
        self.model = Ridge(**yaml_load(self.config_path))

from sklearn.linear_model import RidgeClassifier
class RidgeClassifier_(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = RidgeClassifier(**yaml_load(self.config_path))

from sklearn.linear_model import RidgeCV
class RidgeCrossValidation(MachineLearningModel):
    def build(self):
        self.model = RidgeCV(**yaml_load(self.config_path))

from sklearn.kernel_ridge import KernelRidge
class KernelRidge_(MachineLearningModel):
    def build(self):
        self.model = KernelRidge(**yaml_load(self.config_path))