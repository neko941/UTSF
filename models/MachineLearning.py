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
class LassoRegression(MachineLearningModel):
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
class RidgeRegression(MachineLearningModel):
    def build(self):
        self.model = Ridge(**yaml_load(self.config_path))

from sklearn.linear_model import RidgeClassifier
class RidgeClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = RidgeClassifier(**yaml_load(self.config_path))

from sklearn.linear_model import RidgeCV
class RidgeCrossValidation(MachineLearningModel):
    def build(self):
        self.model = RidgeCV(**yaml_load(self.config_path))

from sklearn.kernel_ridge import KernelRidge
class KernelRidgeRegression(MachineLearningModel):
    def build(self):
        self.model = KernelRidge(**yaml_load(self.config_path))


from sklearn.ensemble import RandomForestRegressor
class RandomForestRegression(MachineLearningModel):
    def build(self):
        self.model = RandomForestRegressor(**yaml_load(self.config_path))

from sklearn.ensemble import RandomForestClassifier
class RandomForestClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = RandomForestClassifier(**yaml_load(self.config_path))

from sklearn.ensemble import GradientBoostingRegressor
class GradientBoostingRegression(MachineLearningModel):
    def build(self):
        self.model = GradientBoostingRegressor(**yaml_load(self.config_path))

from sklearn.ensemble import GradientBoostingClassifier
class GradientBoostingClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = GradientBoostingClassifier(**yaml_load(self.config_path))

from sklearn.ensemble import ExtraTreesRegressor
class ExtraTreesRegression(MachineLearningModel):
    def build(self):
        self.model = ExtraTreesRegressor(**yaml_load(self.config_path))

from sklearn.ensemble import ExtraTreesClassifier
class ExtraTreesClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = ExtraTreesClassifier(**yaml_load(self.config_path))

from sklearn.ensemble import BaggingRegressor
class BaggingRegression(MachineLearningModel):
    def build(self):
        self.model = BaggingRegressor(**yaml_load(self.config_path))

from sklearn.ensemble import BaggingClassifier
class BaggingClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = BaggingClassifier(**yaml_load(self.config_path))

from sklearn.ensemble import AdaBoostRegressor
class AdaBoostRegression(MachineLearningModel):
    def build(self):
        self.model = AdaBoostRegressor(**yaml_load(self.config_path))

from sklearn.ensemble import AdaBoostClassifier
class AdaBoostClassification(MachineLearningModel):
    def build(self):
        self.is_classifier = True
        self.model = AdaBoostClassifier(**yaml_load(self.config_path))

from sklearn.linear_model import OrthogonalMatchingPursuit
class OrthogonalMatchingPursuitRegression(MachineLearningModel):
    def build(self):
        self.model = OrthogonalMatchingPursuit(**yaml_load(self.config_path))

from sklearn.linear_model import OrthogonalMatchingPursuitCV
class OrthogonalMatchingPursuitCrossValidation(MachineLearningModel):
    def build(self):
        self.model = OrthogonalMatchingPursuitCV(**yaml_load(self.config_path))