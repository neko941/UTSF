import numpy as np

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from utils.general import yaml_load
from models.Base import MachineLearningModel

class SupportVectorMachinesClassification(MachineLearningModel):
    def build(self):
        self.model = SVC(**yaml_load(self.config_path))

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X=self.preprocessing(x=X_train), 
                       y=np.ravel([i.astype(int) for i in self.preprocessing(x=y_train)], order='C'))

class SupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = SVR(**yaml_load(self.config_path))

class NuSupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = NuSVR(**yaml_load(self.config_path))

class LinearSupportVectorMachinesRegression(MachineLearningModel):
    def build(self):
        self.model = LinearSVR(**yaml_load(self.config_path))