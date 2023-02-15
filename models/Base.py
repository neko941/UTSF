import os
from abc import abstractmethod

import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from tensorflow.data import AUTOTUNE

import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils.general import yaml_load
from pathlib import Path

class BaseModel:
    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

class MachineLearningModel(BaseModel):
    def __init__(self):
        pass
    
    def preprocessing(self, x):
        return [i.flatten() for i in x]

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(self.preprocessing(x=X_train), np.ravel(self.preprocessing(x=y_train), order='C'))
        # self.model.fit(self.preprocessing(x=X_train), self.preprocessing(x=y_train))
    
    def save(self, file_name:str, save_dir:str='.', extension:str='.pkl'):
        os.makedirs(name=save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name+extension)
        pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        return file_path

    def load(self, weight):
        if not os.path.exists(weight): pass
        self.model = pickle.load(open(weight, "rb"))

    def predict(self, X):
        return self.model.predict(self.preprocessing(x=X))
        
# from keras import backend as K
# from keras.layers.core import Activation
# from keras.utils.generic_utils import get_custom_objects

# def SnakeActivation(x, a):
#     return x - K.cos(2*a*x)/(2*a) + 1/(2*a)

# get_custom_objects().update({'xsinsquared': Activation(lambda x: x + (K.sin(x)) ** 2),
#                              'xsin': Activation(lambda x: x + (K.sin(x))),
#                              'snake_a.5': Activation(lambda x: SnakeActivation(x=x, a=0.5)),
#                              'snake_a1': Activation(lambda x: SnakeActivation(x=x, a=1)),
#                              'snake_a5': Activation(lambda x: SnakeActivation(x=x, a=5)),
#                              })  
class TensorflowModel(BaseModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941):
        self.function_dict = {
            'Adam' : Adam,
            'MSE' : MeanSquaredError,
            'SGD' : SGD
        }
        self.units = units
        self.seed = seed
        self.normalize_layer = normalize_layer
        self.build(input_shape, output_shape, units)
        self.model.summary()

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    def callbacks(self, patience, save_dir, min_delta=0.001, epochs=10_000_000):
        weight_path = os.path.join(save_dir, 'weights')
        os.makedirs(name=weight_path, exist_ok=True)
        log_path = os.path.join(save_dir, 'logs')
        os.makedirs(name=log_path, exist_ok=True)

        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ModelCheckpoint(filepath=os.path.join(weight_path, f"{self.model.name}_best.h5"),
                                save_best_only=True,
                                save_weights_only=False,
                                verbose=0), 
                ModelCheckpoint(filepath=os.path.join(weight_path, f"{self.model.name}_last.h5"),
                                save_best_only=False,
                                save_weights_only=False,
                                verbose=0),
                ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.1,
                                    patience=patience / 5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=min_delta * 10,
                                    cooldown=0,
                                    min_lr=0), 
                CSVLogger(filename=os.path.join(log_path, f'{self.model.name}.csv'), separator=',', append=False)]  
        
    def preprocessing(self, x, y, batchsz):
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batchsz).cache().prefetch(buffer_size=AUTOTUNE)

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE', **kwargs):
        # print(self.function_dict[optimizer](learning_rate=learning_rate), self.function_dict[loss]())
        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
                       validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
                       epochs=epochs, 
                       callbacks=self.callbacks(patience=patience, save_dir=save_dir, min_delta=0.001, epochs=epochs))

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, file_name:str, save_dir:str='.', extension:str='.h5'):
        os.makedirs(name=save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name+extension)
        pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        return file_path
    
    def load(self, weight):
        self.model.load_weights(weight)

class PytorchModel(BaseModel):
    def fit(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def predict(self):
        pass

    def preprocessing(self, x, y, batchsz):
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(x)
        y_train = torch.from_numpy(y)

        # Combine the features and labels into a single tensor
        train_dataset = torch.utils.data.TensorDataset(x, y)

        # Create the data loader
        train_dataloader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True, num_workers=0)

        pass