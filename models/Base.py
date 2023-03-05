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
from keras.models import Sequential 
from keras.layers import Input

from utils.metrics import metric_dict

import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils.general import yaml_load
from pathlib import Path

class BaseModel:
    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

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

    def score(self, y, yhat, r):
        if len(yhat.shape) > 2: 
            nsamples, nx, ny = yhat.shape
            yhat = yhat.reshape((nsamples,nx*ny))
        if r != -1:
            results = [str(np.round(np.float64(metric_dict[key](y, yhat)), r)) for key in metric_dict.keys()]
        else:
            results = [str(metric_dict[key](y, yhat)) for key in metric_dict.keys()]
        return results

class MachineLearningModel(BaseModel):
    def __init__(self, config_path, **kwargs):
        self.config_path = config_path
        self.is_classifier = False
    
    def build(self):
        pass

    def preprocessing(self, x):
        return [i.flatten() for i in x]

    def fit(self, X_train, y_train, **kwargs):
        if self.is_classifier:
            y_train = np.ravel([i.astype(int) for i in self.preprocessing(x=y_train)], order='C') 
        else:
            y_train = np.ravel(self.preprocessing(x=y_train), order='C')
        self.model.fit(X=self.preprocessing(x=X_train), 
                       y=y_train)
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

class TensorflowModel(BaseModel):
    def __init__(self, input_shape, output_shape, units, activations, dropouts, normalize_layer=None, seed=941, **kwargs):
        self.function_dict = {
            'Adam' : Adam,
            'MSE' : MeanSquaredError,
            'SGD' : SGD
        }
        self.units = units
        self.seed = seed
        self.normalize_layer = normalize_layer
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.activations = activations
        self.dropouts = dropouts

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

    def build(self):
        try:
            self.model = Sequential(layers=None, name=self.__class__.__name__)
            # Input layer
            self.model.add(Input(shape=self.input_shape, name='Input_layer'))
            self.body()
            self.model.summary()
        except Exception as e:
            print(e)
            self.model = None

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE', **kwargs):
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
        if os.path.exists(weight): self.model.load_weights(weight)

import torch.optim as optim
import torch.nn as nn

class PytorchModel(BaseModel):
    def __init__(self, model):
        self.model = model
        self.function_dict = {
            'Adam' : optim.Adam,
            'MSE' : nn.MSELoss,
            'SGD' : optim.SGD
        }

    def preprocessing(self, x, y, batchsz):
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(x)
        y_train = torch.from_numpy(y)

        # Combine the features and labels into a single tensor
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Create the data loader
        train_dataloader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True, num_workers=0)

        return train_dataloader

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE'):
        # Preprocess data
        train_dataloader = self.preprocessing(X_train, y_train, batchsz)
        val_dataloader = self.preprocessing(X_val, y_val, batchsz)

        # Set optimizer and loss function
        self.optimizer = self.function_dict[optimizer](params=self.model.parameters(), lr=learning_rate)
        self.loss_fn = self.function_dict[loss]()

        # Train the model
        best_loss = float('inf')
        early_stop_count = 0
        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Train step
            self.model.train()
            for i, (inputs, targets) in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    val_loss += loss.item()

            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Save the best model
            if val_loss < best_loss:
                print('Saving model...')
                torch.save(self.model.state_dict(), save_dir)
                best_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print('Stopping early.')
                    break

    def save(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)

    def load(self, save_dir):
        self.model.load_state_dict(torch.load(save_dir))

    def predict(self, X_test):
        # Preprocess test data
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test))
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        # Make predictions
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_dataloader:
                outputs = self.model(inputs[0])
                predictions.append(outputs.numpy())
                
        return np.array(predictions).squeeze()
