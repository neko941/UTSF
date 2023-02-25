import numpy as np
import tensorflow as tf

from models.Base import TensorflowModel
from models.Base import PytorchModel

from utils.metrics import metric_dict

from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.initializers import GlorotUniform

import torch.nn as nn

class VanillaLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # LSTM Layer 1
        self.model.add(LSTM(units=self.units[0],
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[0],
                            name='LSTM_layer'))
        # FC Layer
        self.model.add(Dense(units=self.units[1],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[1],
                             name='Fully_Connected_layer'))
        # Output Layer
        self.model.add(Dense(units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2],
                             name='Output_layer'))

class VanillaLSTM__Tensorflow(PytorchModel):
    def build(self):
        self.model = nn.Sequential(
            nn.LSTM(input_size=self.input_shape[1], hidden_size=self.units, batch_first=True),
            nn.Flatten(),
            nn.Linear(in_features=self.units, out_features=self.output_shape[1])
        )
        if self.normalize_layer:
            self.model = nn.Sequential(self.normalize_layer, self.model)

class BiLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # BiLSTM Layer 1 
        self.model.add(Bidirectional(LSTM(units=self.units[0], 
                                          return_sequences=True,
                                          kernel_initializer=GlorotUniform(seed=self.seed),
                                          activation=self.activations[0]),
                                     name='BiLSTM_layer_1'))
        # BiLSTM Layer 2 
        self.model.add(Bidirectional(LSTM(units=self.units[1], 
                                          return_sequences=True,
                                          kernel_initializer=GlorotUniform(seed=self.seed),
                                          activation=self.activations[1]),
                                     name='BiLSTM_layer_2'))
        # BiLSTM Layer 3 
        self.model.add(Bidirectional(LSTM(units=self.units[2], 
                                          return_sequences=False,
                                          kernel_initializer=GlorotUniform(seed=self.seed),
                                          activation=self.activations[2]),
                                     name='BiLSTM_layer_3'))
        # FC Layer
        self.model.add(Dense(units=self.units[3],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[3],
                             name='Fully_Connected_layer'))

        # Output Layer
        self.model.add(Dense(units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[4],
                             name='Output_layer'))
        
class ConvLSTM__Tensorflow(TensorflowModel):
    def body(self): 
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Conv3D(filters=self.output_shape, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
        
    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE', **kwargs):
        if self.model == None:
            X_train = np.array(X_train)[:,:, np.newaxis, np.newaxis]
            y_train = np.array(y_train)[:,:,np.newaxis, np.newaxis]
            X_val = np.array(X_val)[:,:,np.newaxis, np.newaxis]
            y_val = np.array(y_val)[:,:,np.newaxis, np.newaxis]
            self.input_shape = (None, *X_train.shape[-3:])
            self.build()

        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
                       validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
                       epochs=epochs, 
                       callbacks=self.callbacks(patience=patience, save_dir=save_dir, min_delta=0.001, epochs=epochs))

    def predict(self, X):
        return self.model.predict(np.array(X)[:,:, np.newaxis, np.newaxis])

    def score(self, y, yhat):
        y = np.array(y)[:,:, np.newaxis, np.newaxis]
        results = []
        for metric, func in metric_dict.items():
            result = func(y, yhat)
            results.append(str(result))
        return results