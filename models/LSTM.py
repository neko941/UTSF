import numpy as np

from models.Base import TensorflowModel
from models.Base import PytorchModel

from utils.metrics import metric_dict


from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import Flatten
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
        self.model.add(LSTM(name='LSTM_layer',
                            units=self.units[0],
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[0]))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[1],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[1]))
        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[2]))

class VanillaLSTM__Pytorch(PytorchModel):
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
        self.model.add(Bidirectional(name='BiLSTM_layer_1',
                                     layer=LSTM(units=self.units[0],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[0])))
        # BiLSTM Layer 2 
        self.model.add(Bidirectional(name='BiLSTM_layer_2',
                                     layer=LSTM(units=self.units[1],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[1])))
        # BiLSTM Layer 3 
        self.model.add(Bidirectional(name='BiLSTM_layer_3',
                                     layer=LSTM(units=self.units[2],
                                                return_sequences=False,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[2])))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[3],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[3]))

        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[4]))
        
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
        # self.model.add(Flatten())
        # self.model.add(Dense(name='Output_layer',
        #                      units=self.output_shape, 
        #                      kernel_initializer=GlorotUniform(seed=self.seed)))
        
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

    def score(self, y, yhat, r):
        y = np.array(y)[:,:, np.newaxis, np.newaxis]
        if r != -1:
            results = [str(np.round(np.float64(metric_dict[key](y, yhat)), r)) for key in metric_dict.keys()]
        else:
            results = [str(metric_dict[key](y, yhat)) for key in metric_dict.keys()]
        return results
        return results