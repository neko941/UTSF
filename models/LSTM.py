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
import torch

class VanillaLSTM__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # LSTM Layer 
        self.model.add(LSTM(name='LSTM_layer_0',
                            units=self.units[0],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[0]))
        self.model.add(LSTM(name='LSTM_layer_1',
                            units=self.units[1],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[1]))
        self.model.add(LSTM(name='LSTM_layer_2',
                            units=self.units[2],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[2]))
        self.model.add(LSTM(name='LSTM_layer_3',
                            units=self.units[3],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[3]))
        self.model.add(LSTM(name='LSTM_layer_4',
                            units=self.units[4],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[4]))
        self.model.add(LSTM(name='LSTM_layer_5',
                            units=self.units[5],
                            return_sequences=False,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[5]))
        # FC Layer
        self.model.add(Dense(name='Fully_Connected_layer',
                             units=self.units[6],
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[6]))
        # Output Layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape, 
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[7]))

class _VanillaLSTM__Pytorch(nn.Module):
    def __init__(self, input_shape, output_shape, units):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=units[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units[0], hidden_size=units[1], batch_first=True)
        self.lstm3 = nn.LSTM(input_size=units[1], hidden_size=units[2], batch_first=True)
        self.dense = nn.Linear(in_features=units[2], out_features=units[3])
        self.activation = nn.ReLU()
        self.out = nn.Linear(in_features=units[3], out_features=output_shape)
        
    def forward(self, input):
        output, _ = self.lstm1(input)
        output, _ = self.lstm2(output)
        output, _ = self.lstm3(output)
        output = self.dense(output)
        output = self.activation(output)
        output = self.out(output)
        return output 

class VanillaLSTM__Pytorch(PytorchModel):
    def build(self):
        self.model = _VanillaLSTM__Pytorch(input_shape=self.input_shape[-1],
                                           output_shape=self.output_shape,
                                           units=self.units)

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
        


# class _BiLSTM__Pytorch(nn.Module):
#     def __init__(self, input_shape, output_shape, units, **kwargs):
#         super().__init__()
#         print(input_shape)
#         self.lstm1 = nn.LSTM(input_size=input_shape[-1], hidden_size=128, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
#         self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
#         self.dense = nn.Linear(in_features=32, out_features=32)
#         self.activation = nn.ReLU()
#         self.out = nn.Linear(in_features=32, out_features=1)
        
#     def forward(self, input):
#         output, _ = self.lstm1(input)
#         output, _ = self.lstm2(output)
#         output, _ = self.lstm3(output)
#         output = self.dense(output)
#         output = self.activation(output)
#         output = self.out(output)
#         return output  

# import torch.nn as nn

# class _BiLSTM__Pytorch(nn.Module):
#     def __init__(self, input_shape, output_shape, units):
#         super(_BiLSTM__Pytorch, self).__init__()

#         # Normalization
#         # if self.normalize_layer:
#         #     self.normalize = nn.BatchNorm1d(input_dim)
        
#         # BiLSTM Layer 1 
#         self.bilstm1 = nn.LSTM(input_size=input_shape[0], 
#                                hidden_size=units[0], 
#                                bidirectional=True,
#                                batch_first=True)

#         # BiLSTM Layer 2 
#         self.bilstm2 = nn.LSTM(input_size=units[0] * 2, 
#                                hidden_size=units[1], 
#                                bidirectional=True,
#                                batch_first=True)

#         # BiLSTM Layer 3 
#         self.bilstm3 = nn.LSTM(input_size=units[1] * 2, 
#                                hidden_size=units[2], 
#                                bidirectional=True,
#                                batch_first=True)

#         self.activation = nn.ReLU()

#         # FC Layer
#         self.fc = nn.Linear(units[2] * 2, units[3])

#         # Output Layer
#         self.output = nn.Linear(units[3], output_shape)
        
#     def forward(self, x):
#         # # Normalization
#         # if self.normalize_layer:
#         #     x = self.normalize(x)

#         # BiLSTM Layer 1 
#         x, _ = self.bilstm1(x)

#         # BiLSTM Layer 2 
#         x, _ = self.bilstm2(x)

#         # BiLSTM Layer 3 
#         x, _ = self.bilstm3(x)

#         x = self.activation(x)

#         # FC Layer
#         x = self.fc(x[:, -1, :])

#         # Output Layer
#         x = self.output(x)
#         return x


# class BiLSTM__Pytorch(PytorchModel):
#     def build(self):
#         self.model = _BiLSTM__Pytorch(input_shape=self.input_shape, 
#                                       output_shape=self.output_shape,
#                                       units=self.units).double()


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