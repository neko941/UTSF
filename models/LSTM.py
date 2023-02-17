import numpy as np
import tensorflow as tf

from models.Base import TensorflowModel

from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.initializers import GlorotUniform

class VanillaLSTM__Tensorflow(TensorflowModel):
    def build(self):
        self.model = Sequential(layers=None, name=self.__class__.__name__)
        # Input layer
        self.model.add(Input(shape=self.input_shape, name='Input_layer'))
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

class BiLSTM__Tensorflow(TensorflowModel):
    def build(self):
        self.model = Sequential(layers=None, name=self.__class__.__name__)
        # Input layer
        self.model.add(Input(shape=self.input_shape, name='Input_layer'))
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
        
# class ConvLSTM__Tensorflow(TensorflowModel):
#     def build(self, input_shape, output_shape, units):
#         self.model = tf.keras.Sequential(layers=None, 
#                                          name=self.__class__.__name__)
#         self.model.add(tf.keras.Input(shape=(None, *X_train.shape[-3:]),, 
#                                       name='Input_layer'))
#         # Normalization
#         if self.normalize_layer: self.model.add(self.normalize_layer)

#         self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_self.modeluences=True))
#         self.model.add(BatchNormalization())
#         self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_self.modeluences=True))
#         self.model.add(BatchNormalization())
#         self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_self.modeluences=True))
#         self.model.add(BatchNormalization())
#         self.model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_self.modeluences=True))
#         self.model.add(BatchNormalization())
#         self.model.add(Conv3D(filters=output_shape, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

#     def preprocessing(self, x, y, batchsz):
#         return tf.data.Dataset.from_tensor_slices((np.array(x)[:,:, np.newaxis, np.newaxis], np.array(y)[:,:, np.newaxis, np.newaxis])).batch(batchsz).cache().prefetch(buffer_size=AUTOTUNE)

#     def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, save_dir, batchsz, optimizer='Adam', loss='MSE', **kwargs):
#         # print(self.function_dict[optimizer](learning_rate=learning_rate), self.function_dict[loss]())
#         self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
#         self.model.fit(self.preprocessing(x=X_train, y=y_train, batchsz=batchsz), 
#                        validation_data=self.preprocessing(x=X_val, y=y_val, batchsz=batchsz),
#                        epochs=epochs, 
#                        callbacks=self.callbacks(patience=patience, save_dir=save_dir, min_delta=0.001, epochs=epochs))