import tensorflow as tf
from models.Base import TensorflowModel

from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.initializers import GlorotUniform

class VanillaRNN__Tensorflow(TensorflowModel):
    def body(self):
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # RNN Layer 1
        self.model.add(SimpleRNN(name='RNN_layer',
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

class BiRNN__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)

        # BiRNN Layer 1 
        self.model.add(Bidirectional(name='BiRNN_layer_1',
                                     layer=SimpleRNN(units=self.units[0], 
                                                     return_sequences=True,
                                                     kernel_initializer=GlorotUniform(seed=self.seed), 
                                                     activation=self.activations[0])))
        # BiRNN Layer 2 
        self.model.add(Bidirectional(name='BiRNN_layer_2',
                                     layer=SimpleRNN(units=self.units[1], 
                                                     return_sequences=True,
                                                     kernel_initializer=GlorotUniform(seed=self.seed), 
                                                     activation=self.activations[1])))
        # BiRNN Layer 3 
        self.model.add(Bidirectional(name='BiRNN_layer_3', 
                                     layer=SimpleRNN(units=self.units[2], 
                                                     return_sequences=False,
                                                     kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed), 
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