import tensorflow as tf
from models.Base import TensorflowModel
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Model

"""
    https://github.com/xiaochus/TrafficFlowPrediction/blob/master/model/model.py
"""

class StackedAutoEncoders__TensorFlow(TensorflowModel):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.
    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        model: List(Model), List of SAE and SAEs.
    """
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        sae1 = self._get_sae(self.units[0], self.units[1], self.output_shape)
        sae2 = self._get_sae(self.units[1], self.units[2], self.output_shape)
        sae3 = self._get_sae(self.units[2], self.units[3], self.output_shape)

        self.model.add(Dense(self.units[1], name='hidden1'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(self.units[2], name='hidden2'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(self.units[3], name='hidden3'))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_shape, activation='sigmoid'))

        # self.model = tf.concat([sae1, sae2, sae3, self.model], axis=-1)
        self.model = Model([sae1, sae2, sae3, self.model])
        # self.model = [sae1, sae2, sae3, self.model]

    def _get_sae(self, inputs, hidden, output):
        """SAE(Auto-Encoders)
        Build SAE Model.
        # Arguments
            inputs: Integer, number of input units.
            hidden: Integer, number of hidden units.
            output: Integer, number of output units.
        # Returns
            model: Model, nn model.
        """

        model = Sequential()
        model.add(Dense(hidden, input_dim=inputs, name='hidden'))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(output, activation='sigmoid'))

        return model