import os
from pathlib import Path

import tensorflow as tf
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

class MovingAvg(tf.keras.layers.Layer):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=stride, padding='valid')

    def call(self, x):
        # padding on the both ends of time series
        front = tf.tile(x[:, 0:1, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        end = tf.tile(x[:, -1:, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        x = tf.concat([front, x, end], axis=1)
        x = self.avg(x)
        return x


class SeriesDecomp(tf.keras.layers.Layer):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(tf.keras.Model):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual

        # Decompsition Kernel Size
        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = []
            self.Linear_Trend = []
            for i in range(self.channels):
                self.Linear_Seasonal.append(tf.keras.layers.Dense(self.pred_len))
                self.Linear_Trend.append(tf.keras.layers.Dense(self.pred_len))
                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
                # self.Linear_Trend[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        else:
            self.Linear_Seasonal = tf.keras.layers.Dense(self.pred_len)
            self.Linear_Trend = tf.keras.layers.Dense(self.pred_len)
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
            # self.Linear_Trend.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        self.final_layer = Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)

        if self.individual:
            seasonal_output = tf.concat([tf.expand_dims(self.Linear_Seasonal[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            trend_output = tf.concat([tf.expand_dims(self.Linear_Trend[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            x = seasonal_output + trend_output
        else:
            seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            x = tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]

        # print(x.shape)
        return tf.squeeze(self.final_layer(x), axis=-1)


class NLinear(tf.keras.Model):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        
        # Use this line if you want to visualize the weights
        # self.Linear.weights = (1/self.seq_len)*tf.ones([self.seq_len, self.pred_len])
        if self.individual:
            self.Linear = [Dense(self.pred_len) for _ in range(self.channels)]
            # self.Linear = []
            # for i in range(self.channels):
            #     self.Linear.append(tf.keras.layers.Dense(self.pred_len))
        else:
            self.Linear = Dense(self.pred_len)
        self.final_layer = Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:]
        x = x - seq_last
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
        else:
            print(x.shape)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        x = x + seq_last
        # print(tf.squeeze(self.final_layer(x), axis=-1).shape)
        return tf.squeeze(self.final_layer(x), axis=-1)  # [Batch, Output length, Channel]

class Linear(tf.keras.Model):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(Linear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
            # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        if self.individual:
            self.Linear = []
            for i in range(self.channels):
                self.Linear.append(tf.keras.layers.Dense(units=self.pred_len))
        else:
            self.Linear = tf.keras.layers.Dense(units=self.pred_len)
        self.final_layer = tf.keras.layers.Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels)], axis=-1)
            # output = tf.zeros(shape=[x.shape[0], self.pred_len, x.shape[2]], dtype=x.dtype)
            # for i in range(self.channels):
            #     output[:,:,i] = self.Linear[i](x[:,:,i])
            # x = output
        else:
            x = self.Linear(tf.transpose(x, perm=[0,2,1]))
            x = tf.transpose(x, perm=[0,2,1])
        return tf.squeeze(self.final_layer(x), axis=-1) # [Batch, Output length, Channel]


from models.Base import TensorflowModel
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

class LTSF_Linear_Base(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, activations, dropouts, individual, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, activations, dropouts, normalize_layer=normalize_layer, seed=seed)
        self.individual = individual

    def save(self, file_name:str, save_dir:str='.'):
        os.makedirs(name=save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name, "ckpt")
        self.model.save_weights(Path(file_path).absolute())
        return file_path

    def callbacks(self, patience, save_dir, min_delta=0.001, extension=''):
        return super().callbacks(patience=patience, save_dir=save_dir, min_delta=min_delta, extension=extension)  

class LTSF_Linear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Linear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=self.input_shape[-1], individual=self.individual)
        # self.model = Linear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=7, individual=False)

class LTSF_NLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = NLinear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=self.input_shape[-1], individual=self.individual)
        # self.model = NLinear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=7, individual=False)

class LTSF_DLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = DLinear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=self.input_shape[-1], individual=self.individual)
        # self.model = DLinear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=7, individual=False)


    # def get_config(self):
    #     return super().get_config()