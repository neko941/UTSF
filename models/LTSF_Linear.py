import tensorflow as tf
from pathlib import Path
import os

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
        x = self.avg(tf.transpose(x, perm=[0, 2, 1]))
        x = tf.transpose(x, perm=[0, 2, 1])
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
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

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

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])

        if self.individual:
            seasonal_output = tf.zeros([tf.shape(seasonal_init)[0], tf.shape(seasonal_init)[1], self.pred_len], dtype=seasonal_init.dtype)
            trend_output = tf.zeros([tf.shape(trend_init)[0], tf.shape(trend_init)[1], self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]


class NLinear(tf.keras.Model):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weights = (1/self.seq_len)*tf.ones([self.seq_len, self.pred_len])
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = []
            for i in range(self.channels):
                self.Linear.append(tf.keras.layers.Dense(self.pred_len))
        else:
            self.Linear = tf.keras.layers.Dense(self.pred_len)
        self.final_layer = tf.keras.layers.Dense(self.pred_len)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:]
        x = x - seq_last
        if self.individual:
            output = tf.zeros([tf.shape(x)[0], self.pred_len, self.channels], dtype=x.dtype)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = tf.transpose(output, perm=[0, 2, 1])
        else:
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        x = x + seq_last
        return tf.squeeze(self.final_layer(x), axis=-1)  # [Batch, Output length, Channel]

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({'seq_len' : self.seq_len.numpy(),
    #             'pred_len' : self.pred_len.numpy(),
    #             'channels' : self.channels.numpy(),
    #             'individual' : self.individual.numpy()})
    #     return config

from models.Base import TensorflowModel
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
class LTSF_NLinear__Tensorflow(TensorflowModel):
    def build(self):
        self.model = NLinear(seq_len=self.input_shape, pred_len=self.output_shape, enc_in=7, individual=False)
        # self.model.summary()

    def save(self, file_name:str, save_dir:str='.'):
        os.makedirs(name=save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "ckpt")
        # pickle.dump(self.model, open(Path(file_path).absolute(), "wb"))
        # tf.saved_model.save(self.model, Path(file_path).absolute())
        self.model.save_weights(Path(file_path).absolute())
        return file_path

    def callbacks(self, patience, save_dir, min_delta=0.001, epochs=10_000_000):
        log_path = os.path.join(save_dir, 'logs')
        os.makedirs(name=log_path, exist_ok=True)

        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.1,
                                    patience=patience / 5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=min_delta * 10,
                                    cooldown=0,
                                    min_lr=0), 
                CSVLogger(filename=os.path.join(log_path, f'{self.model.name}.csv'), separator=',', append=False)]  

    # def get_config(self):
    #     return super().get_config()