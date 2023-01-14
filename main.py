import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.random import set_seed 
from tensorflow.data import AUTOTUNE

# from keras.utils import plot_model

from keras.layers import Normalization

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.losses import MeanSquaredError

from utils.general import yaml_save
from utils.general import increment_path

# performance metrics
from utils.metrics import MAE
from utils.metrics import MSE
from utils.metrics import RMSE
from utils.metrics import MAPE

from utils.dataset import slicing_window

# display results
from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# deep learning models
from models.RNN import BiRNN__Tensorflow
from models.LSTM import BiLSTM__Tensorflow
from models.GRU import BiGRU__Tensorflow
from models.customized import RNNcLSTM__Tensorflow

# machine learning models
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# from transformers import TFBertForSequenceClassification, TimeSeriesTransformerConfig
# def BERT(input_shape, output_size, normalize_layer=None, seed=941):
#     return TFBertForSequenceClassification(TimeSeriesTransformerConfig(input_shape))

def test(model, X_test, y_test, weight=None):
    # model = model(input_shape=input_shape, output_size=labelsz, normalize_layer=normalize_layer, RANDOM_SEED=seed)
    # model.load_weights(os.path.join(save_dir, 'weights', f"{model.name}_best.h5"))
    # print(weight)
    if weight is not None: model.load_weights(weight)
    # model.load_weights(r'D:\01.Code\00.Github\UTSF\runs\exp1\weights\combined_RNN_LSTM_best.h5')
    y_test_pred = model.predict(X_test)

    print()
    rmse = RMSE(y_test, y_test_pred)
    print(f'RMSE: {rmse}')
    mape = MAPE(y_test, y_test_pred)
    print(f'MAPE: {mape}')
    mse = MSE(y_test, y_test_pred)
    print(f'MSE: {mse}')
    mae = MAE(y_test, y_test_pred)
    print(f'MAE: {mae}')

    return [str(rmse), str(mape), str(mse), str(mae)]

optimizer_dict = {
    'SGD': SGD,
    'Adam' : Adam
}

def train(model, train_ds, val_ds, patience, save_dir, lr, optimizer, min_delta=0.001, epochs=10_000_000):
    model.compile(loss=MeanSquaredError(), 
                  optimizer=optimizer_dict[optimizer](learning_rate=lr))

    weight_path = os.path.join(save_dir, 'weights')
    os.makedirs(name=weight_path, exist_ok=True)
    log_path = os.path.join(save_dir, 'logs')
    os.makedirs(name=log_path, exist_ok=True)
    # model_path = os.path.join(save_dir, 'models')
    # os.makedirs(name=model_path, exist_ok=True)
    # plot_model(model, to_file=os.path.join(model_path, f'{model.name}.png'), show_shapes=False)
    history = model.fit(train_ds, 
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                               ModelCheckpoint(filepath=os.path.join(weight_path, f"{model.name}_best.h5"),
                                               save_best_only=True,
                                               save_weights_only=False,
                                               verbose=0), 
                               ModelCheckpoint(filepath=os.path.join(weight_path, f"{model.name}_last.h5"),
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
                               CSVLogger(filename=os.path.join(log_path, f'{model.name}.csv'), separator=',', append=False)])

    return history, model

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--batchsz', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--inputsz', type=int, default=30, help='')
    parser.add_argument('--labelsz', type=int, default=1, help='')
    parser.add_argument('--offset', type=int, default=1, help='')
    parser.add_argument('--trainsz', type=float, default=0.7, help='')
    parser.add_argument('--valsz', type=float, default=0.2, help='')

    parser.add_argument('--source', default='data.csv', help='dataset')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='optimizer')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')

    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--LinearRegression', action='store_true', help='')
    parser.add_argument('--XGBoost', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--BiRNN__Tensorflow', action='store_true', help='')
    parser.add_argument('--BiLSTM__Tensorflow', action='store_true', help='')
    parser.add_argument('--BiGRU__Tensorflow', action='store_true', help='')
    parser.add_argument('--RNNcLSTM__Tensorflow', action='store_true', help='Model that combineced RNN and LSTM')
    parser.add_argument('--all', action='store_true', help='Use all available models')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))
    yaml_save(os.path.join(save_dir, 'opt.yaml'), vars(opt))

    models_machine_learning = []
    models_tensorflow = []
    if opt.all:
        opt.LinearRegression = True
        opt.XGBoost = True
        opt.BiRNN__Tensorflow = True
        opt.BiLSTM__Tensorflow = True
        opt.BiGRU__Tensorflow = True
        opt.RNNcLSTM__Tensorflow = True
    if not opt.all and opt.MachineLearning:
        opt.LinearRegression = True
        opt.XGBoost = True
    elif not opt.all and opt.DeepLearning:
        opt.BiRNN__Tensorflow = True
        opt.BiLSTM__Tensorflow = True
        opt.BiGRU__Tensorflow = True
        opt.RNNcLSTM__Tensorflow = True
    if opt.XGBoost: models_machine_learning.append(XGBRegressor)    
    if opt.LinearRegression: models_machine_learning.append(LinearRegression)
    if opt.BiRNN__Tensorflow: models_tensorflow.append(BiRNN__Tensorflow)
    if opt.BiLSTM__Tensorflow: models_tensorflow.append(BiLSTM__Tensorflow)
    if opt.BiGRU__Tensorflow: models_tensorflow.append(BiGRU__Tensorflow)
    if opt.RNNcLSTM__Tensorflow: models_tensorflow.append(RNNcLSTM__Tensorflow)
    # models_tensorflow = [BERT]
    
    # set random seed
    set_seed(opt.seed)

    df = pd.read_csv(opt.source, index_col=0) # Đọc file .csv thành DataFrame
    dataset_length = len(df)

    TRAIN_END_IDX = int(opt.trainsz * dataset_length) 
    VAL_END_IDX = int(opt.valsz * dataset_length) + TRAIN_END_IDX
    
    TARGET_NAME = 'Adj Close'

    X_train, y_train = slicing_window(df, 
                                      df_start_idx=0,
                                      df_end_idx=TRAIN_END_IDX,
                                      input_size=opt.inputsz,
                                      label_size=opt.labelsz,
                                      offset=opt.offset,
                                      label_name=TARGET_NAME)

    X_val, y_val = slicing_window(df, 
                                  df_start_idx=TRAIN_END_IDX,
                                  df_end_idx=VAL_END_IDX,
                                  input_size=opt.inputsz,
                                  label_size=opt.labelsz,
                                  offset=opt.offset,
                                  label_name=TARGET_NAME)

    X_test, y_test = slicing_window(df, 
                                  df_start_idx=VAL_END_IDX,
                                  df_end_idx=None,
                                  input_size=opt.inputsz,
                                  label_size=opt.labelsz,
                                  offset=opt.offset,
                                  label_name=TARGET_NAME)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(opt.batchsz)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(opt.batchsz)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(opt.batchsz)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalize_layer = Normalization()
    normalize_layer.adapt(np.vstack((X_train, X_val, X_test)))
    INPUT_SHAPE = X_train.shape[-2:] 


    console = Console(record=True)
    table = Table(title="[cyan]Results", 
              show_header=True, 
              header_style="bold magenta",
              box=rbox.ROUNDED)

    for name in ['Name', 'RMSE', 'MAPE', 'MSE', 'MAE']:
        table.add_column(f'[green]{name}', justify='center')

    for model in models_machine_learning:
        model = model().fit([i.flatten() for i in X_train], [i.flatten() for i in y_train])
        model.predict([i.flatten() for i in X_test])
        errors = test(model=model, X_test=[i.flatten() for i in X_test], y_test=[i.flatten() for i in y_test])
        table.add_row(type(model).__name__, *errors)
        print()

    for model in models_tensorflow:
        model = model(input_shape=INPUT_SHAPE, output_size=opt.labelsz, normalize_layer=normalize_layer, seed=opt.seed)
        model.summary()
        history, model = train(model=model, train_ds=train_ds, val_ds=val_ds, patience=opt.patience, save_dir=save_dir, optimizer=opt.optimizer, lr=opt.lr, epochs=opt.epochs)
        errors = test(model=model, weight=os.path.join(save_dir, 'weights', f"{model.name}_best.h5"), X_test=X_test, y_test=y_test)
        table.add_row(model.name, *errors)
        print()

    console.print(table)
    console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
    # console.save_html(os.path.join(save_dir, 'results.html'), theme=MONOKAI)
    # console.save_text(os.path.join(save_dir, 'results.txt'))

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)