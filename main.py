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
from sklearn.metrics import r2_score

from utils.dataset import slicing_window

# display results
from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# machine learning models
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

# deep learning models
from models.RNN import VanillaRNN__Tensorflow    
from models.RNN import BiRNN__Tensorflow
from models.LSTM import VanillaLSTM__Tensorflow    
from models.LSTM import BiLSTM__Tensorflow    
from models.GRU import VanillaGRU__Tensorflow
from models.GRU import BiGRU__Tensorflow
from models.customized import RNNcLSTM__Tensorflow
from models.NBeats import NBeats

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--batchsz', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--inputsz', type=int, default=30, help='')
    parser.add_argument('--labelsz', type=int, default=1, help='')
    parser.add_argument('--offset', type=int, default=1, help='')
    parser.add_argument('--trainsz', type=float, default=0.7, help='')
    parser.add_argument('--valsz', type=float, default=0.2, help='')

    parser.add_argument('--source', default=r'.\data\stocks\TSLA-Tesla.csv', help='dataset')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='optimizer')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')

    parser.add_argument('--AutoInterpolate', type=str, choices=['', 'forward', 'backward'], default='', help='')
    parser.add_argument('--CyclicalPattern', action='store_true', help='Add sin cos cyclical feature')

    parser.add_argument('--all', action='store_true', help='Use all available models')
    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--LinearRegression', action='store_true', help='')
    parser.add_argument('--SGDRegressor', action='store_true', help='')
    parser.add_argument('--XGBoost', action='store_true', help='')
    parser.add_argument('--Lasso', action='store_true', help='')
    parser.add_argument('--LassoCV', action='store_true', help='Lasso linear model with iterative fitting along a regularization path')
    parser.add_argument('--Ridge', action='store_true', help='')
    parser.add_argument('--RidgeCV', action='store_true', help='Ridge regression with built-in cross-validation')
    parser.add_argument('--Lars', action='store_true', help='Least Angle Regression')
    parser.add_argument('--LarsCV', action='store_true', help='Cross-validated Least Angle Regression')
    parser.add_argument('--OrthogonalMatchingPursuit', action='store_true', help='')
    parser.add_argument('--OrthogonalMatchingPursuitCV', action='store_true', help='')
    parser.add_argument('--RandomForest', action='store_true', help='')
    parser.add_argument('--DecisionTree', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--VanillaRNN__Tensorflow', action='store_true', help='')
    parser.add_argument('--BiRNN__Tensorflow', action='store_true', help='')
    parser.add_argument('--VanillaLSTM__Tensorflow', action='store_true', help='')
    parser.add_argument('--BiLSTM__Tensorflow', action='store_true', help='')
    parser.add_argument('--VanillaGRU__Tensorflow', action='store_true', help='')
    parser.add_argument('--BiGRU__Tensorflow', action='store_true', help='')
    parser.add_argument('--RNNcLSTM__Tensorflow', action='store_true', help='Model that combineced RNN and LSTM')
    parser.add_argument('--NBeats', action='store_true', help='')

    return parser.parse_known_args()[0] if known else parser.parse_args()

optimizer_dict = {
    'SGD': SGD,
    'Adam' : Adam
}

metric_dict = {
    'MAE' : MAE, 
    'MSE' : MSE,
    'RMSE' : RMSE, 
    'MAPE' : MAPE, 
    'R2' : r2_score
}

def test(model, X, y, weight=None):
    # model = model(input_shape=input_shape, output_size=labelsz, normalize_layer=normalize_layer, RANDOM_SEED=seed)
    # model.load_weights(os.path.join(save_dir, 'weights', f"{model.name}_best.h5"))
    # print(weight)
    if weight is not None: model.load_weights(weight)
    # model.load_weights(r'D:\01.Code\00.Github\UTSF\runs\exp1\weights\combined_RNN_LSTM_best.h5')
    yhat = model.predict(X)

    print()
    try:
        name = model.name
    except:
        name = type(model).__name__
    print(f'Model: {name}')
    results = []
    for metric, func in metric_dict.items():
        result = func(y, yhat)
        results.append(str(result))
        print(f'{metric}: {result}')
    return results

def train_tensorflow(model, train_ds, val_ds, patience, save_dir, lr, optimizer, min_delta=0.001, epochs=10_000_000):
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

def main(opt):
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))
    yaml_save(os.path.join(save_dir, 'opt.yaml'), vars(opt))

    # update option
    if opt.all:
        opt.MachineLearning = True
        opt.DeepLearning = True
    if opt.MachineLearning:
        opt.LinearRegression = True
        opt.SGDRegressor = True
        opt.XGBoost = True
        opt.RandomForest = True
        opt.DecisionTree = True
        opt.Lasso = True
        opt.LassoCV = True
        opt.Ridge = True
        opt.RidgeCV = True
        opt.Lars = True
        opt.LarsCV = True
        opt.OrthogonalMatchingPursuit = True
        opt.OrthogonalMatchingPursuitCV = True
    if opt.DeepLearning:
        opt.VanillaRNN__Tensorflow = True
        opt.BiRNN__Tensorflow = True
        opt.VanillaLSTM__Tensorflow = True
        opt.BiLSTM__Tensorflow = True
        opt.VanillaGRU__Tensorflow = True
        opt.BiGRU__Tensorflow = True
        opt.RNNcLSTM__Tensorflow = True
        opt.NBeats = True
    
    # collecting later used models
    models_machine_learning = []
    if opt.XGBoost: models_machine_learning.append(XGBRegressor)    
    if opt.LinearRegression: models_machine_learning.append(LinearRegression)
    if opt.SGDRegressor: models_machine_learning.append(SGDRegressor)
    if opt.Lasso: models_machine_learning.append(Lasso)
    if opt.LassoCV: models_machine_learning.append(LassoCV)
    if opt.Ridge: models_machine_learning.append(Ridge)
    if opt.RidgeCV: models_machine_learning.append(RidgeCV)
    if opt.Lars: models_machine_learning.append(Lars)
    if opt.LarsCV: models_machine_learning.append(LarsCV)
    if opt.OrthogonalMatchingPursuit: models_machine_learning.append(OrthogonalMatchingPursuit)
    if opt.OrthogonalMatchingPursuitCV: models_machine_learning.append(OrthogonalMatchingPursuitCV)
    if opt.RandomForest: models_machine_learning.append(RandomForestRegressor)
    if opt.DecisionTree: models_machine_learning.append(DecisionTreeClassifier)
    models_tensorflow = []
    if opt.VanillaRNN__Tensorflow: models_tensorflow.append(VanillaRNN__Tensorflow) 
    if opt.BiRNN__Tensorflow: models_tensorflow.append(BiRNN__Tensorflow)
    if opt.VanillaLSTM__Tensorflow: models_tensorflow.append(VanillaLSTM__Tensorflow)
    if opt.BiLSTM__Tensorflow: models_tensorflow.append(BiLSTM__Tensorflow)
    if opt.VanillaGRU__Tensorflow: models_tensorflow.append(VanillaGRU__Tensorflow)
    if opt.BiGRU__Tensorflow: models_tensorflow.append(BiGRU__Tensorflow)
    if opt.RNNcLSTM__Tensorflow: models_tensorflow.append(RNNcLSTM__Tensorflow)
    # TODO
    # if opt.NBeats: models_tensorflow.append(NBeats)
    
    # set random seed
    set_seed(opt.seed)

    # TODO: change these into options
    TARGET_NAME = 'Adj Close'
    DATE_VARIABLE = 'Date'
    
    # read data
    df = pd.read_csv(opt.source)
    df[DATE_VARIABLE] = pd.to_datetime(df[DATE_VARIABLE])
    if opt.AutoInterpolate != '':
        df = pd.merge(df,
                 pd.DataFrame(pd.date_range(min(df[DATE_VARIABLE]), max(df[DATE_VARIABLE])), columns=[DATE_VARIABLE]),
                 how='right',
                 left_on=[DATE_VARIABLE],
                 right_on = [DATE_VARIABLE])
        df.fillna(method=f'{list(opt.AutoInterpolate)[0].lower()}fill', inplace=True)

    #  
    df.sort_values(DATE_VARIABLE, inplace=True)
    d = [x.timestamp() for x in df[f'{DATE_VARIABLE}']]
    df.drop([DATE_VARIABLE], axis=1, inplace=True)
    if opt.CyclicalPattern:
        s = 24 * 60 * 60 # Seconds in day  
        year = (365.25) * s # Seconds in year 
        df.insert(loc=0, column='month_cos', value=[np.cos((x) * (2 * np.pi / year)) for x in d])
        df.insert(loc=0, column='month_sin', value=[np.sin((x) * (2 * np.pi / year)) for x in d]) 

    # get dataset length
    dataset_length = len(df)

    TRAIN_END_IDX = int(opt.trainsz * dataset_length) 
    VAL_END_IDX = int(opt.valsz * dataset_length) + TRAIN_END_IDX

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

    for name in ['Name', *list(metric_dict.keys())]:
        table.add_column(f'[green]{name}', justify='center')

    for model in models_machine_learning:
        try:
            model = model().fit([i.flatten() for i in X_train], [i.flatten() for i in y_train])
        except:
            # for DecisionTreeClassifier
            model = model().fit([i.flatten() for i in X_train], [i.flatten().astype(int) for i in y_train])
        # model.predict(np.ravel([i.flatten() for i in X_test]))
        model.predict([i.flatten() for i in X_test])
        errors = test(model=model, X=[i.flatten() for i in X_test], y=[i.flatten() for i in y_test])
        table.add_row(type(model).__name__, *errors)
        print()

    for model in models_tensorflow:
        model = model(input_shape=INPUT_SHAPE, output_size=opt.labelsz, normalize_layer=normalize_layer, seed=opt.seed)
        model.summary()
        history, model = train_tensorflow(model=model, train_ds=train_ds, val_ds=val_ds, patience=opt.patience, save_dir=save_dir, optimizer=opt.optimizer, lr=opt.lr, epochs=opt.epochs)
        errors = test(model=model, weight=os.path.join(save_dir, 'weights', f"{model.name}_best.h5"), X=X_test, y=y_test)
        table.add_row(model.name, *errors)
        print()

    console.print(table)
    console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
    # console.save_html(os.path.join(save_dir, 'results.html'), theme=MONOKAI)
    # console.save_text(os.path.join(save_dir, 'results.txt'))

def run(**kwargs):
    """ 
    Usage (example)
        import main
        main.run(all=True, source='/content/UTSF/data/stocks/TSLA-Tesla.csv')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)