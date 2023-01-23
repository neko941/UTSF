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
from keras.layers import Normalization

# optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam

# callback functions
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

# loss function
from keras.losses import MeanSquaredError

# general utils
from utils.general import yaml_save
from utils.general import yaml_load
from utils.general import increment_path
from tensorflow.random import set_seed 
from tensorflow.data import AUTOTUNE

# performance metrics
from utils.metrics import MAE
from utils.metrics import MSE
from utils.metrics import RMSE
from utils.metrics import MAPE
from utils.metrics import R2

# dataset slicing 
from utils.dataset import slicing_window

# display results
from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# machine learning models
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from sklearn.impute import KNNImputer

# deep learning models
from models.RNN import VanillaRNN__Tensorflow    
from models.RNN import BiRNN__Tensorflow
from models.LSTM import VanillaLSTM__Tensorflow    
from models.LSTM import BiLSTM__Tensorflow    
from models.GRU import VanillaGRU__Tensorflow
from models.GRU import BiGRU__Tensorflow
from models.customized import RNNcLSTM__Tensorflow
from models.customized import GRUcLSTM__Tensorflow
from models.EncoderDecoder import EncoderDecoder__Tensorflow
from models.EncoderDecoder import BiEncoderDecoder__Tensorflow
from models.EncoderDecoder import CNNcLSTMcEncoderDecoder__Tensorflow
""" 
TODO:
    from models.NBeats import NBeats
    from models.LSTNet import LSTNet__Pytorch
    from models.Averaging import StackingAveragedModels
    from models.Averaging import AveragingModels
    from models.LSTM import ConvLSTM__Tensorflow    
"""

optimizer_dict = {
    'SGD': SGD,
    'Adam' : Adam
}

metric_dict = {
    'MAE' : MAE, 
    'MSE' : MSE,
    'RMSE' : RMSE, 
    'MAPE' : MAPE, 
    'R2' : R2
}

model_dict = [
    {
        'name' : 'LinearRegression', 
        'model' : LinearRegression,
        'help' : ''
    },{
        'name' : 'ElasticNet', 
        'model' : ElasticNet,
        'help' : ''
    },{
        'name' : 'SGDRegressor', 
        'model' : SGDRegressor,
        'help' : ''
    },{
        'name' : 'Lasso', 
        'model' : Lasso,
        'help' : ''
    },{
        'name' : 'LassoCV', 
        'model' : LassoCV,
        'help' : 'Lasso linear model with iterative fitting along a regularization path'
    },{
    #     'name' : 'LassoRobustScaler', 
    #     'model' : lambda: make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
    #     'help' : ''
    # },{
        'name' : 'Ridge', 
        'model' : Ridge,
        'help' : ''
    },{
        'name' : 'RidgeCV', 
        'model' : RidgeCV,
        'help' : 'Ridge regression with built-in cross-validation'
    },{
        'name' : 'KernelRidge', 
        'model' : KernelRidge,
        'help' : ''
    },{
        'name' : 'Lars', 
        'model' : Lars,
        'help' : ''
    },{
        'name' : 'LarsCV', 
        'model' : LarsCV,
        'help' : ''
    },{
        'name' : 'OrthogonalMatchingPursuit', 
        'model' : OrthogonalMatchingPursuit,
        'help' : ''
    },{
        'name' : 'OrthogonalMatchingPursuitCV', 
        'model' : OrthogonalMatchingPursuitCV,
        'help' : ''
    },{
        'name' : 'XGBoost', 
        'model' : XGBRegressor,
        'help' : ''
    },{
        'name' : 'LightGBM', 
        'model' : LGBMRegressor,
        'help' : ''
    },{
        'name' : 'CatBoost', 
        'model' : CatBoostRegressor,
        'help' : ''
    },{
        'name' : 'RandomForest', 
        'model' : RandomForestRegressor,
        'help' : ''
    },{
        'name' : 'GradientBoosting', 
        'model' : GradientBoostingRegressor,
        'help' : ''
    },{
    #     'name' : 'StackingAveragedModels', 
    #     'model' : StackingAveragedModels,
    #     'help' : ''
    # },{
    #     'name' : 'AveragingModels', 
    #     'model' : AveragingModels,
    #     'help' : ''
    # },{
        'name' : 'DecisionTree', 
        'model' : DecisionTreeClassifier,
        'help' : ''
    },{
    #     'name' : 'KNNImputer', 
    #     'model' : KNNImputer,
    #     'help' : ''
    # },{
        'name' : 'VanillaRNN__Tensorflow', 
        'model' : VanillaRNN__Tensorflow,
        'help' : ''
    },{
        'name' : 'BiRNN__Tensorflow', 
        'model' : BiRNN__Tensorflow,
        'help' : ''
    },{
        'name' : 'VanillaLSTM__Tensorflow', 
        'model' : VanillaLSTM__Tensorflow,
        'help' : ''
    },{
        'name' : 'BiLSTM__Tensorflow', 
        'model' : BiLSTM__Tensorflow,
        'help' : ''
    },{
    #     'name' : 'ConvLSTM__Tensorflow', 
    #     'model' : ConvLSTM__Tensorflow,
    #     'help' : ''
    # },{
        'name' : 'VanillaGRU__Tensorflow', 
        'model' : VanillaGRU__Tensorflow,
        'help' : ''
    },{
        'name' : 'BiGRU__Tensorflow', 
        'model' : BiGRU__Tensorflow,
        'help' : ''
    },{
        'name' : 'EncoderDecoder__Tensorflow', 
        'model' : EncoderDecoder__Tensorflow,
        'help' : ''
    },{
        'name' : 'BiEncoderDecoder__Tensorflow', 
        'model' : BiEncoderDecoder__Tensorflow,
        'help' : ''
    },{
        'name' : 'CNNcLSTMcEncoderDecoder__Tensorflow', 
        'model' : CNNcLSTMcEncoderDecoder__Tensorflow,
        'help' : ''
    },{
        'name' : 'RNNcLSTM__Tensorflow', 
        'model' : RNNcLSTM__Tensorflow,
        'help' : ''
    },{
        'name' : 'GRUcLSTM__Tensorflow', 
        'model' : GRUcLSTM__Tensorflow,
        'help' : ''
    # },{
    #     'name' : 'NBeats', 
    #     'model' : NBeats,
    #     'help' : ''
    },
]

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

    parser.add_argument('--source', default='data.yaml', help='dataset')
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
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--Tensorflow', action='store_true', help='')
    parser.add_argument('--Pytorch', action='store_true', help='')

    for item in model_dict:
        parser.add_argument(f"--{item['name']}", action='store_true', help=f"{item['help']}")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def test(model, X, y, weight=None):
    # model = model(input_shape=input_shape, output_size=labelsz, normalize_layer=normalize_layer, RANDOM_SEED=seed)
    # model.load_weights(os.path.join(save_dir, 'weights', f"{model.name}_best.h5"))
    # print(weight)
    if weight is not None: model.load_weights(weight)
    # model.load_weights(r'D:\01.Code\00.Github\UTSF\runs\exp1\weights\combined_RNN_LSTM_best.h5')
    yhat = model.predict(X)
    # try:
    #     yhat = model.predict(X)
    # except:
    #     yhat = model.fit_transform()
    if len(yhat.shape) > 2: 
        nsamples, nx, ny = yhat.shape
        yhat = yhat.reshape((nsamples,nx*ny))

    # print()
    # try:
    #     name = model.name
    # except:
    #     name = type(model).__name__
    # print(f'Model: {name}')
    results = []
    for metric, func in metric_dict.items():
        result = func(y, yhat)
        results.append(str(result))
        # print(f'{metric}: {result}')
    return results

# import keras.backend as K
# def weighted_mse(yTrue,yPred):
#     ones = K.ones_like(yTrue[0,:]) #a simple vector with ones shaped as (60,)
#     idx = K.cumsum(ones) #similar to a 'range(1,61)'
#     return K.mean((1/idx)*K.square(yTrue-yPred))


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
    if opt.DeepLearning:
        opt.Tensorflow = True
        opt.Pytorch = True
    for item in model_dict:
        if any([opt.Tensorflow and 'Tensorflow' in item["name"],
                opt.Pytorch and 'Tensorflow' in item["name"],
                opt.MachineLearning and 'Tensorflow' not in item["name"] and 'Pytorch' not in item["name"]]): 
            vars(opt)[f'{item["name"]}'] = True


    modelsMachineLearning = []
    modelsTensorflow = []
    modelsPytorch = []
    for item in model_dict:
        if not vars(opt)[f'{item["name"]}']: continue
        if 'Tensorflow' in item["name"]: modelsTensorflow.append(item['model'])
        elif 'Pytorch' in item["name"]: modelsPytorch.append(item['model'])
        else: modelsMachineLearning.append(item['model'])

    # set random seed
    set_seed(opt.seed)

    # read data.yaml
    data = yaml_load(opt.source)
    if data['features'] is None: 
        opt.CyclicalPattern = True
        data['features'] = []

    # read data
    csvs = []
    extensions = ('.csv')
    if not isinstance(data['data'], list): data['data'] = [data['data']]
    for i in data['data']: 
        if os.path.isdir(i):
            for root, dirs, files in os.walk(i):
                for file in files:
                    if file.endswith(extensions): csvs.append(os.path.join(root, file))
        if i.endswith(extensions) and os.path.exists(i): csvs.append(i)
    assert len(csvs) > 0, 'No csv file(s)'
    df = pd.read_csv(csvs[0])
    for i in csvs[1:]: df = pd.concat([df, pd.read_csv(i)])
    df.reset_index(drop=True, inplace=True)

    # get used cols
    cols = []
    for i in [data['date'], data['features'], data['target']]: 
        if isinstance(i, list): cols.extend(i)
        else: cols.append(i) 
    df = df[cols]

    # convert date to type datetime
    df[data['date']] = pd.to_datetime(df[data['date']])

    # auto fill missing data
    if opt.AutoInterpolate != '':
        df = pd.merge(df,
                 pd.DataFrame(pd.date_range(min(df[data['date']]), max(df[data['date']])), columns=[data['date']]),
                 how='right',
                 left_on=[data['date']],
                 right_on = [data['date']])
        df.fillna(method=f'{list(opt.AutoInterpolate)[0].lower()}fill', inplace=True)

    # sort data by date
    df.sort_values(data['date'], inplace=True)

    # add month sin, month cos (cyclical pattern)
    if opt.CyclicalPattern:
        # Extracting the hour of day
        # d["hour"] = [x.hour for x in d["dt"]]
        # # Creating the cyclical daily feature 
        # d["day_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in d["hour"]]
        # d["day_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in d["hour"]]

        d = [x.timestamp() for x in df[f"{data['date']}"]]
        s = 24 * 60 * 60 # Seconds in day  
        year = (365.25) * s # Seconds in year 
        df.insert(loc=0, column='month_cos', value=[np.cos((x) * (2 * np.pi / year)) for x in d])
        df.insert(loc=0, column='month_sin', value=[np.sin((x) * (2 * np.pi / year)) for x in d]) 

    # remove date col
    df.drop([data['date']], axis=1, inplace=True)

    # get dataset length
    dataset_length = len(df)

    # get train, val indices
    TRAIN_END_IDX = int(opt.trainsz * dataset_length) 
    VAL_END_IDX = int(opt.valsz * dataset_length) + TRAIN_END_IDX

    X_train, y_train = slicing_window(df, 
                                      df_start_idx=0,
                                      df_end_idx=TRAIN_END_IDX,
                                      input_size=opt.inputsz,
                                      label_size=opt.labelsz,
                                      offset=opt.offset,
                                      label_name=data['target'])

    X_val, y_val = slicing_window(df, 
                                  df_start_idx=TRAIN_END_IDX,
                                  df_end_idx=VAL_END_IDX,
                                  input_size=opt.inputsz,
                                  label_size=opt.labelsz,
                                  offset=opt.offset,
                                  label_name=data['target'])

    X_test, y_test = slicing_window(df, 
                                  df_start_idx=VAL_END_IDX,
                                  df_end_idx=None,
                                  input_size=opt.inputsz,
                                  label_size=opt.labelsz,
                                  offset=opt.offset,
                                  label_name=data['target'])

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
    # table header
    for name in ['Name', *list(metric_dict.keys())]: table.add_column(f'[green]{name}', justify='center')

    errors = []
    for model in modelsMachineLearning:
        try:
            try:
                model = model().fit([i.flatten() for i in X_train], [i.flatten() for i in y_train])
            except Exception as e:
                errors.append([model.__name__, str(e)])
                # for DecisionTreeClassifier
                model = model().fit([i.flatten() for i in X_train], [i.flatten().astype(int) for i in y_train])
            # model.predict(np.ravel([i.flatten() for i in X_test]))
            # model.predict([i.flatten() for i in X_test])
            errors = test(model=model, X=[i.flatten() for i in X_test], y=[i.flatten() for i in y_test])
            table.add_row(type(model).__name__, *errors)
        except Exception as e:
            errors.append([model.__name__, str(e)])
            # table.add_row(type(model).__name__, *['_' for _ in range(len(metric_dict.keys()))])
            table.add_row(model.__name__, *list('_' * len(metric_dict.keys())))
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)

    for model in modelsTensorflow:
        try:
            model = model(input_shape=INPUT_SHAPE, output_size=opt.labelsz, normalize_layer=normalize_layer, seed=opt.seed)
            model.summary()
            history, model = train_tensorflow(model=model, train_ds=train_ds, val_ds=val_ds, patience=opt.patience, save_dir=save_dir, optimizer=opt.optimizer, lr=opt.lr, epochs=opt.epochs)
            errors = test(model=model, weight=os.path.join(save_dir, 'weights', f"{model.name}_best.h5"), X=X_test, y=y_test)
            table.add_row(model.name, *errors)
        except Exception as e:
            errors.append([model.name, str(e)])
            table.add_row(model.name, *list('_' * len(metric_dict.keys())))
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
    for error in errors: print(f'{error[0]}\n{error[1]}', end='\n\n\n====================================================================\n\n\n')
    # console.print(table)
    # console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
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