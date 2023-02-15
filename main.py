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
import torch
# import random
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
from utils.metrics import used_metric
from utils.metrics import calculate_score

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
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from sklearn.impute import KNNImputer

from models.SVM import SupportVectorMachines
from models.SVM import LinearSupportVectorMachines
from models.XGBoost import ExtremeGradientBoosting
# deep learning models
from models.RNN import VanillaRNN__Tensorflow    
from models.RNN import BiRNN__Tensorflow
from models.LSTM import VanillaLSTM__Tensorflow    
from models.LSTM import BiLSTM__Tensorflow    
from models.GRU import VanillaGRU__Tensorflow
from models.GRU import BiGRU__Tensorflow
# from models.customized import RNNcLSTM__Tensorflow
# from models.customized import GRUcLSTM__Tensorflow
from models.EncoderDecoder import EncoderDecoder__Tensorflow
from models.EncoderDecoder import BiEncoderDecoder__Tensorflow
from models.EncoderDecoder import CNNcLSTMcEncoderDecoder__Tensorflow
""" 
TODO:
    from models.TabTransformer import TabTransformer
    from models.NBeats import NBeats
    from models.LSTM import ConvLSTM__Tensorflow    
    from models.LSTNet import LSTNet__Pytorch
    from models.Averaging import StackingAveragedModels
    from models.Averaging import AveragingModels
"""

model_dict = [
    {
    #     'name' : 'LinearRegression', 
    #     'model' : LinearRegression,
    #     'help' : ''
    # },{
    #     'name' : 'ElasticNet', 
    #     'model' : ElasticNet,
    #     'help' : ''
    # },{
    #     'name' : 'SGDRegressor', 
    #     'model' : SGDRegressor,
    #     'help' : ''
    # },{
    #     'name' : 'Lasso', 
    #     'model' : Lasso,
    #     'help' : ''
    # },{
    #     'name' : 'LassoCV', 
    #     'model' : LassoCV,
    #     'help' : 'Lasso linear model with iterative fitting along a regularization path'
    # },{
    #     'name' : 'LassoRobustScaler', 
    #     'model' : lambda: make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
    #     'help' : ''
    # },{
    #     'name' : 'Ridge', 
    #     'model' : Ridge,
    #     'help' : ''
    # },{
    #     'name' : 'RidgeCV', 
    #     'model' : RidgeCV,
    #     'help' : 'Ridge regression with built-in cross-validation'
    # },{
    #     'name' : 'KernelRidge', 
    #     'model' : KernelRidge,
    #     'help' : ''
    # },{
    #     'name' : 'Lars', 
    #     'model' : Lars,
    #     'help' : ''
    # },{
    #     'name' : 'LarsCV', 
    #     'model' : LarsCV,
    #     'help' : ''
    # },{
    #     'name' : 'OrthogonalMatchingPursuit', 
    #     'model' : OrthogonalMatchingPursuit,
    #     'help' : ''
    # },{
    #     'name' : 'OrthogonalMatchingPursuitCV', 
    #     'model' : OrthogonalMatchingPursuitCV,
    #     'help' : ''
    # },{
    #     'name' : 'XGBoost', 
    #     'model' : ExtremeGradientBoosting,
    #     'help' : '',
    #     'config': 'configs/XGBoost.yaml'
    # },{
    #     'name' : 'SVM', 
    #     'model' : SupportVectorMachines,
    #     'help' : '',
    #     'config': 'configs/SVM.yaml'
    # },{
    #     'name' : 'LinearSVR', 
    #     'model' : LinearSupportVectorMachines,
    #     'help' : '',
    #     'config': 'configs/LinearSVR.yaml'
    # },{
    #     'name' : 'LightGBM', 
    #     'model' : LGBMRegressor,
    #     'help' : ''
    # },{
    #     'name' : 'CatBoost', 
    #     'model' : CatBoostRegressor,
    #     'help' : ''
    # },{
    #     'name' : 'RandomForest', 
    #     'model' : RandomForestRegressor,
    #     'help' : ''
    # },{
    #     'name' : 'GradientBoosting', 
    #     'model' : GradientBoostingRegressor,
    #     'help' : ''
    # },{
    #     'name' : 'StackingAveragedModels', 
    #     'model' : StackingAveragedModels,
    #     'help' : ''
    # },{
    #     'name' : 'AveragingModels', 
    #     'model' : AveragingModels,
    #     'help' : ''
    # },{
    #     'name' : 'DecisionTree', 
    #     'model' : DecisionTreeClassifier,
    #     'help' : ''
    # },{
    #     'name' : 'KNNImputer', 
    #     'model' : KNNImputer,
    #     'help' : ''
    # },{
        # 'name' : 'VanillaRNN__Tensorflow', 
        # 'model' : VanillaRNN__Tensorflow,
        # 'help' : ''
    # },{
    #     'name' : 'BiRNN__Tensorflow', 
    #     'model' : BiRNN__Tensorflow,
    #     'help' : ''
    # },{
        'model' : VanillaLSTM__Tensorflow,
        'help' : '',
        'units' : [128, 32],
        'type' : 'Tensorflow'
    },{ 
        'model' : BiLSTM__Tensorflow,
        'help' : '',
        'units' : [128, 64, 32, 32],
        'type' : 'Tensorflow'
    # },{
    #     'name' : 'ConvLSTM__Tensorflow', 
    #     'model' : ConvLSTM__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'VanillaGRU__Tensorflow', 
    #     'model' : VanillaGRU__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'BiGRU__Tensorflow', 
    #     'model' : BiGRU__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'EncoderDecoder__Tensorflow', 
    #     'model' : EncoderDecoder__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'BiEncoderDecoder__Tensorflow', 
    #     'model' : BiEncoderDecoder__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'CNNcLSTMcEncoderDecoder__Tensorflow', 
    #     'model' : CNNcLSTMcEncoderDecoder__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'RNNcLSTM__Tensorflow', 
    #     'model' : RNNcLSTM__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'GRUcLSTM__Tensorflow', 
    #     'model' : GRUcLSTM__Tensorflow,
    #     'help' : ''
    # },{
    #     'name' : 'NBeats', 
    #     'model' : NBeats,
    #     'help' : ''
    # },{
    #     'name' : 'TabTransformer', 
    #     'model' : lambda: TabTransformer(numerical_features = NUMERIC_FEATURES,  # List with names of numeric features
    #                                     categorical_features = CATEGORICAL_FEATURES, # List with names of categorical feature
    #                                     categorical_lookup=category_prep_layers,   # Dict with StringLookup layers 
    #                                     numerical_discretisers=None,  # None, we are simply passing the numeric features
    #                                     embedding_dim=32,  # Dimensionality of embeddings
    #                                     out_dim=1,  # Dimensionality of output (binary task)
    #                                     out_activation='sigmoid',  # Activation of output layer
    #                                     depth=4,  # Number of Transformer Block layers
    #                                     heads=8,  # Number of attention heads in the Transformer Blocks
    #                                     attn_dropout=0.1,  # Dropout rate in Transformer Blocks
    #                                     ff_dropout=0.1,  # Dropout rate in the final MLP
    #                                     mlp_hidden_factors=[2, 4],  # Factors by which we divide final embeddings for each layer
    #                                     use_column_embedding=True,  # If we want to use column embeddings
    #                                 ),
    #     'help' : ''
    },
]

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='')
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
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')

    parser.add_argument('--AutoInterpolate', type=str, choices=['', 'forward', 'backward'], default='', help='')
    parser.add_argument('--CyclicalPattern', action='store_true', help='Add sin cos cyclical feature')
    parser.add_argument('--Normalization', action='store_true', help='')

    parser.add_argument('--all', action='store_true', help='Use all available models')
    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--Tensorflow', action='store_true', help='')
    parser.add_argument('--Pytorch', action='store_true', help='')

    for item in model_dict:
        parser.add_argument(f"--{item['model'].__name__}", action='store_true', help=f"{item['help']}")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))
    
    """ Save init options """
    yaml_save(os.path.join(save_dir, 'opt.yaml'), vars(opt))

    """ Update options and save """
    if opt.all:
        opt.MachineLearning = True
        opt.DeepLearning = True
    if opt.DeepLearning:
        opt.Tensorflow = True
        opt.Pytorch = True
    for item in model_dict:
        if any([opt.Tensorflow and item['type']=='Tensorflow',
                opt.Pytorch and item['type']=='Pytorch',
                opt.MachineLearning and item['type']=='MachineLearning']): 
            vars(opt)[f'{item["model"].__name__}'] = True
    yaml_save(os.path.join(save_dir, 'updated_opt.yaml'), vars(opt))

    """ 
    Set random seed 
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    set_seed(opt.seed)
    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # for Multi-GPU, exception safe

    """ Read data config """
    data = yaml_load(opt.source)
    if data['features'] is None: 
        opt.CyclicalPattern = True
        data['features'] = []

    """ Get all files with given extensions and read """
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

    """ Data preprocessing """
    if data['date'] is not None:
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
    # test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(opt.batchsz)

    console = Console(record=True)
    table = Table(title="[cyan]Results", 
                  show_header=True, 
                  header_style="bold magenta",
                  box=rbox.ROUNDED)
    # table header
    for name in ['Name', *list(used_metric())]: table.add_column(f'[green]{name}', justify='center')

    # Normalization
    if opt.Normalization:
        norm = Normalization()
        norm.adapt(np.vstack((X_train, X_val, X_test)))
    else:
        norm = None

    errors = []
    for item in model_dict:
        if not vars(opt)[f'{item["model"].__name__}']: continue
        model = item['model'](input_shape=X_train.shape[-2:], output_shape=opt.labelsz, seed=opt.seed,
                              config_path=item.get('config'), 
                              units=item.get('units'), normalize_layer=norm)
        try:
            model.fit(patience=opt.patience, save_dir=save_dir, optimizer=opt.optimizer, loss=opt.loss, lr=opt.lr, epochs=opt.epochs, learning_rate=opt.lr, batchsz=opt.batchsz,
                      X_train=X_train, y_train=y_train,
                      X_val=X_val, y_val=y_val)
            weight=os.path.join(save_dir, 'weights', f"{model.model.name}_best.h5")
            if not os.path.exists(weight): weight = model.save(save_dir=os.path.join(save_dir, 'weights'),
                                                               file_name=model.model.name)
            if weight is not None: model.load(weight)
            yhat = model.predict(X=X_test)
            scores = calculate_score(y=y_test, yhat=yhat)
            table.add_row(model.model.name, *scores)
        except Exception as e:
            errors.append([model.model.name, str(e)])
            table.add_row(model.model.name, *list('_' * len(used_metric())))
        console.print(table)
        console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
    for error in errors: print(f'{error[0]}\n{error[1]}', end='\n\n\n====================================================================\n\n\n')

def run(**kwargs):
    """ 
    Usage (example)
        import main
        main.run(all=True, 
                 source=data.yaml,
                 Normalization=True)
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)