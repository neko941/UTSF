import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # disable absl INFO and WARNING log messages

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import json
import time
import argparse
import numpy as np
import pandas as pd
from keras.layers import Normalization


# general utils
from utils.general import yaml_save
from utils.general import yaml_load
from utils.general import increment_path
from utils.general import convert_seconds
from utils.general import SetSeed
from utils.activations import get_custom_activations
from utils.visualize import save_plot

# performance metrics
from utils.metrics import used_metric

# dataset slicing 
# from utils.dataset import slicing_window
# from utils.dataset import ReadFileAddFetures
from utils.dataset import DatasetController

# display results
from rich import box as rbox
from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI

# machine learning models
from models.MachineLearning import ExtremeGradientBoostingRegression
from models.MachineLearning import SupportVectorMachinesClassification
from models.MachineLearning import SupportVectorMachinesRegression
from models.MachineLearning import NuSupportVectorMachinesRegression
from models.MachineLearning import LinearSupportVectorMachinesRegression
from models.MachineLearning import LinearRegression_
from models.MachineLearning import LassoRegression
from models.MachineLearning import LassoCrossValidation
from models.MachineLearning import RidgeRegression
from models.MachineLearning import RidgeClassification
from models.MachineLearning import RidgeCrossValidation
from models.MachineLearning import KernelRidgeRegression
from models.MachineLearning import CatBoostRegression
from models.MachineLearning import LightGBM
from models.MachineLearning import RandomForestRegression
from models.MachineLearning import RandomForestClassification
from models.MachineLearning import GradientBoostingRegression
from models.MachineLearning import GradientBoostingClassification
from models.MachineLearning import ExtraTreesRegression
from models.MachineLearning import ExtraTreesClassification
from models.MachineLearning import BaggingRegression
from models.MachineLearning import BaggingClassification
from models.MachineLearning import AdaBoostRegression
from models.MachineLearning import AdaBoostClassification
from models.MachineLearning import OrthogonalMatchingPursuitRegression
from models.MachineLearning import OrthogonalMatchingPursuitCrossValidation

# deep learning models
from models.RNN import VanillaRNN__Tensorflow    
from models.RNN import BiRNN__Tensorflow
from models.LSTM import VanillaLSTM__Tensorflow
from models.LSTM import VanillaLSTM__Pytorch    
from models.LSTM import BiLSTM__Tensorflow 
from models.GRU import VanillaGRU__Tensorflow
from models.GRU import BiGRU__Tensorflow
from models.LTSF_Linear import LTSF_Linear__Tensorflow
from models.LTSF_Linear import LTSF_NLinear__Tensorflow
from models.LTSF_Linear import LTSF_DLinear__Tensorflow
from models.Concatenated import RNNcLSTM__Tensorflow
from models.Concatenated import BiRNNcBiLSTM__Tensorflow
from models.Concatenated import LSTMcGRU__Tensorflow
from models.Concatenated import BiLSTMcBiGRU__Tensorflow
from models.EncoderDecoder import EncoderDecoder__Tensorflow
from models.EncoderDecoder import BiEncoderDecoder__Tensorflow
from models.EncoderDecoder import CNNcLSTMcEncoderDecoder__Tensorflow

""" 
TODO:
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LarsCV
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDRegressor
    from sklearn.impute import KNNImputer

    from models.LSTNet import LSTNet__TensorFlow
    {
        'model' : LSTNet__TensorFlow,
        'help' : '',
        'filters' : [100],
        'kernels' : [3],
        'dropouts' : [0, 0, 0],
        'type' : 'Tensorflow',
        'activations': ['relu', 'relu', 'relu', 'relu']
    }
    from models.AutoEncoders import StackedAutoEncoders__TensorFlow
    {
        'model' : StackedAutoEncoders__TensorFlow,
        'help' : '',
        'units' : [12, 400, 400, 400],
        'type' : 'Tensorflow',
    }
    from models.LSTM import ConvLSTM__Tensorflow   
    }
        'model' : ConvLSTM__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [28, 64, 32, 32],
        'activations': ['tanh', 'tanh', 'tanh', None, None]
    }
    from models.Transformer import VanillaTransformer__Tensorflow
    {
        'model' : VanillaTransformer__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [512, 512, 512, 512, 512, 512, 512],
        'dropouts' : [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'activations' : ['gelu', 'gelu']
    }
    
    from models.TabTransformer import TabTransformer
    from models.NBeats import NBeats
    from models.Averaging import StackingAveragedModels
    from models.Averaging import AveragingModels
"""

model_dict = [
    { 
        'model' : LinearRegression_,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/LinearRegression.yaml'
    },{
        'model' : LassoRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/Lasso.yaml'
    },{
        'model' : LassoCrossValidation,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/LassoCrossValidation.yaml'
    },{
        'model' : RidgeRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/Ridge.yaml'
    },{
        'model' : RidgeClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/RidgeClassification.yaml'
    },{
        'model' : RidgeCrossValidation,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/RidgeCrossValidation.yaml'
    },{
        'model' : KernelRidgeRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/KernelRidge.yaml'
    },{
        'model' : ExtremeGradientBoostingRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/XGBoost.yaml'
    },{
        'model' : SupportVectorMachinesClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/SVC.yaml'
    },{
        'model' : SupportVectorMachinesRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/SVR.yaml'
    },{
        'model' : NuSupportVectorMachinesRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/NuSVR.yaml' #TODO: finish this file
    },{
        'model' : LinearSupportVectorMachinesRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/LinearSVR.yaml'
    },{
        'model' : LightGBM,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/LightGBM.yaml'
    },{
        'model' : CatBoostRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/CatBoost.yaml'
    },{
        'model' : RandomForestRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/RandomForestRegression.yaml' #TODO: finish this file
    },{
        'model' : RandomForestClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/RandomForestClassification.yaml' #TODO: finish this file
    },{
        'model' : GradientBoostingRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/GradientBoostingRegression.yaml' #TODO: finish this file
    },{
        'model' : GradientBoostingClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/GradientBoostingClassification.yaml' #TODO: finish this file
    },{
        'model' : ExtraTreesRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/ExtraTreesRegression.yaml' #TODO: finish this file
    },{
        'model' : ExtraTreesClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/ExtraTreesClassification.yaml' #TODO: finish this file
    },{
        'model' : BaggingRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/BaggingRegression.yaml' #TODO: finish this file
    },{
        'model' : BaggingClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/BaggingClassification.yaml' #TODO: finish this file
    },{
        'model' : AdaBoostRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/AdaBoostRegression.yaml' #TODO: finish this file
    },{
        'model' : AdaBoostClassification,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/AdaBoostClassification.yaml' #TODO: finish this file
    },{
        'model' : OrthogonalMatchingPursuitRegression,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/OrthogonalMatchingPursuitRegression.yaml' #TODO: finish this file
    },{
        'model' : OrthogonalMatchingPursuitCrossValidation,
        'help' : '',
        'type' : 'MachineLearning',
        'config': 'configs/OrthogonalMatchingPursuitCrossValidation.yaml' #TODO: finish this file
    },{
        'model' : VanillaRNN__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [128, 32],
        'activations': ['tanh', None, None]
    },{
        'model' : BiRNN__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [128, 64, 32, 32],
        'activations': ['tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : VanillaLSTM__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [512, 512, 256, 256, 128, 128, 64],
        'activations': ['tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'relu', None]
    },{ 
    #     'model' : VanillaLSTM__Pytorch,
    #     'help' : '',
    #     'type' : 'Pytorch',
    #     'units' : [128, 64, 32, 32],
    #     'activations': ['tanh', 'tanh', 'tanh', None, None]
    # },{ 
        'model' : BiLSTM__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [28, 64, 32, 32],
        'activations': ['tanh', 'tanh', 'tanh', None, None]
    },{ 
        'model' : VanillaGRU__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [128, 32],
        'activations': ['tanh', None, None]
    },{
        'model' : BiGRU__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
        'units' : [128, 64, 32, 32],
        'activations': ['tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : LTSF_Linear__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
    },{
        'model' : LTSF_NLinear__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
    },{
        'model' : LTSF_DLinear__Tensorflow,
        'help' : '',
        'type' : 'Tensorflow',
    },{
        'model' : RNNcLSTM__Tensorflow,
        'help' : '',
        'units' : [128, 64, 32],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : BiRNNcBiLSTM__Tensorflow,
        'help' : '',
        'units' : [128, 64, 32],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : LSTMcGRU__Tensorflow,
        'help' : '',
        'units' : [128, 64, 32],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : BiLSTMcBiGRU__Tensorflow,
        'help' : '',
        'units' : [128, 64, 32],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', 'tanh', 'tanh', None, None]
    },{
        'model' : EncoderDecoder__Tensorflow,
        'help' : '',
        'units' : [256, 128],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', None]
    },{
        'model' : BiEncoderDecoder__Tensorflow,
        'help' : '',
        'units' : [256, 128],
        'type' : 'Tensorflow',
        'activations': ['tanh', 'tanh', None]
    },{
        'model' : CNNcLSTMcEncoderDecoder__Tensorflow,
        'help' : '',
        'units' : [128, 64],
        'filters' : [64, 64],
        'kernels' : [3, 3],
        'dropouts' : [0.1, 0.1, 0.1],
        'type' : 'Tensorflow',
        'activations': ['relu', 'relu', 'relu', 'relu']
    }
]
for model in model_dict: model.setdefault("activations", [None])

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

    parser.add_argument('--indexCol', type=int, default=None, help='')
    parser.add_argument('--delimiter', type=str, default=',', help='')

    parser.add_argument('--granularity', type=int, default=1, help='by minutes')
    parser.add_argument('--startTimeId', type=int, default=0, help='by minutes')

    parser.add_argument('--source', default='data.yaml', help='dataset')
    parser.add_argument('--patience', type=int, default=1000, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')
    parser.add_argument('--round', type=int, default=-1, help='Round decimals in results, -1 to disable')
    parser.add_argument('--individual', action='store_true', help='for LTSF Linear models')
    parser.add_argument('--debug', action='store_true', help='print debug information in table')
    parser.add_argument('--multimodels', action='store_true', help='split data of n segment ids for n models ')

    parser.add_argument('--AutoInterpolate', type=str, choices=['', 'forward', 'backward'], default='', help='')
    parser.add_argument('--CyclicalPattern', action='store_true', help='Add sin cos cyclical feature')
    parser.add_argument('--Normalization', action='store_true', help='')
    parser.add_argument('--DirAsFeature', type=int, default=0, help='')
    parser.add_argument('--DirFeatureName', type=str, default='dir', help='')
    parser.add_argument('--SplitDirFeature', type=int, default=0, help='Segmentation using dir feature')
    parser.add_argument('--SplitFeature', type=str, default=None, help='Segmentation using feature')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--all', action='store_true', help='Use all available models')
    parser.add_argument('--MachineLearning', action='store_true', help='')
    parser.add_argument('--DeepLearning', action='store_true', help='')
    parser.add_argument('--Tensorflow', action='store_true', help='')
    parser.add_argument('--Pytorch', action='store_true', help='')
    parser.add_argument('--LTSF', action='store_true', help='Using all LTSF Linear Models')

    for item in model_dict:
        parser.add_argument(f"--{item['model'].__name__}", action='store_true', help=f"{item['help']}")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """ Get the save directory for this run """
    save_dir = str(increment_path(Path(opt.project) / opt.name, overwrite=opt.overwrite, mkdir=True))
    visualize_path = os.path.join(save_dir, 'plots')
    os.makedirs(name=visualize_path, exist_ok=True)
    
    """ Save init options """
    yaml_save(os.path.join(save_dir, 'opt.yaml'), vars(opt))

    """ Set seed """
    opt.seed = SetSeed(opt.seed)

    """ Update options """
    if opt.all:
        opt.MachineLearning = True
        opt.DeepLearning = True
    if opt.DeepLearning:
        opt.Tensorflow = True
        opt.Pytorch = True
    if opt.LTSF:
        opt.LTSF_Linear__Tensorflow = True
        opt.LTSF_NLinear__Tensorflow = True
        opt.LTSF_DLinear__Tensorflow = True
    for item in model_dict:
        if any([opt.Tensorflow and item['type']=='Tensorflow',
                opt.Pytorch and item['type']=='Pytorch',
                opt.MachineLearning and item['type']=='MachineLearning']): 
            vars(opt)[f'{item["model"].__name__}'] = True

    """ Save updated options """
    yaml_save(os.path.join(save_dir, 'updated_opt.yaml'), vars(opt))

    """ Set device """
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    """ Read data config """
    data = yaml_load(opt.source)
    if data['features'] is None: data['features'] = []
    elif not isinstance(data['features'], list): data['features'] = [data['features']]
    # if not isinstance(data['target'], list): data['target'] = [data['target']]


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
    # df, dir_feature = ReadFileAddFetures(csvs=csvs, 
    #                                      DirAsFeature=opt.DirAsFeature,
    #                                      ColName=opt.DirFeatureName,
    #                                      delimiter=opt.delimiter,
    #                                      index_col=opt.indexCol
    #                                      )
    # # print(df, dir_feature)
    # data['features'].extend(dir_feature)
    # # print(data['features'])
    # # print(df)
    # # exit()

    # """ Convert to datetime """
    # if 'time_as_id' in data and 'date' in data: 
    #     assert df[data['time_as_id']].max() * opt.granularity + opt.startTimeId - 24*60 <= 0, f'time id max should be {(24*60  - opt.startTimeId) / opt.granularity} else it will exceed to the next day'
    #     df[data['date']] = df.apply(lambda row: pd.to_datetime(row[data['date']]) + pd.to_timedelta((row[data['time_as_id']]-1)*opt.granularity+opt.startTimeId, unit='m'), axis=1)
    

    # """ Data preprocessing """
    # # if not isinstance(data['target'], list): data['target'] = [data['target']]
    # if data['date'] is not None:
    #     # get used cols
    #     cols = []
    #     for i in [data['date'], data['features'], data['target']]: 
    #         if isinstance(i, list): cols.extend(i)
    #         else: cols.append(i) 
    #     df = df[cols]

    #     # convert date to type datetime
    #     df[data['date']] = pd.to_datetime(df[data['date']])

    #     # auto fill missing data
    #     # TODO: drop date for case the does not need a whole year
    #     if opt.AutoInterpolate != '':
    #         df = pd.merge(df,
    #                       pd.DataFrame(pd.date_range(min(df[data['date']]), max(df[data['date']])), columns=[data['date']]),
    #                       how='right',
    #                       left_on=[data['date']],
    #                       right_on = [data['date']])
    #         df.fillna(method=f'{list(opt.AutoInterpolate)[0].lower()}fill', inplace=True)

    #     # sort data by date
    #     df.sort_values(data['date'], inplace=True, ignore_index=True)

    #     # add month sin, month cos (cyclical pattern)
    #     if opt.CyclicalPattern:
    #         # Extracting the hour of day
    #         # d["hour"] = [x.hour for x in d["dt"]]
    #         # # Creating the cyclical daily feature 
    #         # d["day_cos"] = [np.cos(x * (2 * np.pi / 24)) for x in d["hour"]]
    #         # d["day_sin"] = [np.sin(x * (2 * np.pi / 24)) for x in d["hour"]]

    #         d = [x.timestamp() for x in df[f"{data['date']}"]]
    #         day = 24 * 60 * 60 # Seconds in day  
    #         year = (365.2425) * day # Seconds in year 

    #         if df[data['date']].dt.day.nunique() > 1:
    #             df.insert(loc=0, column='day_cos', value=[np.cos((x) * (2 * np.pi / day)) for x in d])
    #             df.insert(loc=0, column='day_sin', value=[np.sin((x) * (2 * np.pi / day)) for x in d]) 
            
    #         if df[data['date']].dt.month.nunique() > 1:
    #             df.insert(loc=0, column='month_cos', value=[np.cos((x) * (2 * np.pi / year)) for x in d])
    #             df.insert(loc=0, column='month_sin', value=[np.sin((x) * (2 * np.pi / year)) for x in d]) 

    # assert not all([opt.DirAsFeature != 0, opt.SplitFeature is not None])

    # if opt.DirAsFeature != 0 and opt.SplitDirFeature != -1: segment_feature = dir_feature[opt.SplitDirFeature]
    # elif opt.SplitFeature is not None: segment_feature = opt.SplitFeature
    # else: segment_feature = None

    # X_train, y_train, X_val, y_val, X_test, y_test = slicing_window(df=df, 
    #                                                                 date_feature=data['date'],
    #                                                                 segment_feature=segment_feature,
    #                                                                 split_ratio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz), 
    #                                                                 input_size=opt.inputsz, 
    #                                                                 label_size=opt.labelsz, 
    #                                                                 offset=opt.offset, 
    #                                                                 label_name=data['target'],
    #                                                                 multimodels=opt.multimodels)

    dataset = DatasetController(trainFeatures=data['features'],
                                dateFeature=data['date'],
                                targetFeatures=data['target'],
                                granularity=opt.granularity)
    dataset.ReadFileAddFetures(csvs=csvs, 
                               dirAsFeature=opt.DirAsFeature,
                               newColumnName=opt.DirFeatureName,
                               delimiter=opt.delimiter,
                               indexColumnToDrop=opt.indexCol,
                               hasHeader=True)

    if 'time_as_id' in data and 'date' in data: 
        dataset.TimeIDToDateTime(timeIDColumn=data['time_as_id'], 
                                 startTimeId=opt.startTimeId)
    dataset.GetUsedColumn()
    dataset.GetSegmentFeature(dirAsFeature=opt.DirAsFeature, 
                              splitDirFeature=opt.SplitDirFeature, 
                              splitFeature=opt.SplitFeature)
    dataset.SplittingData(splitRatio=(opt.trainsz, opt.valsz, 1-opt.trainsz-opt.valsz), 
                          lag=opt.inputsz, 
                          ahead=opt.labelsz, 
                          offset=opt.offset, 
                          multimodels=opt.multimodels)
    dataset.save(save_dir=save_dir)
    with open(os.path.join(save_dir, "num_samples.json"), "w") as final: json.dump(dataset.num_samples, final, indent = 4) 
    # ids = [i['id'] for i in sorted(dataset.num_samples, key=lambda d: d['train'])[:500]]
    # print(ids)
    # with open('somefile.txt', 'a') as the_file:
    #     the_file.write('\n'.join(ids))
    # exit()
     
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val, dataset.X_test, dataset.y_test

    

    np.random.shuffle(X_train)
    np.random.shuffle(y_train)
    np.random.shuffle(X_val)
    np.random.shuffle(y_val)
    np.random.shuffle(X_test)
    np.random.shuffle(y_test)

    # if opt.batchsz == -1:
    #     import math
    #     import tensorflow as tf
    #     def autobatch(x, y, batch):
    #         while True:
    #             try:
    #                 tf.data.Dataset.from_tensor_slices((x, y)).batch(batchsz).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    #                 print('dasdasd')
    #                 break
    #             except:
    #                 autobatch(x, y, batch/2)
    #         return batch
    #     opt.bathsz = autobatch(X_train, y_train, 2**(math.floor(math.log2(len(X_train)))))
    #     print('dasdasd')
        
    #     print(opt.bathsz)

    
    console = Console(record=True)
    table = Table(title="[cyan]Results", 
                  show_header=True, 
                  header_style="bold magenta",
                  box=rbox.ROUNDED,
                  show_lines=True)
    for name in ['Name', *list(used_metric())]: table.add_column(f'[green]{name}', justify='center')
    
    # train_console = Console(record=True)
    # train_table = Table(title="[cyan]Test on Train", 
    #                     show_header=True, 
    #                     header_style="bold magenta",
    #                     box=rbox.ROUNDED,
    #                     show_lines=True)
    # for name in ['Name', *list(used_metric())]: train_table.add_column(f'[green]{name}', justify='center')

    if opt.debug:
        debug_console = Console(record=True)
        debug_table = Table(title="[cyan]Debug", 
                    show_header=True, 
                    header_style="bold magenta",
                    box=rbox.ROUNDED,
                    show_lines=True)
        for name in ['Name', 'Time', 'Activation', 'predShape']: debug_table.add_column(f'[green]{name}', justify='center')
    
    """ Data Normalization """
    if opt.Normalization:
        norm = Normalization()
        norm.adapt(np.vstack((X_train, X_val, X_test)))
    else:
        norm = None

    errors = []
    get_custom_activations()
    if opt.multimodels:
        for item in model_dict:
            start = time.time()
            num_model = X_train.shape[0]
            if not vars(opt)[f'{item["model"].__name__}']: continue
            model_list = []
            all_scores = []
            # train_all_scores = []
            for model_id in range(num_model):
                model = item['model'](input_shape=X_train.shape[-2:], output_shape=opt.labelsz, seed=opt.seed,
                                      config_path=item.get('config'), 
                                      activations=item.get('activations'),
                                      units=item.get('units'), 
                                      kernels=item.get('kernels'), 
                                      filters=item.get('filters'),
                                      dropouts=item.get('dropouts'),
                                      lag=opt.inputsz,
                                      individual=opt.individual,
                                      normalize_layer=norm,
                                      enc_in=1) #TODO: make this dynamic enc_in=len(data['target'])
                model.build()
                model_list.append(model)
                model_list[model_id].__class__.__name__ = f'{model_list[model_id].__class__.__name__.split("~")[0]}~{model_id}'
                sub_X_train = X_train[model_id]
                sub_y_train = y_train[model_id]
                sub_X_val = X_val[model_id]
                sub_y_val = y_val[model_id]
                sub_X_test = X_test[model_id]
                sub_y_test = y_test[model_id]
                try:
                    model_list[model_id].fit(patience=opt.patience, save_dir=save_dir, optimizer=opt.optimizer, loss=opt.loss, lr=opt.lr, epochs=opt.epochs, learning_rate=opt.lr, batchsz=opt.batchsz,
                            X_train=sub_X_train, y_train=sub_y_train,
                            X_val=sub_X_val, y_val=sub_y_val)
                    weight=os.path.join(save_dir, 'weights', f"{model_list[model_id].__class__.__name__}_best.h5")
                    if not os.path.exists(weight): weight = model_list[model_id].save(save_dir=save_dir,
                                                                                      file_name=model_list[model_id].__class__.__name__)
                    if weight is not None: model_list[model_id].load(weight)
                    yhat = model_list[model_id].predict(X=sub_X_test)
                    scores = model_list[model_id].score(y=sub_y_test, yhat=yhat, r=opt.round)
                    all_scores.append(scores)
                    
                    if opt.labelsz == 1:
                        save_plot(filename=os.path.join(visualize_path, f'{model_list[model_id].__class__.__name__}.png'),
                                  data=[{'data': [range(len(sub_y_test)), sub_y_test],
                                          'color': 'green',
                                          'label': 'y'},
                                          {'data': [range(len(yhat)), yhat],
                                          'color': 'red',
                                          'label': 'yhat'}],
                                  xlabel='Sample',
                                  ylabel='Value')
                    if model.history is not None:
                        loss = model_list[model_id].history.history.get('loss')
                        val_loss = model_list[model_id].history.history.get('val_loss')
                        if all([len(loss)>1, len(val_loss)>1]):
                            save_plot(filename=os.path.join(visualize_path, f'{model_list[model_id].__class__.__name__}-Loss.png'),
                                      data=[{'data': [range(len(loss)), loss],
                                              'color': 'green',
                                              'label': 'loss'},
                                              {'data': [range(len(val_loss)), val_loss],
                                              'color': 'red',
                                              'label': 'val_loss'}],
                                      xlabel='Epoch',
                                      ylabel='Loss Value')
                    # yhat = model_list[model_id].predict(X=sub_X_train)
                    # scores = model_list[model_id].score(y=sub_y_train, yhat=yhat, r=opt.round)
                    # train_all_scores.append(scores)

                    # save_plot(filename=os.path.join(visualize_path, f'{model_list[model_id].__class__.__name__}_train.png'),
                    #           data=[{'data': sub_y_train,
                    #                  'color': 'green',
                    #                  'label': 'y'},
                    #                  {'data': yhat,
                    #                  'color': 'red',
                    #                  'label': 'yhat'}])
                except Exception as e:
                    errors.append([model_list[model_id].__class__.__name__, str(e)])
            try:
                table.add_row(item["model"].__name__.split('~')[0], *[str(a) for a in np.mean(np.array(all_scores).astype(np.float64), axis=0)])
                # train_table.add_row(item["model"].__name__, *[str(a) for a in np.mean(np.array(train_all_scores).astype(np.float64), axis=0)])
                if opt.debug:
                    debug_table.add_row(item["model"].__name__.split('~')[0], 
                                        convert_seconds(time.time()-start), 
                                        '\n'.join(['None' if a == None else a for a in item.get('activations')]),
                                        str(yhat.shape)
                                        )
                # save_plot(filename=os.path.join(visualize_path, f'{item["model"].__name__.split('~')[0]}-Loss.png'),
                #                 data=[{'data': [range(len(loss)), loss],
                #                         'color': 'green',
                #                         'label': 'loss'},
                #                         {'data': [range(len(val_loss)), val_loss],
                #                         'color': 'red',
                #                         'label': 'val_loss'}],
                #                 xlabel='Epoch',
                #                 ylabel='Loss Value')
            except Exception as e:
                table.add_row(item["model"].__name__.split('~')[0], *list('_' * len(used_metric())))
                if opt.debug:
                    theshape = str(model_list[model_id].predict(X_test).shape) if model_list[model_id].model is not None else '_'
                    debug_table.add_row(item["model"].__name__.split('~')[0], 
                                        convert_seconds(time.time()-start), 
                                        '\n'.join(['None' if a == None else a for a in item.get('activations')]),
                                        theshape)
                # train_table.add_row(item["model"].__name__, *list('_' * len(used_metric())))
            console.print(table)
            console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
            # train_console.print(train_table)
            # train_console.save_svg(os.path.join(save_dir, 'results_on_train.svg'), theme=MONOKAI)
    else:
        for item in model_dict:
            start = time.time()
            if not vars(opt)[f'{item["model"].__name__}']: continue
            model = item['model'](input_shape=X_train.shape[-2:], output_shape=opt.labelsz, seed=opt.seed,
                                  config_path=item.get('config'), 
                                  activations=item.get('activations'),
                                  units=item.get('units'), 
                                  kernels=item.get('kernels'), 
                                  filters=item.get('filters'),
                                  dropouts=item.get('dropouts'),
                                  lag=opt.inputsz,
                                  individual=opt.individual,
                                  normalize_layer=norm,
                                  enc_in=1) #TODO: make this dynamic enc_in=len(data['target'])
            model.build()
            try:
                model.fit(patience=opt.patience, 
                          save_dir=save_dir, 
                          optimizer=opt.optimizer, 
                          loss=opt.loss, 
                          lr=opt.lr, 
                          epochs=opt.epochs, 
                          learning_rate=opt.lr, 
                          batchsz=opt.batchsz,
                          X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val)
                weight=os.path.join(save_dir, 'weights', f"{model.__class__.__name__}_best.h5")
                if not os.path.exists(weight): weight = model.save(save_dir=save_dir, file_name=model.__class__.__name__)
                else: model.save(save_dir=save_dir, file_name=model.__class__.__name__)
                # weight = r'runs\exp809\weights\VanillaLSTM__Tensorflow_best.h5'
                if weight is not None: model.load(weight)
                yhat = model.predict(X=X_test)
                scores = model.score(y=y_test, 
                                     yhat=yhat, 
                                     r=opt.round,
                                     path=save_dir)
                
                if opt.labelsz == 1:
                    save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}.png'),
                              data=[{'data': [range(len(y_test)), y_test],
                                      'color': 'green',
                                      'label': 'y'},
                                     {'data': [range(len(yhat)), yhat],
                                      'color': 'red',
                                      'label': 'yhat'}],
                              xlabel='Sample',
                              ylabel='Value')
                if model.history is not None:
                    loss = model.history.history.get('loss')
                    val_loss = model.history.history.get('val_loss')
                    if all([len(loss)>1, len(val_loss)>1]):
                        save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}-Loss.png'),
                                  data=[{'data': [range(len(loss)), loss],
                                         'color': 'green',
                                         'label': 'loss'},
                                        {'data': [range(len(val_loss)), val_loss],
                                         'color': 'red',
                                         'label': 'val_loss'}],
                                  xlabel='Epoch',
                                  ylabel='Loss Value')
                # if opt.labelsz == 1:
                #     save_plot(filename=os.path.join(visualize_path, f'{model.__class__.__name__}_train.png'),
                #                 data=[{'data': y_train,
                #                         'color': 'green',
                #                         'label': 'y'},
                #                         {'data': yhat,
                #                         'color': 'red',
                #                         'label': 'yhat'}])
                table.add_row(model.__class__.__name__, *scores)
                if opt.debug:
                    debug_table.add_row(model.__class__.__name__, 
                                        convert_seconds(time.time()-start), 
                                        '\n'.join(['None' if a == None else a for a in item.get('activations')]),
                                        str(yhat.shape)
                                        )
            except Exception as e:
                errors.append([model.__class__.__name__, str(e)])
                table.add_row(model.__class__.__name__, *list('_' * len(used_metric())))
                if opt.debug:
                    theshape = str(model.predict(X_test).shape) if model.model is not None else '_'
                    debug_table.add_row(model.__class__.__name__, 
                                        convert_seconds(time.time()-start), 
                                        '\n'.join(['None' if a == None else a for a in item.get('activations')]),
                                        theshape)
            console.print(table)
            console.save_svg(os.path.join(save_dir, 'results.svg'), theme=MONOKAI)
    if opt.debug: 
        debug_console.print(debug_table)
        debug_console.save_svg(os.path.join(save_dir, 'debug.svg'), theme=MONOKAI)
        print(f'{X_train.shape = }')
        print(f'{y_train.shape = }')
        print(f'{X_val.shape = }')
        print(f'{y_val.shape = }')
        print(f'{X_test.shape = }')
        print(f'{y_test.shape = }')
    print(f'\n\n{opt}', end='\n\n\n\n')
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