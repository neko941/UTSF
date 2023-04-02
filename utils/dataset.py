import os
import numpy as np
import pandas as pd
import polars as pl
from rich.style import Style
from rich.progress import track
from rich.console import Console
from utils.general import flatten_list

# No more warning
pd.options.mode.chained_assignment = None 

def ReadFileAddFetures(csvs, DirAsFeature, ColName, delimiter, index_col):
    dir_features = []
    if DirAsFeature == 0: df = pd.concat([pd.read_csv(filepath_or_buffer=filename, delimiter=delimiter, index_col=index_col) for filename in track(csvs, description='Reading data')], axis=0, ignore_index=True)
    else:
        dfs = []
        for csv in track(csvs, description='Reading data'):
            path = os.path.abspath(csv)
            features = [int(p) if p.isdigit() else p for p in path.split(os.sep)[-DirAsFeature-1:-1]]
            df = pd.read_csv(filepath_or_buffer=path, delimiter=delimiter, index_col=index_col)
            for idx, f in enumerate(features): df[f'{ColName}{idx}'] = f
            dir_features.append(f'{ColName}{idx}')
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
    return df, list(set(dir_features))

# def ReadFileAddFetures(csvs, DirAsFeature, ColName, delimiter, index_col, has_header=True):
#     dir_features = []
#     if DirAsFeature == 0: df = pl.concat([pl.read_csv(source=filename, separator=delimiter, has_header=has_header) for filename in track(csvs, description='Reading data')])
#     else:
#         dfs = []
#         for csv in track(csvs, description='Reading data'):
#             path = os.path.abspath(csv)
#             features = [int(p) if p.isdigit() else p for p in path.split(os.sep)[-DirAsFeature-1:-1]]
#             df = pl.read_csv(source=path, separator=delimiter, has_header=has_header)
#             for idx, f in enumerate(features): df = df.with_column(pl.lit(f).alias(f'{ColName}{idx}'))
#             dir_features.append(f'{ColName}{idx}')
#             dfs.append(df)
#         df = pl.concat(dfs)
#     if index_col is not None: df = df.drop(index_col)
#     return df, list(set(dir_features))

def _slicing_window(df, date_feature, df_start_idx, df_end_idx, input_size, label_size, offset, label_name, description):
    features = [] 
    labels = []

    # print(df)

    
    if date_feature in df.columns: df.drop([date_feature], axis=1, inplace=True)

    if df_end_idx == None: df_end_idx = len(df) - label_size - offset
    df_start_idx = df_start_idx + input_size + offset

    # print(f'{len(df) = }')

    for idx in track(range(df_start_idx, df_end_idx+1), description=description):
        feature_start_idx = idx - input_size - offset
        feature_end_idx = feature_start_idx + input_size
        label_start_idx = idx - 1
        label_end_idx = label_start_idx + label_size
        # feature = df[feature_start_idx:feature_end_idx].loc[:, df.columns != date_feature] # Lấy X
        feature = df[feature_start_idx:feature_end_idx]
        label = df[label_name].iloc[label_start_idx:label_end_idx] # Lấy y
        # print(feature[label_name].notna())
        # print(all(feature[label_name].notna()))
        # print(feature)
        # print(label.notna())
        # print(all(label.notna()))
        # print(label)
        # exit()
        # print(feature, '\n', df[['date', label_name]].iloc[label_start_idx:label_end_idx], end='\n\n\n')
        # print(feature, '\n', flatten_list(feature.notna().values.tolist()), '\n', df[['date', label_name]].iloc[label_start_idx:label_end_idx], '\n', df[['date', label_name]].iloc[label_start_idx:label_end_idx].notna(), '\n', all(flatten_list(df[['date', label_name]].iloc[label_start_idx:label_end_idx].notna().values.tolist())), end='\n\n\n')

        if all(flatten_list(label.notna().values.tolist())) and all(flatten_list(feature.notna().values.tolist())): 
            features.append(feature) 
            labels.append(label)

    # for f, l in zip(features, labels): 
        # print(f, '\n', flatten_list(f.notna().values.tolist()), '\n', l, '\n', l.notna(), '\n', all(*l.notna().values.tolist()), end='\n\n\n')
        # print(f, '\n', l)
    # exit()

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def slicing_window(df, 
                   date_feature,
                   segment_feature,
                   split_ratio, input_size, label_size, offset, label_name,
                   granularity=5,
                   multimodels=False):

    if segment_feature:
        if date_feature: df.sort_values(by=[segment_feature, date_feature], inplace=True, ignore_index=True)
        else: df.sort_values(by=[segment_feature], inplace=True, ignore_index=True)

        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
        console = Console()

        for i in df[segment_feature].unique():
            print('PROCESSING ID', end=' ')
            console.print(i, style=Style())

            d = df.loc[df[segment_feature] == i]
            d[date_feature] = pd.to_datetime(d[date_feature])
            d.sort_values(date_feature, inplace=True, ignore_index=True)
            d = pd.merge(d,
                         pd.DataFrame(pd.date_range(start=min(d[date_feature]), 
                                                    end=max(d[date_feature]), 
                                                    freq=f'{granularity}T'), 
                                    columns=[date_feature]),
                                    how='right',
                                    left_on=[date_feature],
                                    right_on = [date_feature])

            dataset_length = len(d)
            TRAIN_END_IDX = int(split_ratio[0] * dataset_length)


            VAL_END_IDX = int(split_ratio[1] * dataset_length) + TRAIN_END_IDX

            # assert dataset_length-VAL_END_IDX >= input_size, f'{input_size = } and for testset we have {dataset_length-VAL_END_IDX} samples ==> cannot widow slide ==> final testset sample = 0'
            x1, y1 = _slicing_window(df=d, 
                                     date_feature=date_feature,
                                     df_start_idx=0,
                                     df_end_idx=TRAIN_END_IDX,
                                     input_size=input_size,
                                     label_size=label_size,
                                     offset=offset,
                                     label_name=label_name,
                                     description='  Training')
            x2, y2 = _slicing_window(df=d, 
                                     date_feature=date_feature,
                                     df_start_idx=TRAIN_END_IDX,
                                     df_end_idx=VAL_END_IDX,
                                     input_size=input_size,
                                     label_size=label_size,
                                     offset=offset,
                                     label_name=label_name,
                                     description='Validation')
            x3, y3 = _slicing_window(df=d, 
                                     date_feature=date_feature,
                                     df_start_idx=VAL_END_IDX,
                                     df_end_idx=None,
                                     input_size=input_size,
                                     label_size=label_size,
                                     offset=offset,
                                     label_name=label_name,
                                     description='   Testing')
            # exit()
            if multimodels:
                X_train.append(x1)
                y_train.append(y1)
                X_val.append(x2)
                y_val.append(y2)
                X_test.append(x3)
                y_test.append(y3)
            else:
                X_train.extend(x1)
                y_train.extend(y1)
                X_val.extend(x2)
                y_val.extend(y2)
                X_test.extend(x3)
                y_test.extend(y3)
        X_train, y_train, X_val, y_val, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)
    else:
        dataset_length = len(df)
        TRAIN_END_IDX = int(split_ratio[0] * dataset_length) 
        VAL_END_IDX = int(split_ratio[1] * dataset_length) + TRAIN_END_IDX
        X_train, y_train = _slicing_window(df, 
                                     date_feature=date_feature,
                                      df_start_idx=0,
                                      df_end_idx=TRAIN_END_IDX,
                                      input_size=input_size,
                                      label_size=label_size,
                                      offset=offset,
                                      label_name=label_name,
                                      description='  Training')

        X_val, y_val = _slicing_window(df, 
                                     date_feature=date_feature,
                                    df_start_idx=TRAIN_END_IDX,
                                    df_end_idx=VAL_END_IDX,
                                    input_size=input_size,
                                    label_size=label_size,
                                    offset=offset,
                                    label_name=label_name,
                                    description='Validation')

        X_test, y_test = _slicing_window(df, 
                                     date_feature=date_feature,
                                        df_start_idx=VAL_END_IDX,
                                        df_end_idx=None,
                                        input_size=input_size,
                                        label_size=label_size,
                                        offset=offset,
                                        label_name=label_name,
                                        description='   Testing')
    return X_train, y_train, X_val, y_val, X_test, y_test


# def _slicing_window(df, ratio, input_size, label_size, offset, label_name):