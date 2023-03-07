import os
import numpy as np
import pandas as pd

def ReadFileAddFetures(csvs, DirAsFeature, ColName):
    # path = os.path.abspath(filename)
    # features = [int(p) if p.isdigit() else p for p in path.split(os.sep)[-DirAsFeature-1:-1]]
    # print(features)
    # df = pd.read_csv(path)
    # for idx, f in enumerate(features): df[f'Dir{idx}'] = f
    # print(df)
    dir_features = []
    if DirAsFeature == 0: df = pd.concat([pd.read_csv(filename) for filename in csvs], axis=0, ignore_index=True)
    else:
        dfs = []
        for csv in csvs:
            path = os.path.abspath(csv)
            features = [int(p) if p.isdigit() else p for p in path.split(os.sep)[-DirAsFeature-1:-1]]
            df = pd.read_csv(path)
            for idx, f in enumerate(features):
                df[f'{ColName}{idx}'] = f
            dir_features.append(f'{ColName}{idx}')
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
    return df, list(set(dir_features))

# # Khai báo hàm Windowing (dùng để tạo các cặp X, y cho time series data)
def _slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset, label_name):
    features = [] # Khai báo list dùng để lưu trữ các X
    labels = [] # Khai báo list dùng để lưu trữ các y

    # Nếu df_end_idx = chỉ mục cuối cùng bảng dữ liệu, cần phải dời xuống 1 khoảng = window size 
    if df_end_idx == None:
        df_end_idx = len(df) - label_size - offset

    df_start_idx = df_start_idx + input_size + offset
    # Duyệt qua từng mẫu dữ liệu
    for idx in range(df_start_idx, df_end_idx):
        feature_start_idx = idx - input_size - offset
        feature_end_idx = feature_start_idx + input_size

        label_start_idx = idx - 1
        label_end_idx = label_start_idx + label_size

        feature = df[feature_start_idx:feature_end_idx] # Lấy X
        label = df[label_name][label_start_idx:label_end_idx] # Lấy y

        features.append(feature) 
        labels.append(label)

    # Chuyển list thành np.ndarrray
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def slicing_window(df, 
                              date_feature,
                              segment_feature,
                              split_ratio, input_size, label_size, offset, label_name):

    if segment_feature:
        if date_feature:
            df.sort_values(by=[segment_feature, date_feature], inplace=True, ignore_index=True)
        else:
            df.sort_values(by=[segment_feature], inplace=True, ignore_index=True)
        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
        for i in df[segment_feature].unique():
            d = df.loc[df[segment_feature] == i]
            d.drop([date_feature], axis=1, inplace=True)
            dataset_length = len(d)
            TRAIN_END_IDX = int(split_ratio[0] * dataset_length) 
            VAL_END_IDX = int(split_ratio[1] * dataset_length) + TRAIN_END_IDX

            # print(df[segment_feature].unique(), dataset_length, TRAIN_END_IDX)

            x, y = _slicing_window(df=d, 
                                        df_start_idx=0,
                                        df_end_idx=TRAIN_END_IDX,
                                        input_size=input_size,
                                        label_size=label_size,
                                        offset=offset,
                                        label_name=label_name)

            X_train.extend(x)
            y_train.extend(y)
            x, y = _slicing_window(d, 
                                        df_start_idx=TRAIN_END_IDX,
                                        df_end_idx=VAL_END_IDX,
                                        input_size=input_size,
                                        label_size=label_size,
                                        offset=offset,
                                        label_name=label_name)
            X_val.extend(x)
            y_val.extend(y)
            x, y = _slicing_window(d, 
                                            df_start_idx=VAL_END_IDX,
                                            df_end_idx=None,
                                            input_size=input_size,
                                            label_size=label_size,
                                            offset=offset,
                                            label_name=label_name)
            X_test.extend(x)
            y_test.extend(y)
            assert dataset_length-VAL_END_IDX > input_size, f'{input_size = } and for testset we have {dataset_length-VAL_END_IDX} samples ==> cannot widow slide ==> final testset sample = 0'
        X_train, y_train, X_val, y_val, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)
    else:
        df.drop([date_feature], axis=1, inplace=True)
        dataset_length = len(df)
        TRAIN_END_IDX = int(split_ratio[0] * dataset_length) 
        VAL_END_IDX = int(split_ratio[1] * dataset_length) + TRAIN_END_IDX
        X_train, y_train = _slicing_window(df, 
                                      df_start_idx=0,
                                      df_end_idx=TRAIN_END_IDX,
                                      input_size=input_size,
                                      label_size=label_size,
                                      offset=offset,
                                      label_name=label_name)

        X_val, y_val = _slicing_window(df, 
                                    df_start_idx=TRAIN_END_IDX,
                                    df_end_idx=VAL_END_IDX,
                                    input_size=input_size,
                                    label_size=label_size,
                                    offset=offset,
                                    label_name=label_name)

        X_test, y_test = _slicing_window(df, 
                                        df_start_idx=VAL_END_IDX,
                                        df_end_idx=None,
                                        input_size=input_size,
                                        label_size=label_size,
                                        offset=offset,
                                        label_name=label_name)
    return X_train, y_train, X_val, y_val, X_test, y_test


# def _slicing_window(df, ratio, input_size, label_size, offset, label_name):