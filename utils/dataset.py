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
    if DirAsFeature == 0: df = pd.concat([pd.read_csv(filename) for filename in csvs], axis=0, ignore_index=True)
    else:
        dfs = []
        for csv in csvs:
            path = os.path.abspath(csv)
            features = [int(p) if p.isdigit() else p for p in path.split(os.sep)[-DirAsFeature-1:-1]]
            df = pd.read_csv(path)
            for idx, f in enumerate(features):
                df[f'{ColName}{idx}'] = f
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
    return df

# Khai báo hàm Windowing (dùng để tạo các cặp X, y cho time series data)
def slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset, label_name):
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

# def slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset, label_name):