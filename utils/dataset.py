import os
import numpy as np
import pandas as pd
import polars as pl
from rich.style import Style
from rich.progress import track
from datetime import timedelta
from utils.general import flatten_list
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from time import sleep
from rich import progress
from multiprocessing import Pool
from multiprocessing import Pool
from rich.progress import Progress
from rich.progress import (
                        BarColumn,
                        MofNCompleteColumn,
                        Progress,
                        TextColumn,
                        TimeElapsedColumn,
                        TimeRemainingColumn,
                    )

class DatasetController():
    def __init__(self, trainFeatures, targetFeatures, granularity=1, dateFeature=None, workers=8):
        if trainFeatures is None: trainFeatures = []
        elif not isinstance(trainFeatures, list): trainFeatures = [trainFeatures]
        self.trainFeatures = trainFeatures
        
        self.targetFeatures = targetFeatures
        self.dateFeature = dateFeature
        self.workers = workers

        self.df = None
        self.csvs = []
        self.dirFeatures = []
        self.segmentFeature = None

        self.granularity = granularity

        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test = []

        self.num_samples = []

    def ProgressBar(self):
        return Progress("[bright_cyan][progress.description]{task.description}",
                          BarColumn(),
                          TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                          TextColumn("•Items"),
                          MofNCompleteColumn(), # "{task.completed}/{task.total}",
                          TextColumn("•Remaining"),
                          TimeRemainingColumn(),
                          TextColumn("•Total"),
                          TimeElapsedColumn())

    def ReadFileAddFetures(self, csvs, dirAsFeature=0, newColumnName='dir', delimiter=',', indexColumnToDrop=None, hasHeader=True):
        if not isinstance(csvs, list): self.csvs = [csvs]
        else: self.csvs = csvs 
        self.csvs = [os.path.abspath(csv) for csv in self.csvs]
        if dirAsFeature == 0:
            with self.ProgressBar() as progress:
                df = pl.concat([pl.read_csv(source=csv, separator=delimiter, has_header=hasHeader, try_parse_dates=True) for csv in progress.track(self.csvs, description='  Reading data')])
        else:
            dfs = []
            for csv in track(self.csvs, description='  Reading data'):
                features = [int(p) if p.isdigit() else p for p in csv.split(os.sep)[-dirAsFeature-1:-1]]
                df = pl.read_csv(source=csv, separator=delimiter, has_header=hasHeader, try_parse_dates=True)
                for idx, f in enumerate(features): 
                    df = df.with_column(pl.lit(f).alias(f'{newColumnName}{idx}'))
                self.dirFeatures.append(f'{newColumnName}{idx}')
                dfs.append(df)
            df = pl.concat(dfs)
            self.dirFeatures = list(set(self.dirFeatures))
            self.trainFeatures.extend(self.dirFeatures)
        if indexColumnToDrop: df.drop_in_place(df.columns[indexColumnToDrop])
        
        if self.df is None: self.df = df
        else: self.df = pl.concat([self.df, df])

    def TimeIDToDateTime(self, timeIDColumn, startTimeId=0):
        max_time_id = self.df[timeIDColumn].max() * self.granularity + startTimeId - 24*60
        assert max_time_id <= 0, f'time id max should be {(24*60 - startTimeId) / self.granularity} else it will exceed to the next day'
        self.df = self.df.with_column(pl.col(self.dateFeature).cast(pl.Datetime) + pl.duration(minutes=(pl.col(timeIDColumn)-1)*self.granularity+startTimeId))
    
    def GetUsedColumn(self):
        self.df = self.df[[col for i in [self.dateFeature, self.trainFeatures, self.targetFeatures] for col in (i if isinstance(i, list) else [i])]]

    def UpdateDateColumnDataType(self, dateFormat='%Y-%M-%d'):
        self.df = self.df.with_column(pl.col(self.dateFeature).str.strptime(pl.Date, fmt=dateFormat).cast(pl.Datetime))

    def GetSegmentFeature(self, dirAsFeature=0, splitDirFeature=0, splitFeature=None):
        assert not all([dirAsFeature != 0, splitFeature is not None])
        # if dirAsFeature != 0 and splitDirFeature != -1: self.segmentFeature = self.dirFeatures[splitDirFeature]
        # elif splitFeature is not None: self.segmentFeature = splitFeature
        # else: self.segmentFeature = None
        self.segmentFeature = self.dirFeatures[splitDirFeature] if dirAsFeature != 0 and splitDirFeature != -1 else splitFeature if splitFeature else None
        # TODO: consider if data in segmentFeature are number or not. 

    def CyclicalPattern(self): pass

    def FillDate(self, df=None, low=None, high=None): 
        # TODO: cut date
        if not self.dateFeature: return
        # if df.is_empty(): df=self.df
        if not low: low=self.df[self.dateFeature].min()
        if not high: high=self.df[self.dateFeature].max()

        d = pl.date_range(low=low,
                          high=high,
                          interval=timedelta(minutes=self.granularity),
                          closed='both',
                          name=self.dateFeature).to_frame()
        df = df.join(other=d, 
                     on=self.dateFeature, 
                     how='outer')
        return df

    def TimeBasedCrossValidation(self, args):
        d, lag, ahead, offset, splitRatio = args
        features = []
        labels = []
        for idx in range(len(d)-offset-lag+1):
            feature = d[idx:idx+lag]
            label = d[self.targetFeatures][idx+lag+offset-ahead:idx+lag+offset].to_frame()
            if all(flatten_list(feature.with_columns(pl.all().is_not_null()).rows())) and all(flatten_list(label.with_columns(pl.all().is_not_null()).rows())): 
                labels.append(np.squeeze(label.to_numpy()))
                features.append(feature.to_numpy()) 

        length = len(features)
        if splitRatio[1]==0 and splitRatio[2]==0: 
            train_end = length 
            val_end = length
        elif splitRatio[1]!=0 and splitRatio[2]==0:
            train_end = int(length*splitRatio[0])
            val_end = length
        else:
            train_end = int(length*splitRatio[0])
            val_end = int(length*(splitRatio[0] + splitRatio[1]))
        return [features[0:train_end], features[train_end:val_end], features[val_end:length]], [labels[0:train_end], labels[train_end:val_end], labels[val_end:length]]

    def SplittingData(self, splitRatio, lag, ahead, offset, multimodels=False):
        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])
        
        if offset<ahead: offset=ahead

        if self.segmentFeature:
            data = []
            with self.ProgressBar() as progress:
                for ele in progress.track(self.df[self.segmentFeature].unique(), description='Splitting jobs'):
                    d = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
                    d = self.FillDate(df=d)
                    d.drop_in_place(self.dateFeature) 
                    data.append([d, lag, ahead, offset, splitRatio])
            
            with self.ProgressBar() as progress:
                task_id = progress.add_task("Splitting data", total=len(data))
                with Pool(8) as p:
                    for result in p.imap(self.TimeBasedCrossValidation, data):
                        x = result[0]
                        y = result[1]
                        if multimodels:
                            self.X_train.append(x[0])
                            self.y_train.append(y[0])
                            self.X_val.append(x[1])
                            self.y_val.append(y[1])
                            self.X_test.append(x[2])
                            self.y_test.append(y[2])
                        else:
                            self.X_train.extend(x[0])
                            self.y_train.extend(y[0])
                            self.X_val.extend(x[1])
                            self.y_val.extend(y[1])
                            self.X_test.extend(x[2])
                            self.y_test.extend(y[2])
                        
                        self.num_samples.append({'id' : ele,
                                                 'train': len(y[0]),
                                                 'val': len(y[1]),
                                                 'test': len(y[2])})
                        progress.advance(task_id)
        else:
            d = self.df.clone()
            d = self.FillDate(df=d)
            d.drop_in_place(self.dateFeature) 
            
            x, y = self.TimeBasedCrossValidation(args=[d, lag, ahead, offset, splitRatio]) 
            self.X_train.extend(x[0])
            self.y_train.extend(y[0])
            self.X_val.extend(x[1])
            self.y_val.extend(y[1])
            self.X_test.extend(x[2])
            self.y_test.extend(y[2])

            self.num_samples.append({'id' : ele,
                                     'train': len(y[0]),
                                     'val': len(y[1]),
                                     'test': len(y[2])})

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
    
    def save(self, save_dir):
        save_dir = os.path.join(save_dir, 'values')
        os.makedirs(save_dir, exist_ok=True)
        np.save(open(os.path.join(save_dir, 'X_train.npy'), 'wb'), self.X_train)
        np.save(open(os.path.join(save_dir, 'y_train.npy'), 'wb'), self.y_train)
        np.save(open(os.path.join(save_dir, 'X_val.npy'), 'wb'), self.X_val)
        np.save(open(os.path.join(save_dir, 'y_val.npy'), 'wb'), self.y_val)
        np.save(open(os.path.join(save_dir, 'X_test.npy'), 'wb'), self.X_test)
        np.save(open(os.path.join(save_dir, 'y_test.npy'), 'wb'), self.y_test)

    def display(self): pass