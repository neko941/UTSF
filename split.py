import os
import polars as pl 
from rich.progress import track
from utils.general import yaml_load
from utils.dataset import DatasetController

data = yaml_load('./data/yaml/test_traffic_a_year.yaml')
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

dataset = DatasetController(trainFeatures=data['features'],
                                dateFeature=data['date'],
                                targetFeatures=data['target'],
                                granularity=5,
                                workers=8)
dataset.ReadFileAddFetures(csvs=csvs, 
                           dirAsFeature=0,
                           newColumnName='dir',
                           delimiter='|',
                           indexColumnToDrop=0,
                           hasHeader=True)

dataset.GetSegmentFeature(dirAsFeature=0, 
                              splitDirFeature=0, 
                              splitFeature='current_geopath_id')

df = dataset.df
segmentFeature = dataset.segmentFeature

for ele in track(df[segmentFeature].unique()):
	d = df.filter(pl.col(segmentFeature) == ele).clone()
	d.write_csv(os.path.join('a_year', f'{ele}.csv'), separator='|')