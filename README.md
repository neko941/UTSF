# Ultimate Time Series Forcasting

# Installation

### Colab

```bash
!git clone https://github.com/neko941/UTSF
%cd UTSF
!pip install -r requirements_colab.txt
```

### Window (virtualenv)

```bash
git clone https://github.com/neko941/UTSF
cd UTSF
python -m pip install --upgrade pip
pip install virtualenv
python -m virtualenv venv --python=3.10.8
.\venv\Scripts\activate
pip install -r requirements.txt
```

# Usage

### Python

```python
import main
main.run(all=True,
	 source='/content/UTSF/data.yaml',
	 epochs=100)
```

### Command

```bash
python main.py --all --labelsz=7
```

# Salinity

## Case 1: Multiple ids, split by id (second last component in the data path)

```
data:
  - .\data\salinity\602\2019.csv
  - .\data\salinity\614\2019.csv
  - .\data\salinity\615\2019.csv
  - .\data\salinity\616\2019.csv
target: average
date: dt
features:
```

```
python .\main.py --inputsz=5 --labelsz=1 --DirAsFeature=1 --SplitDirFeature=0 --source=./data/yaml/test_case_1.yaml
```

## Case 2: Multiple ids, split by column

```
data:
  - .\data\salinity\2019_all.csv
target: average
date: dt
features:
```

```
python .\main.py --inputsz=5 --labelsz=1 --SplitFeature='station' --source=./data/yaml/test_case_2.yaml
```

## Case 3: One id, multi-step forecasting

```
data:
  - .\data\salinity\615\2014.csv
  - .\data\salinity\615\2015.csv
  - .\data\salinity\615\2016.csv
  - .\data\salinity\615\2017.csv
  - .\data\salinity\615\2018.csv
  - .\data\salinity\615\2019.csv
target: average
date: dt
features:
```

```
python .\main.py --inputsz=15 --labelsz=2 --source=./data/yaml/test_case_3.yaml
```

# Traffic

## Case 1
```
python .\main.py --inputsz=1 --labelsz=1 --trainsz=0.8 --valsz=0.15 --batchsz=1 --SplitFeature=current_geopath_id --source=./data/yaml/test_traffic.yaml --indexCol=0 --delimiter='|' --granularity=5 --startTimeId=240
```
## Case 2

```
data: .\data\traffic\vehiclework_segment_avg_speed.csv
target: speed
date: date
time_as_id: time_id
features: road_segment_id
```

```
python .\main.py --inputsz=5 --labelsz=1 --SplitFeature=current_geopath_id --source=./data/yaml/test_traffic_1.yaml --delimiter='|' --indexCol=0 --granularity=5 --startTimeId=240
```