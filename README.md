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

# Case 1: Multiple ids, split by id (second last component in the data path)

```
# file: data.yaml
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
python .\main.py --inputsz=15 --DirAsFeature=1 --SplitDirFeature=0
```

# Case 2: Multiple ids, split by column

```
# file: data.yaml
data:
  - .\data\salinity\2019_all.csv
target: average
date: dt
features:
```

```
python .\main.py --inputsz=15 --DirAsFeature=0 --SplitDirFeature=-1 --SplitFeature='station'
```
