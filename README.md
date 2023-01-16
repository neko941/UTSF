# Ultimate Time Series Forcasting

# Installation
## Colab
```bash
!git clone https://github.com/neko941/UTSF
%cd UTSF
!pip install -r requirements_colab.txt
```
```python
import main
main.run(all=True, 
		 source='/content/UTSF/data/stocks/TSLA-Tesla.csv',
		 epochs=100)
```
## Window (virtualenv)
### Download repo
```bash
git clone https://github.com/neko941/UTSF
cd UTSF
```

### Upgrade pip
```bash
pip install --upgrade pip
```

### Install virtual environment
```bash
pip install virtualenv
```

### Create new virtual environment
```bash
python -m virtualenv env --python=3.10.8
```

### Run virtual environment
```bash
.\env\Scripts\activate
```

### Install packages
```bash
pip install -r requirements.txt
```
