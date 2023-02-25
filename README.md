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
