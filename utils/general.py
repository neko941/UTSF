# import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path

def yaml_save(file='opt.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)

def increment_path(path, overwrite=False, sep='', mkdir=False):
    path = Path(path)  
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}' 
            if not os.path.exists(p): 
                if overwrite: p = f'{path}{sep}{n-1}{suffix}'
                break
                    
        path = Path(p)

    if mkdir: path.mkdir(parents=True, exist_ok=True)  

    return path

# def display(history, save_name):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 3))
#     ax = ax.ravel()

#     for i, metric in enumerate(['accuracy', 'loss']):
#         ax[i].plot(history.history[metric])
#         ax[i].plot(history.history['val_' + metric])
#         ax[i].set_title(f'Model {metric}')
#         ax[i].set_xlabel('epochs')
#         ax[i].set_ylabel(metric)
#         ax[i].legend(['Train', 'Validation'])
    
#     plt.show()
#     plt.savefig(save_name)