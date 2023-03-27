import matplotlib.pyplot as plt

def save_plot(filename, data, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    for datum in data: ax.plot(*datum['data'], color=datum['color'], label=datum['label'])
    ax.legend()
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    fig.savefig(filename)