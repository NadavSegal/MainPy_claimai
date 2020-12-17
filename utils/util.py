import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class Figures:

    def __init__(self, fig_n, title, ylab, n_samples):
        dat = range(n_samples)
        self.fig_n = fig_n
        self.fig = plt.figure(fig_n)
        self.ax = self.fig.add_subplot(111)
        self.y1, = self.ax.plot(dat, '.', label='Predicted', markersize=16)
        self.y2, = self.ax.plot(dat, '.', label='Real', markersize=16)
        plt.legend(handles=[self.y1, self.y2])

        plt.xlabel('Sample Number')
        plt.ylabel(ylab)
        plt.title(title)
        plt.show(block=False)
        plt.grid()

    def plot_bar(self, chart_sum, keys):
        self.fig.figure(self.fig_n)
        self.fig.xticks = keys
        self.fig.bar(keys, chart_sum)
        self.fig.draw()

    def plot_dots(self, y1, y2):
        y1 = y1.cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
        plt.figure(self.fig_n)
        self.y1.set_ydata(y1)
        self.y2.set_ydata(y2)
        plt.ylim(min(min(y2), min(y1))*0.8, max(max(y2), max(y1))*1.2)
        plt.pause(0.5)

    def save_figs(self, path, epoch):
        multi_page(path + '/' + 'figures_epoch_' + epoch)


def multi_page(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
