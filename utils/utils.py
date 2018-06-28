import pickle
from pathlib import Path
import logging

from utils.plots import _plot_confusion_matrix

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def read_pickle_file(pickle_file_path):
    with Path(pickle_file_path).open('rb') as pickle_file:
        log.info(f'loading {Path(pickle_file_path).as_posix()}')
        ret = pickle.load(pickle_file)
    return ret


def write_pickle_file(obj_to_pickle, pickle_file_path):
    with Path(pickle_file_path).open('wb') as pickle_file:
        log.info(f'saving to {Path(pickle_file_path).as_posix()}')
        pickle.dump(obj_to_pickle, pickle_file, protocol=4)
        
        
def normalize_pixels(x):
    x = x.astype('float32')
    x /= 255
    return x

class MURAMetrics():
    
    def __init__(self, true_label, pred_label):
        assert true_label.shape[0] == pred_label.shape[0], (
            'true_label and pred_laben must have the same length!')
        self.y = true_label
        self.yhat = pred_label
        self.N = true_label.shape[0]
        self.cm = confusion_matrix(true_label, pred_label)
        
    def accuracy(self):
        return (self.cm[0, 0] + self.cm[1, 1]) / self.N

    def kappa(self):
        p_observed = self.accuracy()
        p_expected = (
            self.cm[1, :].sum() * self.cm[:, 1].sum() 
            + self.cm[0, :].sum() * self.cm[:, 0].sum()) / self.N**2
        return (p_observed - p_expected) / (1 - p_expected)
    
    def precision_and_recall(self):
        return {
            'precision': self.cm[1, 1] / self.cm[:, 1].sum(),
            'recall': self.cm[1, 1] / self.cm[1, :].sum()
        }
    
    def plot_confusion_matrix(self, normalize=False, figsize=None):
        _plot_confusion_matrix(self.cm, classes=('negative', 'positive'),
                              normalize=normalize, figsize=figsize)
