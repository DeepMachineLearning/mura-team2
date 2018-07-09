from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import utils
from pathlib import Path
import pickle

import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def accuracy(true_label=None, pred_label=None, cm=None):
    if cm is None:
        cm = confusion_matrix(true_label, pred_label)
    N = cm.sum()
    return (cm[0, 0] + cm[1, 1]) / N

def kappa(true_label=None, pred_label=None, cm=None):
    if cm is None:
        cm = confusion_matrix(true_label, pred_label)
    p_observed = accuracy(cm=cm)
    N = cm.sum()
    p_expected = (
        cm[1, :].sum() * cm[:, 1].sum() 
        + cm[0, :].sum() * cm[:, 0].sum()) / N**2
    return (p_observed - p_expected) / (1 - p_expected)

def precision_and_recall(true_label=None, pred_label=None, cm=None):
    if cm is None:
        cm = confusion_matrix(true_label, pred_label)
    return {
        'precision': cm[1, 1] / cm[:, 1].sum(),
        'recall': cm[1, 1] / cm[1, :].sum()
    }

def report_metrics(true_label, pred_label, verbose=1, plot_cm=True):
    cm = confusion_matrix(true_label, pred_label)
    acc = accuracy(cm=cm)
    kap = kappa(cm=cm)
    pnr = precision_and_recall(cm=cm)
    if verbose >= 1:
        log.info(f'Accuracy: {acc}')
        log.info(f'Kappa: {kap}')
        log.info(f'Precision: {pnr["precision"]}')
        log.info(f'Recall: {pnr["recall"]}')
    if plot_cm:
        utils.plot_confusion_matrix(pred_label=pred_label, true_label=true_label, classes=['0', '1'])
    return {'accuracy': acc,
            'kappa': kap,
            **pnr
           }

class MURAMetrics():
    
    def __init__(self, true_label, pred_label, valid_group_mapping_file='./data/MURA-v1.1/valid_groups.pkl'):
        assert true_label.shape[0] == pred_label.shape[0], (
            'true_label and pred_laben must have the same length!')
        self.y = true_label
        self.yhat = pred_label
        self.N = true_label.shape[0]
        with Path(valid_group_mapping_file).open('rb') as pkl_file:
            self.valid_groups = pickle.load(pkl_file)
        
        self.valid_groups['true_label'] = self.valid_groups['target_label']
        assert np.absolute(self.valid_groups['true_label'] - true_label).sum() == 0, (
            'true_label does not match with target_label in valid_groups file')
        self.valid_groups['pred_label'] = pred_label
        
    def report_by_image(self, verbose=1, plot_cm=True):
        result = report_metrics(self.y, self.yhat,
                                plot_cm=plot_cm, verbose=verbose)
        return result
    
    def report_by_study(self, agg='mean', verbose=1, plot_cm=True):
        # agg can be mean or max
        agg_func = getattr(np, agg)
        valid_grouped = (self.valid_groups.groupby(
            ['body_part', 'patient_id', 'study_id'])[['true_label', 'pred_label']]
            .agg([agg_func]).round(0)
        )
        result = report_metrics(
            valid_grouped.values[:, 0], valid_grouped.values[:, 1],
            plot_cm=plot_cm, verbose=verbose)
        return result

    def report_by_body_parts(self, agg='mean', verbose=1, plot_cm=True):
        agg_func = getattr(np, agg)
        results = {}
        for group, df in self.valid_groups.groupby('body_part'):
            valid_grouped = (df.groupby(
                ['patient_id', 'study_id'])[['true_label', 'pred_label']]
                .agg([agg_func]).round(0)
            )
            log.info('='*50)
            log.info(f'Report for {group}')
            log.info('='*50)
            results.update(
                {group: report_metrics(
                    valid_grouped.values[:, 0], 
                    valid_grouped.values[:, 1], 
                    plot_cm=plot_cm, verbose=verbose)})
        
        return results
