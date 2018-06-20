from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from PIL import Image, ImageOps
import pickle
import gc
import logging
from utils.utils import *

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def read_pickle_mura(mura_path, sample, target_size=512):
    '''
    Reads mura datasets, pads with 0's to a square image,
    resizes each edge to be target_size, and then saves
    the datasets as pickle files
    
    Parameters
    ----------
    mura_path: str
        string path of MURA-v1.1
    sample: str
        switch for running for train or valid sample
    target_size: int
        target edge length for resizing
        
    Returns
    -------
    None
    
    '''
    assert sample in ('train', 'valid')
    log.info(f'reading {sample} image paths')
    paths = pd.read_csv(
        mura_path.joinpath(f'{sample}_image_paths.csv'), header=None)
    paths.columns=['path']
    i = 0
    img_list = []
    label_list = []
    target_size = 512
    with progressbar.ProgressBar(max_value=paths.shape[0]) as bar:
        for image_file in paths['path']:
            label = 1 if 'positive' in image_file else 0
            img = Image.open(
                mura_path.parent.joinpath(image_file).as_posix()).convert('L')
            delta_w = 512 - img.width
            delta_h = 512 - img.height
            padding = (delta_w//2, delta_h//2, 
                       delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_img = ImageOps.expand(img, padding)

            img_list.append(np.asarray(new_img.resize(
                (target_size, target_size), Image.ANTIALIAS)))
            label_list.append(label)
            i += 1
            bar.update(i)
            
    log.info('stacking images')
    x = np.stack(img_list, axis=0)
    # forcing garbage collection to save memory
    del img_list; gc.collect()
    y = np.stack(label_list, axis=0)
    del label_list; gc.collect()
    
    log.info('picking images')
    write_pickle_file(x, mura_path.joinpath(f'x_{sample}.pkl'))
    del x; gc.collect()
    write_pickle_file(y, mura_path.joinpath(f'y_{sample}.pkl'))
    del y; gc.collect()
    # with mura_path.joinpath(f'x_{sample}.pkl').open('wb') as pickle_file:
    #     pickle.dump(x, pickle_file, protocol=4)
    # del x; gc.collect()
    # with mura_path.joinpath(f'y_{sample}.pkl').open('wb') as pickle_file:
    #     pickle.dump(y, pickle_file, protocol=4)
    # del y; gc.collect()


def read_mura_pickle(path='data/MURA-v1.1', sample=None):
    '''
    Get MURA data

    Parameters
    ----------
    path: str
        path to pickled MURA data
    sample: str
        if None, then read both train and valid sample and
        return a tuple of 4 elements
        otherwise, return a tuple of two elements corresponding
        to the x and y datasets for the sample.

    Return
    ------
    `obj`:tuple of `obj`:numpy.ndarray
    '''
    mura_path = Path(path)
    if not sample or sample == 'train':
        train_X = read_pickle_file(mura_path.joinpath(f'x_train.pkl'))
        train_Y = read_pickle_file(mura_path.joinpath(f'y_train.pkl'))
        if sample == 'train':
            return train_X, train_Y
    if not sample or sample == 'valid':
        test_X = read_pickle_file(mura_path.joinpath(f'x_valid.pkl'))
        test_Y = read_pickle_file(mura_path.joinpath(f'y_valid.pkl'))
        if sample == 'valid':
            return test_X, test_Y
    return train_X, train_Y, test_X, test_Y
