from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from PIL import Image, ImageOps
import pickle
import gc
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def read_pickle_mura(mura_path, sample, target_size=512):
    assert sample in ('train', 'valid')
    log.info(f'reading {sample} image paths')
    paths = pd.read_csv(mura_path.joinpath(f'{sample}_image_paths.csv'), header=None)
    paths.columns=['path']
    i = 0
    img_list = []
    label_list = []
    target_size = 512
    with progressbar.ProgressBar(max_value=paths.shape[0]) as bar:
        for file in paths['path']:
            label = 1 if 'positive' in file else 0
            img = Image.open(mura_path.parent.joinpath(file).as_posix()).convert('L')
            delta_w = 512 - img.width
            delta_h = 512 - img.height
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_img = ImageOps.expand(img, padding)

            img_list.append(np.asarray(new_img.resize((target_size, target_size), Image.ANTIALIAS)))
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
    with mura_path.joinpath(f'x_{sample}.pkl').open('wb') as file:
        pickle.dump(x, file, protocol=4)
    del x; gc.collect()
    with mura_path.joinpath(f'y_{sample}.pkl').open('wb') as file:
        pickle.dump(y, file, protocol=4)
    del y; gc.collect()
    