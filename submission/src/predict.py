import keras
from keras.models import load_model
import re
import numpy as np
import pandas as pd
import logging
from PIL import Image, ImageOps
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_path(df):
    df['study_path'] = df['path'].apply(lambda x: re.split('image', x, flags=re.IGNORECASE)[0])
    return df

def normalize_pixels(x):
    x = x.astype('float32')
    x /= 255
    return x

def read_mura(path_file_csv, parent_folder, target_size=256):
    paths = pd.read_csv(path_file_csv, header=None)
    paths.columns = ['path']
    paths = parse_path(paths)
    img_list = []
    study_path_list = []
    for i in range(paths.shape[0]):
        image_file = paths.iloc[i]['path']
        img = Image.open(parent_folder + image_file).convert('L')
        delta_w = 512 - img.width
        delta_h = 512 - img.height
        padding = (delta_w//2, delta_h//2, 
                   delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_img = ImageOps.expand(img, padding)
        img_list.append(np.asarray(new_img.resize(
            (target_size, target_size), Image.ANTIALIAS)))
        study_path_list.append(paths.iloc[i]['study_path'])

    x = np.stack(img_list, axis=0)
    return x, pd.DataFrame({'study': study_path_list})

if __name__ == '__main__':
    path_file_csv = sys.argv[1]
    output_csv_path = sys.argv[2]
    try:
        model_name = sys.argv[3]
    except:
        model_name = '2_1_submodel_335'
    finally: 
        pass
    model_file = f'src/model/{model_name}.h5'
    try:
        parent_folder = sys.argv[4]
    except:
        parent_folder = './'
    finally:
        pass

    log.info(f'Path file: {path_file_csv}')
    log.info(f'Parent folder to data: {parent_folder}')
    log.info(f'Model file: {model_file}')
    log.info(f'Parent folder for input data: {parent_folder}')

    log.info('-'*20 + 'Loading Model ' + '-'*20)
    model = load_model(model_file)


    log.info('-'*20 + 'Reading Image Files' + '-'*20)
    x, studies = read_mura(path_file_csv, parent_folder)
    x = normalize_pixels(x)
    xs = x.shape
    log.info(f'{xs[0]} images in total from {studies.shape[0]} studies.')
    log.info('-'*20 + 'Making predictions' + '-'*20)
    y_hat = model.predict(np.reshape(x, (xs[0], xs[1], xs[2], 1)))
    studies['y'] = y_hat
    predict = studies.groupby('study')[['y']].mean().round(0).reset_index()

    log.info('-'*20 + f'Writing results to {output_csv_path}' + '-'*20)
    predict.to_csv(output_csv_path, header=False, index=False)
