import pickle
from pathlib import Path
import logging

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