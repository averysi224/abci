import os
import logging

import torch
from torch.utils.data.dataloader import default_collate


def setup_logging(log_file='log.txt', resume=True, dummy=False):
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file,
            filemode=file_mode,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def handle_error(message):
    logging.error(message)
    raise Exception(message)


class SamplesGlobalCollator(object):
    def __init__(self, K=None):
        self.K = K

    def __call__(self, batch):
        batch = default_collate(batch)
        shape = batch.shape
        batch = batch.view(shape[0]//self.K, self.K, *shape[1:])
        return batch


class SamplesPatchesCollator(object):
    def __init__(self, K=None):
        self.K = K

    def __call__(self, batch):
        batch = default_collate(batch)
        shape = batch.shape
        batch = batch.reshape(shape[0]//self.K, self.K, shape[1], *shape[2:])
        batch = batch.transpose(1,2)
        batch = batch.reshape(shape[0] * shape[1], *shape[2:])
        shape = batch.shape
        batch = batch.view(shape[0]//self.K, self.K, *shape[1:])
        return batch


class GroundTruthsPatchesCollator(object):
    def __call__(self, batch):
        batch = default_collate(batch)
        shape = batch.shape
        batch = batch.reshape(shape[0] * shape[1], *shape[2:])
        return batch
