import os

import torch

# Home directory
HOME = os.getcwd()

log_path = str(HOME) + "/log/"
# data directory
data_dir = HOME

# Model directory
# CHECKPOINT = data_dir + '/checkpoint'

TINY_IMAGENET_PATH = r'./data/tiny-imagenet-200/'


class Config:

    @staticmethod
    def device():
        device = "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device('cuda')
        return device
