import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        print(f"INFO: Device: {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed_all(seed)
