import numpy as np


def select_clients(num_total, num_active):
    if num_total == num_active:
        select_client_idx = [i for i in range(num_total)]
    else:
        select_client_idx = list(np.random.choice(range(num_total), num_active, replace=False))
        print('INFO: Participants list:', select_client_idx, flush=True)
    return select_client_idx
