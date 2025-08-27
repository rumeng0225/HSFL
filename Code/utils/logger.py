import os
import logging

import json
import time

from config import log_path


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def _Get_Log_Num(log_count_file):
    if os.path.exists(log_count_file):
        with open(log_count_file, 'r') as f:
            file_content = f.readlines()
        log_num = int(file_content[-1]) + 1
        with open(log_count_file, 'a') as f:
            f.writelines(str(log_num) + '\n')
    else:
        with open(log_count_file, 'w') as f:
            f.writelines('0\n')
        log_num = 0
    return log_num


def _Get_Log_Name(args, log_num, file_name='train'):
    start_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    log_name = (f"{log_num:d}-{start_time}-{file_name}-{args.dataset if 'dataset' in args else ''}"
                f"-{args.arch if 'arch' in args else ''}-{args.sampler if args.sampler != 'iid' else 'iid'}")
    return log_name


def Get_Logger(file_name, file_save=True, display=True):
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if file_save:
        if os.path.isfile(file_name):
            fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        else:
            fh = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def save_hyperparameter(log_dir, args):
    os.makedirs(log_dir, exist_ok=True)

    if not args.evaluate:
        filename = os.path.join(log_dir, 'hparams_train.json')
    else:
        filename = os.path.join(log_dir, 'hparams_eval.json')
    hparams = vars(args)
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)


def Gen_Log_Dir(args, file_name='train', tensorboard_subdir=True):
    log_count_file = log_path + 'log_count.txt'

    if os.path.exists(log_path) is False:
        mkdirs(log_path)

    log_num = _Get_Log_Num(log_count_file)
    log_name = _Get_Log_Name(args=args, log_num=log_num, file_name=file_name)
    log_dir = log_path + log_name + '/'

    mkdirs(log_dir)

    if tensorboard_subdir:
        tensorboard_dir = log_dir + '/tensorboard/'
        return log_dir, tensorboard_dir
    else:
        return log_dir
