import argparse
import os

import setproctitle
import torch
from torch.backends import cudnn

import train_params
from modules.servers.FLServerFedAvg import FLServerFedAvg
from modules.servers.FLServerFedProx import FLServerFedProx
from modules.servers.SFLServerFedAvg import SFLServerFedAvg
from utils.logger import Gen_Log_Dir, Get_Logger
from utils.set_random_seed import seed_everything

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser(description='Heterogeneous SFL')
    args = train_params.add_parser_params(parser)
    seed_everything(args.seed)

    log_dir, _ = Gen_Log_Dir(args)

    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    args.model_dir = log_dir + '/model'
    setproctitle.setproctitle(args.algo_name)

    print(f"INFO: Training Algorithm: {args.algo_name}")
    if args.algo_name == "FedAvg":
        assert not args.arch.endswith('sl')
        FedAvg = FLServerFedAvg(args, log_file, log_dir)
        FedAvg.train()
    elif args.algo_name == "FedAvg_SFL":
        assert args.num_clients * args.frac == args.num_clients, "num_selected must === num_clients Now"
        assert args.arch.endswith('sl')
        FedAvg = SFLServerFedAvg(args, log_file, log_dir)
        FedAvg.train()
    elif args.algo_name == "FedProx":
        assert not args.arch.endswith('sl')
        FedProx = FLServerFedProx(args, log_file, log_dir)
        FedProx.train()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
