import copy
import json

import torch

from dataset import factory
from modules.clients.FLClientFedProx import FLClientFedProx
from modules.servers.ServerBase import ServerBase
from utils import metric, summary
from utils.client_sample import select_clients
from utils.logger import save_hyperparameter
from utils.model_utils import generate_model
from utils.util import save_checkpoint


class FLServerFedProx(ServerBase):
    def __init__(self, args, log_file, log_dir: str):
        super(FLServerFedProx, self).__init__(log_dir=log_dir, log_file=log_file, total_round=args.total_round)
        self.args = args
        self.num_clients: int = args.num_clients
        self.num_active_clients: int = int(args.num_clients * args.frac)
        self.server_model = None
        self.client_idx = None
        self.client_models = None
        self.clients = []
        self.val_loader = None

        self.init_clients()

    def init_clients(self):
        batch_sizes = self.args.batch_size * self.num_clients
        self.args.batch_size = batch_sizes

        save_hyperparameter(self.log_dir, self.args)

        train_loader, _, val_loader = factory.get_data_loader(self.args.data,
                                                              batch_sizes=batch_sizes,
                                                              eval_batch_size=self.args.eval_batch_size,
                                                              dataset=self.args.dataset,
                                                              num_workers=self.args.workers,
                                                              num_clients=self.args.num_clients,
                                                              dirichlet=self.args.dirichlet,
                                                              seed=self.args.seed,
                                                              noniid=self.args.sampler)

        self.val_loader = val_loader

        client_models = [generate_model(model_name=self.args.arch, args=self.args, device=self.device)
                         for _ in range(self.num_clients)]
        self.client_models = client_models

        self.server_model = generate_model(model_name=self.args.arch, args=self.args, device=self.device)

        # print(client_models[0])

        for model in client_models:
            model.load_state_dict(copy.deepcopy(self.server_model.state_dict()))

        model_parameter_list = []
        for index, model in enumerate(client_models):
            model_parameter_list.append(metric.get_the_number_of_params(model))

        self.log_file.info(
            f"INFO: The number of parameters in the client-side model is {model_parameter_list}")

        summary.save_model_to_json(self.args, client_models, mode='w')

        self.clients = [FLClientFedProx(id=i,
                                        args=self.args,
                                        train_loader=train_loader[i],
                                        client_net=client_models[i],
                                        log_dir=self.log_dir,
                                        total_round=self.total_round)
                        for i in range(self.num_clients)]

    def train(self):
        best_acc1 = 0.0
        result_dict = {}

        for r in range(self.total_round):
            print(f"======================= Round {r + 1} =======================")
            self.client_idx = select_clients(num_total=self.num_clients, num_active=self.num_active_clients)

            client_loss_list = []
            for index, client_id in enumerate(self.client_idx):
                current_client = self.clients[client_id]
                if r != 0:
                    current_client.update(net_params=copy.deepcopy(self.server_model.state_dict()))
                client_loss, acc = current_client.train(current_round=r)
                client_loss_list.append(client_loss)

            client_params = [self.clients[index].client_net.state_dict() for index in self.client_idx]
            client_samples = [self.clients[index].sample_num for index in self.client_idx]
            client_weights = [sample / sum(client_samples) for sample in client_samples]

            self.aggregate(client_params, client_weights)

            test_loss, test_acc = self.validate()
            print(f'[Epoch: {r + 1}/{self.total_round}] [test_loss: {test_loss:.3f}] '
                  f'[test_acc: {test_acc:.2f}] ')

            aver_train_loss = sum([client_loss_list[i] * client_weights[i] for i in range(self.num_active_clients)])
            self.log_file.info(
                f'{r}-th round | train loss: {aver_train_loss:.3f} | test loss: {test_loss:.3f} | '
                f'test acc: {test_acc:.2f}')

            is_best = test_acc > best_acc1
            best_acc1 = max(test_acc, best_acc1)

            ckpt_dict_model = {
                'round': r,
                'arch': self.args.arch,
                'state_dict': self.server_model.state_dict(),
                'best_acc1': best_acc1,
            }
            save_checkpoint(ckpt_dict_model, is_best, self.args.model_dir)

            result_file_name = f'{self.log_dir}/result.json'
            result_dict[r] = {
                "train_loss": round(aver_train_loss, 3),
                "test_loss": round(test_loss, 3),
                "test_acc": round(test_acc, 2)
            }
            if (r + 1) % 5 == 0 or (r + 1) == self.total_round:
                with open(result_file_name, 'w') as f:
                    json.dump(result_dict, f, indent=4)

                for client in self.clients:
                    client.save_train_result()

    def aggregate(self, client_parameters, client_weights):
        g_params = copy.deepcopy(client_parameters[0])
        w = client_parameters
        for key in g_params.keys():
            g_params[key] = torch.zeros(g_params[key].shape).to(self.args.device)
            if 'num_batches_tracked' in key:
                g_params[key] = w[0][key]
                continue

            for model, weight in zip(w, client_weights):
                g_params[key] += model[key] * weight

        self.server_model.load_state_dict(g_params)

    def validate(self):
        client_idx = self.client_idx[0]
        client = self.clients[client_idx]
        net = self.server_model
        loss, acc = client.validate(net=net, val_loader=self.val_loader)
        return loss, acc
