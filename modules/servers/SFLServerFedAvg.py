import copy
import json

import torch

from dataset import factory
from modules.clients.SFLClientFedAvg import SFLClientFedAvg
from modules.servers.ServerBase import ServerBase
from utils import metric, summary
from utils.client_sample import select_clients
from utils.logger import save_hyperparameter
from utils.model_utils import generate_model
from utils.util import save_checkpoint


class SFLServerFedAvg(ServerBase):
    def __init__(self, args, log_file, log_dir):
        super(SFLServerFedAvg, self).__init__(log_dir=log_dir, log_file=log_file, total_round=args.total_round)
        self.args = args
        self.num_clients = args.num_clients
        self.num_active_clients = args.num_clients
        self.server_models = None
        self.global_model = None
        self.client_idx = None
        self.clients = None
        self.val_loader = None

        self.init_clients()

    def init_clients(self):
        batch_sizes = self.args.batch_size * self.num_clients
        train_layer = [2] * self.num_clients

        self.args.train_layer = train_layer
        self.args.batch_size = batch_sizes
        save_hyperparameter(self.log_dir, self.args)

        train_loader, _, val_loader = factory.get_data_loader(self.args.data,
                                                              batch_sizes=batch_sizes,
                                                              dataset=self.args.dataset,
                                                              num_workers=self.args.workers,
                                                              num_clients=self.args.num_clients,
                                                              dirichlet=self.args.dirichlet,
                                                              seed=self.args.seed,
                                                              noniid=self.args.sampler)

        self.val_loader = val_loader

        global_model = generate_model(model_name=self.args.arch, args=self.args, model_type="global",
                                      device=self.device)
        # print(global_model)

        client_models = [generate_model(model_name=self.args.arch, args=self.args, train_layer=train_layer[i],
                                        model_type="client", device=self.device)
                         for i in range(self.num_active_clients)]

        self.global_model = global_model

        server_models = [generate_model(model_name=self.args.arch, args=self.args, train_layer=train_layer[i],
                                        model_type="server", device=self.device) for i in
                         range(self.num_active_clients)]

        server_model_params = copy.deepcopy(global_model.state_dict())

        self.distribute(server_models, server_model_params)

        self.distribute(client_models, server_model_params)
        self.server_models = server_models

        model_parameter_list = []
        for index, model in enumerate(client_models):
            model_parameter_list.append(metric.get_the_number_of_params(model))
        self.log_file.info(
            f"INFO: The number of parameters in the client-side model is {model_parameter_list}")

        summary.save_model_to_json(self.args, client_models, server_models, mode='w')

        self.clients = [SFLClientFedAvg(id=i, args=self.args,
                                        train_loader=train_loader[i],
                                        client_net=client_models[i],
                                        server_net=server_models[i],
                                        total_round=self.total_round,
                                        log_dir=self.log_dir,
                                        train_layer=train_layer[i])
                        for i in range(self.num_clients)]

    def train(self):
        best_acc1 = 0.0
        result_dict = {}
        for r in range(self.total_round):
            print(f"======================= Round {r + 1} =======================")
            self.client_idx = select_clients(num_total=self.num_clients, num_active=self.num_active_clients)
            client_loss_list = []
            for n in range(0, self.num_active_clients):
                current_client = self.clients[self.client_idx[n]]
                client_loss, acc = current_client.train(current_round=r)
                client_loss_list.append(client_loss)

            if (r + 1) % self.args.local_epoch == 0 or r == self.total_round - 1:
                client_params = [self.clients[n].client_net.state_dict() for n in self.client_idx]
                client_models = [self.clients[n].client_net for n in self.client_idx]
                server_models = [self.clients[n].server_net for n in self.client_idx]

                client_samples = [self.clients[n].sample_num for n in self.client_idx]
                client_weights = [client_samples[n] / sum(client_samples) for n in range(self.num_active_clients)]

                server_params = [model.state_dict() for model in self.server_models]
                self.aggregate(client_params, server_params, client_weights)
                print("INFO: Aggregate the model parameters")

                global_model_params = self.global_model.state_dict()

                self.distribute(server_models, global_model_params)

                self.distribute(client_models, global_model_params)

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
                    'state_dict': self.global_model.state_dict(),
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

    def aggregate(self, client_params, server_params, client_weights):
        new_global_model = copy.deepcopy(self.global_model.state_dict())

        for k, v in new_global_model.items():
            new_global_model[k] = torch.zeros(v.shape).to(self.args.device)

        params_list = []
        for i in range(len(client_params)):
            params = {}
            params.update(client_params[i])
            params.update(server_params[i])
            params_list.append(params)

        for weight, client_model in zip(client_weights, params_list):
            for k, v in client_model.items():
                new_global_model[k] += weight * v

        self.global_model.load_state_dict(new_global_model)

    def distribute(self, models, global_params):
        for model in models:
            model_dict = model.state_dict()
            exhibit_keys = model_dict.keys()
            for key in exhibit_keys:
                model_dict[key] = copy.deepcopy(global_params[key])
            model.load_state_dict(model_dict)

    def validate(self):
        loss, acc = self.clients[-1].validate(self.global_model, self.val_loader)
        return loss, acc
