import time

from modules.clients.ClientBase import ClientBase
from utils import metric
from utils.util import save_time_to_file


class SFLClientFedAvg(ClientBase):
    def __init__(self, id: int, args, train_loader, train_layer: int, log_dir, total_round, client_net=None,
                 server_net=None, local_epoch=1):
        super(SFLClientFedAvg, self).__init__(id=id, total_round=total_round, log_dir=log_dir)
        self.args = args
        self.lr = args.lr
        self.train_loader = train_loader
        self.client_net = client_net
        self.server_net = server_net
        self.local_epoch = local_epoch
        self.sample_num = len(train_loader.dataset)
        self.train_layer = train_layer
        self.train_dict = {}

        self.optimizer_client = self.get_optimizer(optimizer_name=self.args.optimizer, model=client_net, lr=self.lr,
                                                   momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                                                   is_nesterov=self.args.is_nesterov)

        self.optimizer_server = self.get_optimizer(optimizer_name=self.args.optimizer, model=server_net, lr=self.lr,
                                                   momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                                                   is_nesterov=self.args.is_nesterov)

        self.scheduler_client = self.get_scheduler(scheduler_name=self.args.scheduler, optimizer=self.optimizer_client,
                                                   total_round=self.total_round, end_lr=self.args.end_lr)

        self.scheduler_server = self.get_scheduler(scheduler_name=self.args.scheduler, optimizer=self.optimizer_server,
                                                   total_round=self.total_round, end_lr=self.args.end_lr)

    def train(self, current_round):
        self.client_net.train()

        loss_list = []
        acc_list = []
        client_time = 0.0
        server_time = 0.0

        client_start_time = time.time()
        for batch_idx, (image, target) in enumerate(self.train_loader):
            image, target = image.to(self.device), target.to(self.device)

            fx = self.client_net(image)
            client_fx = fx.clone().detach().requires_grad_(True)

            client_time += time.time() - client_start_time

            server_start_time = time.time()
            ce_loss, client_dfx, acc1 = self.server_train(client_fx=client_fx, target=target)
            server_time += time.time() - server_start_time

            client_start_time = time.time()
            fx.backward(client_dfx)
            self.optimizer_client.step()
            self.optimizer_client.zero_grad()

            loss_list.append(ce_loss.item())
            acc_list.append(acc1)

            client_time += time.time() - client_start_time
            client_start_time = time.time()

        if self.scheduler_client is not None:
            self.scheduler_client.step()
        if self.scheduler_server is not None:
            self.scheduler_server.step()

        train_loss = sum(loss_list) / len(loss_list)
        train_acc = sum(acc_list) / len(acc_list)

        self.train_dict[current_round] = {
            "client_time": round(client_time, 3),
            "server_time": round(server_time, 3),
            "total_time": round(client_time + server_time, 3)
        }

        print(f'[Epoch: {current_round + 1}/{self.total_round}] [client: {self.id}] '
              f'[tt_loss: {train_loss:.3f}] [train_acc: {train_acc:.2f}]')

        return train_loss, train_acc

    def server_train(self, client_fx, target):
        self.server_net.train()
        self.optimizer_server.zero_grad()

        outputs = self.server_net(client_fx)
        ce_loss = self.criterion(outputs, target)

        ce_loss.backward()
        self.optimizer_server.step()
        client_dfx = client_fx.grad.clone().detach()

        acc1 = metric.accuracy(outputs, target)

        return ce_loss, client_dfx, acc1

    def save_train_result(self):
        save_time_to_file(file_name=self.log_dir, client_id=self.id, train_dict=self.train_dict)
