import time

from modules.clients.ClientBase import ClientBase
from utils import metric
from utils.util import save_time_to_file


class FLClientFedAvg(ClientBase):
    def __init__(self, id: int, args, train_loader, total_round, client_net, log_dir, local_epoch: int = 1):
        super(FLClientFedAvg, self).__init__(id=id, total_round=total_round, log_dir=log_dir)
        self.train_loader = train_loader
        self.args = args
        self.lr = args.lr
        self.local_epoch = local_epoch
        self.client_net = client_net
        self.sample_num = len(train_loader.dataset)
        self.train_dict = {}

        self.optimizer = self.get_optimizer(optimizer_name=self.args.optimizer, model=self.client_net, lr=self.lr,
                                            momentum=self.args.momentum, weight_decay=self.args.weight_decay,
                                            is_nesterov=self.args.is_nesterov)

        self.scheduler_client = self.get_scheduler(scheduler_name=self.args.scheduler, optimizer=self.optimizer,
                                                   total_round=self.total_round, end_lr=self.args.end_lr)

    def train(self, current_round):
        self.client_net.train()
        loss_list = []
        acc_list = []

        start_time = time.time()
        for batch_idx, (image, target) in enumerate(self.train_loader):
            image, target = image.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.client_net(image)
            ce_loss = self.criterion(outputs, target)

            ce_loss.backward()
            self.optimizer.step()

            acc1 = metric.accuracy(outputs, target)
            loss_list.append(ce_loss.item())
            acc_list.append(acc1)

        end_time = time.time()

        client_time = end_time - start_time

        if self.scheduler_client is not None:
            self.scheduler_client.step()

        train_loss = sum(loss_list) / len(loss_list)
        train_acc = sum(acc_list) / len(acc_list)

        self.train_dict[current_round] = {
            "total_time": round(client_time, 3)
        }

        print(f'[Epoch: {current_round + 1}/{self.total_round}] [client: {self.id}] '
              f'[tt_loss: {train_loss:.3f}] [train_acc: {train_acc:.2f}]')

        return train_loss, train_acc

    def save_train_result(self):
        save_time_to_file(file_name=self.log_dir, client_id=self.id, train_dict=self.train_dict)
