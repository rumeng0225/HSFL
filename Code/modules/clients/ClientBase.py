import torch
from torch import nn, optim

from config import Config
from utils import metric


class ClientBase():
    def __init__(self, id: int, total_round: int, log_dir):
        super(ClientBase).__init__()
        self.id = id
        self.device = Config.device()
        self.criterion = nn.CrossEntropyLoss()
        self.total_round = total_round
        self.log_dir = log_dir
        self.client_net = None

    def validate(self, net, val_loader):
        loss_list = []
        acc_list = []

        net.eval()

        with torch.no_grad():
            for index, (images, target) in enumerate(val_loader):
                images, target = images.to(self.device), target.to(self.device)

                outputs = net(images)

                loss = self.criterion(outputs, target)
                acc1 = metric.accuracy(outputs, target)

                loss_list.append(loss.item())
                acc_list.append(acc1)

        test_loss = sum(loss_list) / len(loss_list)
        acc = sum(acc_list) / len(acc_list)
        return test_loss, acc

    def get_optimizer(self, optimizer_name, model, lr, momentum, weight_decay, is_nesterov):
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True if is_nesterov else False)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay)
        else:
            raise ValueError(f"No such optimizer: {optimizer_name}")

        return optimizer

    def get_scheduler(self, scheduler_name, optimizer, total_round, end_lr):
        if scheduler_name == 'None':
            return None
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_round, eta_min=end_lr)
        else:
            raise ValueError(f"No such scheduler: {scheduler_name}")
        return scheduler

    def update(self, net_params):
        self.client_net.load_state_dict(net_params)
