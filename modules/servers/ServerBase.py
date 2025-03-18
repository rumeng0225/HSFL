from torch import nn, optim

from config import Config


class ServerBase():
    def __init__(self, log_dir: str, log_file, total_round: int):
        super(ServerBase).__init__()
        self.device = Config.device()
        self.criterion = nn.CrossEntropyLoss()
        self.total_round = total_round
        self.log_dir = log_dir
        self.log_file = log_file

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
