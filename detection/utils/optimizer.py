from torch import nn, optim
from . import LRScheduler


class HyperOptimizer:
    def __init__(self, model, warmup_epochs, warmup_lr, max_epoch, no_aug_epochs, basic_lr_per_img, batch_size,
                 momentum, scheduler, min_lr_ratio, weight_decay):

        if warmup_epochs > 0:
            lr = warmup_lr
        else:
            lr = basic_lr_per_img * batch_size

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer = optim.SGD(
            pg0, lr=lr, momentum=momentum, nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": weight_decay}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.max_epoch = max_epoch
        self.no_aug_epochs = no_aug_epochs
        self.min_lr_ratio = min_lr_ratio

    def get_lr_scheduler(self, lr, iters_per_epoch):
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def setp(self):
        self.optimizer.step()
