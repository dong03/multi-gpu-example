from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=20, power=0.8, last_epoch=-1,cycle=True):
        self.max_iter = max_iter
        self.power = power
        self.cycle = cycle
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch_div = (self.last_epoch + 1) % self.max_iter
        scale = (self.last_epoch + 1) // self.max_iter + 1.0 if self.cycle else 1
        return [(base_lr * ((1 - float(self.last_epoch_div) / self.max_iter) ** (self.power))) / scale for base_lr in self.base_lrs]

