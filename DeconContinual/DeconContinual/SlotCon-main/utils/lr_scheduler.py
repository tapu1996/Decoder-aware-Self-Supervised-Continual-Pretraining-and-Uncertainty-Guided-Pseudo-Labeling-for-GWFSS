# noinspection PyProtectedMember
import math
import warnings
from typing import (
    cast,
    List,
)
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LRScheduler
from torch import Tensor


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warmup_steps, last_epoch=-1):
        self.multiplier = multiplier
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr/ self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_steps + 1.) for base_lr in self.base_lrs]
    

class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs, total_steps, multiplier, n_iter_per_epoch, last_epoch=-1):
        """
        Args:
            optimizer: optimizer
            total_epochs: total number of epochs
            warmup_epochs: number of warmup epochs
            total_steps: total number of steps
            multiplier: multiplier for warmup
            n_iter_per_epoch: number of iterations per epoch
            last_epoch: last epoch
        
        This class is a wrapper around the CosineAnnealingLR scheduler and the LinearWarmupScheduler.
        It allows for a warmup phase followed by a cosine decay phase.
        It is here to avoid the warning of the slotcon scheduler that needs a "step" argument in the step function.
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * n_iter_per_epoch
        self.total_steps = total_epochs * n_iter_per_epoch
        self.multiplier = multiplier
        self.eta_min = 0.000001
        self.T_max = total_steps
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.last_step = 0
        self.optimizer = optimizer
        self.linear_scheduler = LinearWarmupScheduler(optimizer, multiplier, self.warmup_steps)
        self.cosine_scheduler =  CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=self.eta_min,
            T_max=self.total_steps, 
            last_epoch=last_epoch
            )

    def get_lr(self):
        if self.last_step <= self.warmup_steps:
            # Linear warmup
            lr = self.linear_scheduler.get_lr()
        else:
            # Cosine decay
            lr = self.cosine_scheduler.get_lr()
        return lr
            
    def step(self):
        self.last_step += 1
        if self.last_step > self.warmup_steps:
            self.cosine_scheduler.step()
        else:
            self.linear_scheduler.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key not in['optimizer', 'linear_scheduler','cosine_scheduler']}
        state['linear_scheduler'] = self.linear_scheduler.state_dict()
        state['cosine_scheduler'] = self.cosine_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        linear_scheduler_state = state_dict.pop('linear_scheduler')
        cosine_scheduler_state = state_dict.pop('cosine_scheduler')
        self.__dict__.update(state_dict)
        self.linear_scheduler.load_state_dict(linear_scheduler_state)
        self.cosine_scheduler.load_state_dict(cosine_scheduler_state)
        self.resume_step_cosine()

    def resume_step_cosine(self, ):
        if hasattr(self.cosine_scheduler, "_get_closed_form_lr"):
            values = cast(List[float], self.cosine_scheduler._get_closed_form_lr())
        else:
            values = self.cosine_scheduler.get_lr()
        for i, data in enumerate(zip(self.cosine_scheduler.optimizer.param_groups, values)):
            param_group, lr = data
            if isinstance(param_group["lr"], Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr

        self.cosine_scheduler._last_lr: List[float] = [
            group["lr"] for group in self.cosine_scheduler.optimizer.param_groups
        ]

def get_scheduler(optimizer, n_iter_per_epoch, args):
    if args.previous_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)

        if args.warmup_epoch > 0:
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=args.warmup_multiplier,
                after_scheduler=scheduler,
                warmup_epoch=args.warmup_epoch * n_iter_per_epoch)
        return scheduler
    else:
        return WarmupCosineDecayScheduler(
            optimizer=optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epoch,
            total_steps=(args.epochs - args.warmup_epoch) * n_iter_per_epoch,
            multiplier=args.warmup_multiplier,
            n_iter_per_epoch=n_iter_per_epoch)

def get_scheduler2(optimizer, n_iter_per_epoch, args):
    return WarmupCosineDecayScheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epoch,
        total_steps=(args.epochs - args.warmup_epoch) * n_iter_per_epoch,
        multiplier=args.warmup_multiplier,
        n_iter_per_epoch=n_iter_per_epoch)

if __name__ == "__main__":

    # testing lr scheduler
    import os 
    import copy
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parents[1]))
    from main_pretrain import get_parser

    
    def save(state_dict, file_name):
        state = {

            'scheduler': state_dict,
        }

        file_name = os.path.join("/home/sebquet/VisionResearchLab/DenseSSL/DenseSSL/output", file_name)
        torch.save(state, file_name)
        
    args = get_parser()
    args.world_size = 1 
    model = torch.nn.Linear(3, 3)
    model2 = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    opt2 = torch.optim.SGD(
            model2.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    niter_per_epoch = 5

    scheduler = get_scheduler(opt, niter_per_epoch, args)
    scheduler2 = get_scheduler2(opt2, niter_per_epoch, args)
    lr_values = []
    lr_values2 = []

    for epoch in range(args.epochs):
        for i in range(niter_per_epoch):
            if epoch == 189 and i == niter_per_epoch-1:
                save(scheduler.state_dict(), "ckpt_scheduler.pth")
                save(scheduler2.state_dict(), "ckpt_scheduler2.pth")
                ckpt_scheduler = copy.deepcopy(scheduler.state_dict())
                ckpt_scheduler2 = copy.deepcopy(scheduler2.state_dict())
                opt_ckpt = opt.state_dict()
                opt2_ckpt = opt2.state_dict()
            scheduler.step()
            scheduler2.step()
            lr_values.append(scheduler.get_lr()[0])
            lr_values2.append(scheduler2.get_lr()[0])
            assert np.isclose(scheduler.get_lr()[0], scheduler2.get_lr()[0], atol=1e-7), f"{scheduler.get_lr()[0]} != {scheduler2.get_lr()[0]} at step {i} of epoch {epoch}"
    # plt.plot(lr_values, 'r+', label="old")
    # plt.plot(lr_values2, 'b*', label="new")
    # plt.title("old vs new lr scheduler from start of training")
    # plt.legend()
    # plt.show()
    print("scheduler.state_dict()", scheduler.state_dict())
    print(ckpt_scheduler)
    print("scheduler2.state_dict()", scheduler2.state_dict())
    print(ckpt_scheduler2)
    # exit(0)
    print("===============================RESUMING FROM CHECKPOINT=====================================")
    opt = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    opt2 = torch.optim.SGD(
            model2.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    print("BEfore LOADING")
    print("opt2 state dict", opt2.state_dict())
    opt.load_state_dict(opt_ckpt)
    opt2.load_state_dict(opt2_ckpt)
    print("AFTER LOADING")
    print("opt2 state dict", opt2.state_dict())
    # exit(0)
    scheduler = get_scheduler(opt, niter_per_epoch, args)

    scheduler.load_state_dict(# ckpt_scheduler)
        torch.load("/home/sebquet/VisionResearchLab/DenseSSL/DenseSSL/output/ckpt_scheduler.pth")['scheduler'])
    scheduler2 = get_scheduler2(opt2, niter_per_epoch, args)
    scheduler2.load_state_dict(# ckpt_scheduler2)
        torch.load("/home/sebquet/VisionResearchLab/DenseSSL/DenseSSL/output/ckpt_scheduler2.pth")['scheduler'])

    lr_values = []
    lr_values2 = []
    step = 0
    for epoch in range(args.epochs):
        for i in range(niter_per_epoch):
            scheduler.step()
            scheduler2.step()
            print("step", step)
            step += 1
            lr =scheduler.get_lr()[0]
            lr2 = scheduler2.get_lr()[0]
            print("lr1 is", lr)
            print("lr2 is", lr2)
            lr_values.append(lr)
            lr_values2.append(lr2)
            assert np.isclose(scheduler.get_lr()[0], scheduler2.get_lr()[0], atol=1e-7), f"{scheduler.get_lr()[0]} != {scheduler2.get_lr()[0]} at step {i} of epoch {epoch}"
    plt.plot(lr_values, 'r+', label="old")
    plt.plot(lr_values2, 'b*', label="new")
    plt.title("old vs new lr scheduler from ckpt")
    plt.legend()
    plt.show()
