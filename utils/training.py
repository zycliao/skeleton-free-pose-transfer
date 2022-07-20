import os, shutil
import numpy as np
import torch
import time
from datetime import datetime


class LossManager(object):
    def __init__(self, loss_names, writer=None, epoch=0):
        self.loss_names = loss_names
        self.epoch = epoch
        self.latest_loss = {}
        self.accum_loss = {}
        for name in self.loss_names:
            self.latest_loss[name] = 0.
            self.accum_loss[name] = 0.
        self.iter = 0
        self.writer = writer
        self.t = time.time()

    def update_epoch(self, epoch):
        self.epoch = epoch
        self.iter = 0
        for name in self.loss_names:
            self.accum_loss[name] = 0.
        self.t = time.time()

    def add_loss(self, iter=None, **kwargs):
        if iter:
            self.iter = iter + 1
        else:
            self.iter += 1
        for name, loss in kwargs.items():
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().item()
            self.latest_loss[name] = loss
            self.accum_loss[name] += loss

    def print_latest(self):
        info = f"Epoch {self.epoch}, iter {self.iter-1}.\t"
        for name in self.loss_names:
            info += f"{name}: {self.latest_loss[name]:.5f}\t"
        info += f"time: {time.time()-self.t:.2f}"
        self.t = time.time()
        print(info)

    def epoch_summary(self):
        if self.iter == 0:
            return
        if self.writer:
            for name in self.loss_names:
                self.writer.add_scalar(name, self.accum_loss[name] / self.iter, self.epoch)
        self.print_epoch()

    def print_epoch(self):
        info = f"Validation Epoch {self.epoch}.\t"
        for name in self.loss_names:
            info += f"{name}: {(self.accum_loss[name] / self.iter):.5f}\t"
        print(info)


def euc_distance(x1, x2):
    return torch.mean(torch.sqrt(torch.sum((x1 - x2)**2, -1)))


def handle_batch(handle_idx, device=None):
    """
    The default batch of torch-geometric data loader used here
    is for vertex instead of handle. So the batch index for handle
    must be computed here
    handle_idx: list. each element is numpy array with shape (1, num_handles)
    """
    batch = []
    ptr = [0]
    for i, hd in enumerate(handle_idx):
        batch.extend([i] * hd.shape[1])
        ptr.append(ptr[-1]+hd.shape[1])
    batch = torch.from_numpy(np.array(batch)).long()
    ptr = torch.from_numpy(np.array(ptr)).long()
    if device:
        batch = batch.to(device)
        ptr = ptr.to(device)
    return batch, ptr


def list_index(a, index):
    y = []
    for i in index:
        y.append(a[i])
    return y


def load_random_prefix():
    return datetime.now().strftime('%M.%S.%f')


def backup_file(src, dst):
    if len([k for k in os.listdir(src) if k.endswith('.py') or k.endswith('.sh')]) == 0:
        return
    os.makedirs(dst, exist_ok=True)
    all_files = os.listdir(src)
    for fname in all_files:
        fname_full = os.path.join(src, fname)
        fname_dst = os.path.join(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif fname.endswith('.py') or fname.endswith('.sh'):
            shutil.copy(fname_full, fname_dst)
