import torch
import numpy as np


def torch2np(*x):
    ret = []
    for xx in x:
        ret.append(xx.detach().cpu().numpy())
    return ret

def np2torch(*x, dtype=torch.float, dev=torch.device('cpu')):
    ret = []
    for xx in x:
        ret.append(torch.from_numpy(xx).to(dtype).to(dev))
    return ret