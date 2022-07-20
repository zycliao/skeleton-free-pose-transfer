import numpy as np
import torch


def get_geo_index(rows, cols, vals, num_nodes):
    m = np.ones([num_nodes, num_nodes], dtype=np.float32) * 10
    m[rows, cols] = vals
    return torch.from_numpy(m)