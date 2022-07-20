import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.lbs import lbs
from scipy.sparse import csr_matrix
from utils.geometry import get_normal
from data_utils.common import get_geo_index
from utils.general import np2torch


class RignetDataset(Dataset):
    def __init__(self, data_dir, flag='train', preload=True, shuffle_handle=True):
        super(RignetDataset, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        self.shuffle_handle = shuffle_handle
        with open(os.path.join(self.data_dir, flag+'.txt')) as f:
            self.names = f.read().splitlines()

        self.vs, self.fs = [], []
        self.joints, self.parents, self.weights = [], [], []
        self.tpl_edge_indexs = []
        self.geo_edge_indexs = []

        if self.preload:
            self._preload()
        else:
            raise ValueError

        print('Rignet dataset:', flag)
        print('Number of subjects:', len(self))

    def get(self, index):
        v0, f, joints, parents, weights, tpl_edge_index, geo_edge_index, name = self.load(index)
        tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
        tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=v0.shape[0])
        geo_edge_index = torch.from_numpy(geo_edge_index).long()

        theta1 = np.random.normal(scale=0.3, size=joints.shape)
        theta1[0] *= 0
        theta2 = np.random.normal(scale=0.3, size=joints.shape)
        theta2[0] *= 0
        weights = np.array(weights.todense())
        v1, j_trans1, T1 = lbs(v0, theta1, joints, parents, weights, verbose=True)
        v2, j_trans2, T2 = lbs(v0, theta2, joints, parents, weights, verbose=True)

        v0 = torch.from_numpy(v0).float()
        v1 = torch.from_numpy(v1).float()
        v2 = torch.from_numpy(v2).float()
        T1 = torch.from_numpy(T1).float()

        normal_v0 = get_normal(v0, f)
        normal_v1 = get_normal(v1, f)

        return Data(v0=v0, v1=v1, tpl_edge_index=tpl_edge_index, triangle=f[None].astype(int), name=name,
                    aug_v0=v0, aug_v1=v1, aug_T=T1, aug_joints=joints[None],
                    feat0=normal_v0, feat1=normal_v1, geo_edge_index=geo_edge_index,
                    joints=joints[None], weights=weights, parents=parents[None], theta=theta1[None],
                    num_nodes=len(v0), dataset=2), \
               Data(v0=v0, v1=v1, tpl_edge_index=tpl_edge_index, triangle=f[None].astype(int), name=name,
                    aug_v0=v0, aug_v1=v1, aug_T=T1, aug_joints=joints[None],
                    feat0=normal_v0, feat1=normal_v1, geo_edge_index=geo_edge_index,
                    joints=joints[None], weights=weights, parents=parents[None], theta=theta1[None],
                    num_nodes=len(v0), dataset=2)

    def get_uniform(self, index):
        return self.get(index)

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.joints[index], \
                   self.parents[index], self.weights[index], self.tpl_edge_indexs[index], \
                   self.geo_edge_indexs[index], self.names[index]

    def len(self):
        return len(self.names)

    def _preload(self):
        for idx in self.names:
            c = np.load(os.path.join(self.data_dir, 'npz', idx+'.npz'))

            v = c["v"]
            center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
            scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
            v = (v - center) / scale
            joints = c["joints"]
            joints = (joints - center) / scale

            self.vs.append(v)
            self.fs.append(c["f"])
            self.joints.append(joints)
            self.parents.append(c["parents"])
            self.weights.append(csr_matrix((c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                                           shape=c["weights_shape"], dtype=np.float32))
            self.tpl_edge_indexs.append(c["tpl_edge_index"].astype(int).T)
            self.geo_edge_indexs.append(np.stack([c["geo_rows"].astype(np.int32), c["geo_cols"].astype(np.int32)], 0))


if __name__ == '__main__':
    from torch_geometric.data import Batch, DataLoader
    dataset = RignetDataset("/Users/zliao/Data/ModelResource_RigNetv1_preproccessed", flag='humanoid_test_tet')
    a = dataset.get(5)
    b = Batch(batch=a)
    b = 1
