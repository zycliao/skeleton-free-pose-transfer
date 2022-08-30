import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.lbs import lbs
from utils.geometry import get_tpl_edges, fps_np, get_normal
from utils.o3d_wrapper import Mesh, MeshO3d
from global_var import *


class CustomDataset(Dataset):
    def __init__(self, data_dir, flag=None, preload=True):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        if isinstance(flag, list) or isinstance(flag, tuple):
            self.names = flag
        elif flag is None:
            self.names = [k.replace('.obj', '') for k in os.listdir(os.path.join(data_dir, 'obj_remesh')) if k.endswith('.obj')]
        else:
            with open(os.path.join(self.data_dir, flag+'.txt')) as f:
                self.names = f.read().splitlines()

        self.vs, self.fs = [], []
        self.tpl_edge_indexs = []
        if self.preload:
            self._preload()
        else:
            raise ValueError

        print('Number of subjects:', len(self))

    def get(self, index):
        v, f, tpl_edge_index, name = self.load(index)
        tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
        tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=v.shape[0])

        center = (np.max(v, 0, keepdims=True) + np.min(v, 0, keepdims=True)) / 2
        scale = np.max(v[:, 1], 0) - np.min(v[:, 1], 0)
        v0 = (v - center) / scale

        v0 = torch.from_numpy(v0).float()
        normal_v0 = get_normal(v0, f)
        return Data(v0=v0, tpl_edge_index=tpl_edge_index, triangle=f[None].astype(int),
                    feat0=normal_v0,
                    name=name, num_nodes=len(v0))

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.tpl_edge_indexs[index], self.names[index]

    def len(self):
        return len(self.names)

    def _preload(self):
        for idx in self.names:
            mesh_path = os.path.join(self.data_dir, 'obj_remesh', idx+'.obj')
            m = Mesh(filename=mesh_path)
            self.vs.append(m.v)
            self.fs.append(m.f)
            tpl_edge_index = get_tpl_edges(m.v, m.f)
            self.tpl_edge_indexs.append(tpl_edge_index.astype(int).T)


class CustomMotionDataset(Dataset):
    def __init__(self, data_dir, flag=None, preload=True):
        super(CustomMotionDataset, self).__init__()
        self.data_dir = data_dir
        self.preload = preload
        if isinstance(flag, list) or isinstance(flag, tuple):
            self.names = flag
        elif flag is None:
            self.names = [k.replace('.obj', '') for k in os.listdir(data_dir) if k.endswith('.obj') and k != "rest.obj"]
        else:
            with open(os.path.join(self.data_dir, flag+'.txt')) as f:
                self.names = f.read().splitlines()

        self.vs, self.fs = [], None
        self.v0, self.normal_v0 = None, None
        self.tpl_edge_indexs = None
        if self.preload:
            self._preload()
        else:
            raise ValueError

        print('Number of poses:', len(self))

    def get(self, index):
        v, _, _, name = self.load(index)
        v1 = (v - self.center) / self.scale
        v1 = torch.from_numpy(v1).float()
        normal_v1 = get_normal(v1, self.f)
        return Data(v0=self.v0, v1=v1, tpl_edge_index=self.tpl_edge_index, triangle=self.f[None].astype(int),
                    feat0=self.normal_v0, feat1=normal_v1,
                    name=name, num_nodes=len(v1))

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs, self.tpl_edge_indexs, self.names[index]

    def len(self):
        return len(self.names)

    def _preload(self):
        # rest mesh
        mesh_path = os.path.join(self.data_dir, 'rest.obj')
        m = Mesh(filename=mesh_path)
        self.v0 = m.v
        self.f = m.f
        tpl_edge_index = get_tpl_edges(m.v, m.f)
        tpl_edge_index = tpl_edge_index.astype(int).T
        tpl_edge_index = torch.from_numpy(tpl_edge_index).long()
        self.tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=self.v0.shape[0])

        self.center = (np.max(self.v0, 0, keepdims=True) + np.min(self.v0, 0, keepdims=True)) / 2
        self.scale = np.max(self.v0[:, 1], 0) - np.min(self.v0[:, 1], 0)
        self.v0 = (self.v0 - self.center) / self.scale
        self.v0 = torch.from_numpy(self.v0).float()
        self.normal_v0 = get_normal(self.v0, self.f)

        # posed mesh
        for idx in self.names:
            mesh_path = os.path.join(self.data_dir, idx+'.obj')
            m = Mesh(filename=mesh_path)
            self.vs.append(m.v)


