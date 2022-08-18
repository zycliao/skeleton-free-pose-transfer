import torch
import numpy as np
from utils.geometry import get_nearest_face, barycentric


class MeshResMapper(object):
    def __init__(self, v=None, f=None, orig_v=None, mapper_path=None):
        """
        Given two aligned meshes (m and orig_m) with different triangulations, this class finds their mapping.
        Usually orig_m is the original mesh and m is a remeshed version of orig_m.
        When m deforms, the corresponding orig_m can be obtained using upsample().
        :param v, f: The vertex and face of v. Shape: (N, 3), (M, 3)
        :param orig_v: The vertex of orig_m. Shape: (N2, 3)
        :param mapper_path: a npz path that contains all the mapping parameters.
                            when it is provided, other arguments can be None. Such npz file can be saved using save().
        """
        if mapper_path is not None:
            assert v is not None and orig_v is not None and f is not None
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(orig_v, torch.Tensor):
                orig_v = orig_v.detach().cpu().numpy()
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()

            self.nearest_face = get_nearest_face(orig_v, v, f)
            self.bary = barycentric(orig_v, v[f[:, 0][self.nearest_face]],
                                    v[f[:, 1][self.nearest_face]],
                                    v[f[:, 2][self.nearest_face]])
            dist = np.sum(np.square(orig_v[:, None] - v[None]), 2)
            self.nearest_s2v = np.argmin(dist, 0)
            self.nearest_v2s = np.argmin(dist, 1)
            self.f = f
        else:
            c = np.load(mapper_path)
            self.nearest_face = c['nearest_face']
            self.bary = c['bary']
            self.nearest_s2v = c['s2v']
            self.nearest_v2s = c['v2s']
            self.f = c['f']

    def save(self, path):
        np.savez(path, nearest_face=self.nearest_face,
                 s2v=self.nearest_s2v, v2s=self.nearest_v2s,
                 bary=self.bary, f=self.f)

    def upsample(self, v):
        rec_v = v[self.f[:, 0][self.nearest_face]] * self.bary[:, 0:1] + \
                v[self.f[:, 1][self.nearest_face]] * self.bary[:, 1:2] + \
                v[self.f[:, 2][self.nearest_face]] * self.bary[:, 2:3]
        return rec_v
