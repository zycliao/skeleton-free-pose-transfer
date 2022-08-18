import os
import numpy as np
import torch

from smplx.lbs import lbs


class SMPL2Mesh:
    def __init__(self, bm_path, bm_type='smplh'):
        self.models = {}
        male_npz = np.load(os.path.join(bm_path, 'male/model.npz'))
        female_npz = np.load(os.path.join(bm_path, 'female/model.npz'))
        self.models['male'] = {k: male_npz[k] for k in male_npz}
        self.models['female'] = {k: female_npz[k] for k in female_npz}
        self.f = self.models['male']['f']
        self.bm_type = bm_type
        simplify_path = f"{bm_path}/simplify.npz"
        if os.path.exists(simplify_path):
            c = np.load(simplify_path)
            self.nearest_face = c['nearest_face']
            self.bary = c['bary']
            self.low_f = c['f']

    def simplify(self, v_):
        if isinstance(v_, torch.Tensor):
            v = v_.detach().cpu().numpy()
        else:
            v = v_
        ret = v[self.f[:, 0][self.nearest_face]] * self.bary[:, 0:1] + \
              v[self.f[:, 1][self.nearest_face]] * self.bary[:, 1:2] + \
              v[self.f[:, 2][self.nearest_face]] * self.bary[:, 2:3]
        if isinstance(v_, torch.Tensor):
            return torch.from_numpy(ret).to(v_.device).to(v_.dtype)
        else:
            return ret

    def __call__(self, pose_all, shape_all, gender_all, ret_J=False):
        if isinstance(pose_all, np.ndarray):
            pose_all = torch.from_numpy(pose_all).to(torch.float32)
        if isinstance(shape_all, np.ndarray):
            shape_all = torch.from_numpy(shape_all).to(torch.float32)
        if isinstance(gender_all, torch.Tensor):
            gender_all = gender_all.detach().cpu().numpy()

        all_size = len(gender_all)
        batch_size = 256
        batch_num = 1 + (all_size - 1) // batch_size
        mesh_all = []
        J_all = []
        for bi in range(batch_num):
            l = bi * batch_size
            r = np.minimum((bi + 1) * batch_size, all_size)
            cur_bs = r - l
            pose = pose_all[l: r]
            shape = shape_all[l: r]
            gender = gender_all[l: r]

            gender_ind = {}
            gender_ind['male'] = [idx for (idx, g) in enumerate(gender) if g==-1]
            gender_ind['female'] = [idx for (idx, g) in enumerate(gender) if g==1]

            verts = {}
            Js = {}
            for gdr in ['male', 'female']:
                if not gender_ind[gdr]:
                    continue

                gdr_betas = shape[gender_ind[gdr]]
                gdr_pose = pose[gender_ind[gdr]]

                v_template = np.repeat(self.models[gdr]['v_template'][np.newaxis], len(gdr_betas), axis=0)
                v_template = torch.tensor(v_template, dtype=torch.float32)


                shapedirs = torch.tensor(self.models[gdr]['shapedirs'], dtype=torch.float32)

                posedirs = self.models[gdr]['posedirs']
                posedirs = posedirs.reshape(posedirs.shape[0]*3, -1).T
                posedirs = torch.tensor(posedirs, dtype=torch.float32)
                if no_psd:
                    posedirs = 0 * posedirs

                J_regressor = torch.tensor(self.models[gdr]['J_regressor'], dtype=torch.float32)

                parents = torch.tensor(self.models[gdr]['kintree_table'][0], dtype=torch.int32).long()

                lbs_weights = torch.tensor(self.models[gdr]['weights'], dtype=torch.float32)

                v, J = lbs(gdr_betas, gdr_pose, v_template, shapedirs, posedirs,
                           J_regressor, parents, lbs_weights)

                verts[gdr] = v
                Js[gdr] = J

            mesh = torch.zeros(cur_bs, 6890, 3)
            Js_batch = torch.zeros(cur_bs, 52, 3)

            for gdr in ['male', 'female']:
                if gdr in verts:
                    mesh[gender_ind[gdr]] = verts[gdr]
                    Js_batch[gender_ind[gdr]] = Js[gdr]

            mesh_all.append(mesh)
            if ret_J:
                J_all.append(Js_batch)
        mesh_all = torch.cat(mesh_all, 0)

        if ret_J:
            J_all = torch.cat(J_all, 0)
            return mesh_all, J_all
        else:
            return mesh_all
