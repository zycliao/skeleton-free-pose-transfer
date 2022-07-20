import os
import numpy as np
import torch
from torch import Tensor

from smplx.lbs import batch_rodrigues, batch_rigid_transform, blend_shapes, vertices2joints


def lbs(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
        no_lbs = False,
        verbose = False,
        disp = None,
):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    if no_lbs:
        I_T = torch.eye(4).float().to(J.device)[None][None]
        I_T = I_T.repeat(v_posed.shape[0], v_posed.shape[1], 1, 1)
        return v_posed, J, I_T

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    if disp is not None:
        v_posed = v_posed + disp
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if verbose:
        return verts, J_transformed, T
    else:
        return verts, J_transformed


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

    def __call__(self, pose_all, shape_all, gender_all, disp_all=None, no_lbs=False, no_psd=False, ret_J=False, ret_T=False):
        if ret_T:
            ret_J = True
        if isinstance(pose_all, np.ndarray):
            pose_all = torch.from_numpy(pose_all).to(torch.float32)
        if isinstance(shape_all, np.ndarray):
            shape_all = torch.from_numpy(shape_all).to(torch.float32)
        if isinstance(gender_all, torch.Tensor):
            gender_all = gender_all.detach().cpu().numpy()
        if disp_all is not None and isinstance(disp_all, np.ndarray):
            disp_all = torch.from_numpy(disp_all).to(torch.float32)

        all_size = len(gender_all)
        batch_size = 256
        batch_num = 1 + (all_size - 1) // batch_size
        mesh_all = []
        J_all = []
        T_all = []
        for bi in range(batch_num):
            l = bi * batch_size
            r = np.minimum((bi + 1) * batch_size, all_size)
            cur_bs = r - l
            pose = pose_all[l: r]
            shape = shape_all[l: r]
            gender = gender_all[l: r]
            if disp_all is not None:
                disp = disp_all[l: r]

            gender_ind = {}
            gender_ind['male'] = [idx for (idx, g) in enumerate(gender) if g==-1]
            gender_ind['female'] = [idx for (idx, g) in enumerate(gender) if g==1]

            verts = {}
            Js = {}
            Ts= {}
            for gdr in ['male', 'female']:
                if not gender_ind[gdr]:
                    continue

                gdr_betas = shape[gender_ind[gdr]]
                gdr_pose = pose[gender_ind[gdr]]

                v_template = np.repeat(self.models[gdr]['v_template'][np.newaxis], len(gdr_betas), axis=0)
                v_template = torch.tensor(v_template, dtype=torch.float32)

                if disp_all is not None:
                    gdr_disp = disp[gender_ind[gdr]]
                else:
                    gdr_disp = None

                shapedirs = torch.tensor(self.models[gdr]['shapedirs'], dtype=torch.float32)

                posedirs = self.models[gdr]['posedirs']
                posedirs = posedirs.reshape(posedirs.shape[0]*3, -1).T
                posedirs = torch.tensor(posedirs, dtype=torch.float32)
                if no_psd:
                    posedirs = 0 * posedirs

                J_regressor = torch.tensor(self.models[gdr]['J_regressor'], dtype=torch.float32)

                parents = torch.tensor(self.models[gdr]['kintree_table'][0], dtype=torch.int32).long()

                lbs_weights = torch.tensor(self.models[gdr]['weights'], dtype=torch.float32)

                v, J, T = lbs(gdr_betas, gdr_pose, v_template, shapedirs, posedirs,
                           J_regressor, parents, lbs_weights, no_lbs=no_lbs, verbose=True, disp=gdr_disp)

                verts[gdr] = v
                Js[gdr] = J
                Ts[gdr] = T

            mesh = torch.zeros(cur_bs, 6890, 3)
            Js_batch = torch.zeros(cur_bs, 52, 3)
            Ts_batch = torch.zeros(cur_bs, 6890, 4, 4)

            for gdr in ['male', 'female']:
                if gdr in verts:
                    mesh[gender_ind[gdr]] = verts[gdr]
                    Js_batch[gender_ind[gdr]] = Js[gdr]
                    Ts_batch[gender_ind[gdr]] = Ts[gdr]

            mesh_all.append(mesh)
            if ret_J:
                J_all.append(Js_batch)
            if ret_T:
                T_all.append(Ts_batch)
        mesh_all = torch.cat(mesh_all, 0)

        if ret_T:
            J_all = torch.cat(J_all, 0)
            T_all = torch.cat(T_all, 0)
            return mesh_all, J_all, T_all
        elif ret_J:
            J_all = torch.cat(J_all, 0)
            return mesh_all, J_all
        else:
            return mesh_all
