import os
import numpy as np
import torch
import torch.utils.data
from scipy import sparse
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.geometry import get_tpl_edges, part_scaling, get_normal
from utils.lbs import lbs
from utils.general import np2torch, torch2np
from global_var import *


def normalize_T(T, scale, center):
    M = np.eye(4)
    M[np.arange(3), np.arange(3)] = 1/scale
    M[:3, 3] = -center/scale

    M = torch.from_numpy(M).float().to(T.device)
    M_inv = torch.linalg.inv(M)
    return M@T@M_inv


class AmassDataset(Dataset):
    def __init__(self, data_path, smpl, flag='train', part_augmentation=False, part_aug_scale=1,
                 simplify=True):
        super(AmassDataset, self).__init__()
        self.no_hand = True
        file_path = os.path.join(data_path, flag + '.npy')
        print(file_path)
        self.data = np.load(file_path)
        if part_aug_scale <= 0:
            part_augmentation = False
        self.part_augmentation = part_augmentation
        self.part_aug_scale = part_aug_scale
        self.smpl = smpl
        self.simplify = simplify

        self.subj_ind = {}
        self.subj_list = []
        for i, row in enumerate(self.data):
            if row['f0'] not in self.subj_ind:
                self.subj_ind[row['f0']] = [i]
            else:
                self.subj_ind[row['f0']].append(i)
            self.subj_list.append(row['f0'])
        self.num_subj = len(self.subj_ind)

        self.canonical_v = self.smpl.models['male']['v_template'].astype(np.float32)
        self.canonical_f = self.smpl.models['male']['f']

        self.d_weights = np.array(smpl.models['male']['weights'].astype(np.float32))
        if self.no_hand:
            self.d_weights[:, 20] += np.sum(self.d_weights[:, 22: 37], 1)
            self.d_weights[:, 21] += np.sum(self.d_weights[:, 37: 52], 1)
            self.d_weights = self.d_weights[:, :22]
        self.s_weights = sparse.csr_matrix(self.d_weights)

        if self.simplify:
            assert hasattr(self.smpl, "nearest_face")
            self.low_f = self.smpl.low_f
            self.canonical_v = self.smpl.simplify(self.canonical_v)
            self.canonical_f = self.smpl.low_f
            self.d_weights = self.smpl.simplify(self.d_weights)
            self.s_weights = sparse.csr_matrix(self.d_weights)

        parents = smpl.models['male']['kintree_table'][0].astype(int)
        parents[0] = -1
        if self.no_hand:
            parents = parents[:22]

        self.parents = parents
        tpl_edge_index = get_tpl_edges(self.canonical_v, self.canonical_f)
        self.tpl_edge_index = torch.from_numpy(tpl_edge_index.T).long()
        self.tpl_edge_index, _ = add_self_loops(self.tpl_edge_index, num_nodes=self.canonical_v.shape[0])
        self.tpl_edge_index_np = tpl_edge_index.T

        # for part augmentation
        children = [[] for _ in range(len(parents))]
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)

        def add_children(idx, joint_idxs):
            joint_idxs.append(idx)
            if len(children[idx]) == 0:
                return
            for c in children[idx]:
                add_children(c, joint_idxs)
        # 1. head
        neck = 12
        self.head_joint_idxs = []
        add_children(neck, self.head_joint_idxs)
        # 2. arms
        lshoulder = 16
        rshoulder = 17
        self.arm_joint_idxs = []
        add_children(lshoulder, self.arm_joint_idxs)
        add_children(rshoulder, self.arm_joint_idxs)
        # 3. legs
        lleg = 1
        rleg = 2
        self.leg_joint_idxs = [0]
        add_children(lleg, self.leg_joint_idxs)
        add_children(rleg, self.leg_joint_idxs)

        print('Amass dataset:', flag)
        print('Number of subjects:', self.num_subj)
        print('Number of meshes:', len(self.data))

    def get(self, index):
        if index >= len(self):
            raise StopIteration

        idx1 = index
        subj_i = self.subj_list[index]
        # idx2 = np.random.choice(self.subj_ind[subj_i], 1)[0]
        while True:
            idx2 = np.random.randint(0, len(self))
            if self.subj_list[idx2] != subj_i:
                break

        _, theta1, s1, gdr1 = self.get_data(idx1)
        if self.no_hand:
            theta1[22*3:] *= 0
        _, _, s2, gdr2 = self.get_data(idx2)
        theta2 = theta1

        v0_1, joints0_1 = self.smpl(np.zeros([1, 156]), s1[None], [gdr1], ret_J=True)
        v0_2, joints0_2 = self.smpl(np.zeros([1, 156]), s2[None], [gdr2], ret_J=True)
        v0_1 = v0_1[0]
        v0_2 = v0_2[0]
        joints0_1 = joints0_1[0]
        joints0_2 = joints0_2[0]
        v0_1, v0_2, joints0_1, joints0_2 = torch2np(v0_1, v0_2, joints0_1, joints0_2)
        if self.simplify:
            v0_1 = self.smpl.simplify(v0_1)
            v0_2 = self.smpl.simplify(v0_2)

        if self.no_hand:
            theta1 = theta1[:22*3]
            theta2 = theta2[:22*3]
            joints0_1 = joints0_1[:22]
            joints0_2 = joints0_2[:22]

        center1 = (np.max(v0_1, 0, keepdims=True) + np.min(v0_1, 0, keepdims=True)) / 2
        scale1 = np.max(v0_1[:, 1], 0) - np.min(v0_1[:, 1], 0)
        center2 = (np.max(v0_2, 0, keepdims=True) + np.min(v0_2, 0, keepdims=True)) / 2
        scale2 = np.max(v0_2[:, 1], 0) - np.min(v0_2[:, 1], 0)

        v0_1 = (v0_1 - center1) / scale1
        joints0_1 = (joints0_1 - center1) / scale1
        v0_2 = (v0_2 - center2) / scale2
        joints0_2 = (joints0_2 - center2) / scale2


        scale_min = np.log(np.array([0.5, 0.6, 0.3]))
        scale_max = np.log(np.array([4, 1, 1.5]))
        scale1 = np.exp(np.random.uniform(scale_min, scale_max))
        scale2 = np.exp(np.random.uniform(scale_min, scale_max))
        if not self.part_augmentation:
            scale1 = scale2 = np.ones([3])

        aug_v0_1, aug_joints0_1 = self.part_augment(v0_1, joints0_1, scale1)
        aug_v0_2, aug_joints0_2 = self.part_augment(v0_2, joints0_2, scale2)

        v1, joints1, T1 = lbs(v0_1, theta1, joints0_1, self.parents, self.d_weights, verbose=True)
        v2, joints2, T2 = lbs(v0_2, theta2, joints0_2, self.parents, self.d_weights, verbose=True)
        aug_v1, aug_joints1, aug_T1 = lbs(aug_v0_1, theta1, aug_joints0_1,  self.parents, self.d_weights, verbose=True)
        aug_v2, aug_joints2, aug_T2 = lbs(aug_v0_2, theta2, aug_joints0_2,  self.parents, self.d_weights, verbose=True)

        v0_1, v0_2, v1, v2, T1, T2 = np2torch(v0_1, v0_2, v1, v2, T1, T2)
        aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2 = np2torch(aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2)

        weights = self.d_weights
        parents = self.parents[None]

        # v_idx = np.where(np.sum(self.d_weights[:, self.head_joint_idxs], 1) > 0.1)[0]
        # f = get_part_mesh(v_idx, self.canonical_f)
        # v0_1 = v0_1[v_idx]
        normal_v0_1 = get_normal(v0_1, self.canonical_f)
        normal_v1 = get_normal(v1, self.canonical_f)
        normal_v0_2 = get_normal(v0_2, self.canonical_f)
        normal_v2 = get_normal(v2, self.canonical_f)


        return Data(v0=v0_1, v1=v1, tpl_edge_index=self.tpl_edge_index, triangle=self.canonical_f[None], name=str(index),
                    aug_v0=aug_v0_1, aug_v1=aug_v1, aug_T=aug_T1, aug_joints=aug_joints0_1[None],
                    feat0=normal_v0_1, feat1=normal_v1,
                    joints=joints0_1[None], weights=weights, parents=parents, theta=theta1[None],
                    num_nodes=len(v0_1), dataset=0), \
               Data(v0=v0_2, v1=v2, tpl_edge_index=self.tpl_edge_index, triangle=self.canonical_f[None], name=str(index),
                    aug_v0=aug_v0_2, aug_v1=aug_v2, aug_T=aug_T2, aug_joints=aug_joints0_2[None],
                    feat0=normal_v0_2, feat1=normal_v2,
                    joints=joints0_2[None], weights=weights, parents=parents, theta=theta2[None],
                    num_nodes=len(v0_2), dataset=0)

    def get_uniform(self, index):
        return self.get(index)

    def get_data(self, idx):
        gdr = self.data[idx]['f1']
        shape = self.data[idx]['f2']
        pose = self.data[idx]['f3']
        tpose_v = self.smpl(pose[None]*0, shape[None], [gdr]).numpy()[0]
        return tpose_v, pose, shape, gdr

    def part_augment(self, v, joints, scale=(1., 1., 1.)):
        deformed_v, deformed_joints = part_scaling(v, self.tpl_edge_index_np, self.d_weights, joints,
                                                   self.head_joint_idxs, scale[0])
        deformed_v, deformed_joints = part_scaling(deformed_v, self.tpl_edge_index_np, self.d_weights, deformed_joints,
                                                   self.arm_joint_idxs, [[scale[1], 1., 1.]])
        deformed_v, deformed_joints = part_scaling(deformed_v, self.tpl_edge_index_np, self.d_weights, deformed_joints,
                                                   self.leg_joint_idxs, [[1., scale[2], 1.]])

        if np.isnan(deformed_v).any() or np.isinf(deformed_v).any():
            return v, joints

        center = (np.max(deformed_v, 0, keepdims=True) + np.min(deformed_v, 0, keepdims=True)) / 2
        scale = np.max(deformed_v[:, 1], 0) - np.min(deformed_v[:, 1], 0)
        deformed_v = (deformed_v - center) / scale
        deformed_joints = (deformed_joints - center) / scale
        return deformed_v, deformed_joints


    def len(self):
        return len(self.data)


class AmassSeqDataset(Dataset):
    def __init__(self, data_path, smpl, flag, simplify=True):
        super(AmassSeqDataset, self).__init__()
        file_path = os.path.join(data_path, flag + '.npy')
        print(file_path)
        self.data = np.load(file_path)
        self.smpl = smpl
        self.flag = flag
        self.simplify = simplify
        self.no_hand = True

        self.canonical_v = self.smpl.models['male']['v_template']
        self.canonical_f = self.smpl.models['male']['f']

        self.d_weights = np.array(smpl.models['male']['weights'].astype(np.float32))
        if self.no_hand:
            self.d_weights[:, 20] += np.sum(self.d_weights[:, 22: 37], 1)
            self.d_weights[:, 21] += np.sum(self.d_weights[:, 37: 52], 1)
            self.d_weights = self.d_weights[:, :22]
        self.s_weights = sparse.csr_matrix(self.d_weights)

        if self.simplify:
            assert hasattr(self.smpl, "nearest_face")
            self.low_f = self.smpl.low_f
            self.canonical_v = self.smpl.simplify(self.canonical_v)
            self.canonical_f = self.smpl.low_f
            self.d_weights = self.smpl.simplify(self.d_weights)
            self.s_weights = sparse.csr_matrix(self.d_weights)
        self.canonical_f = self.canonical_f.astype(int)

        parents = smpl.models['male']['kintree_table'][0].astype(int)
        parents[0] = -1
        if self.no_hand:
            parents = parents[:22]

        self.parents = parents
        tpl_edge_index = get_tpl_edges(self.canonical_v, self.canonical_f)
        tpl_edge_index = torch.from_numpy(tpl_edge_index.T).long()
        self.tpl_edge_index, _ = add_self_loops(tpl_edge_index, num_nodes=self.canonical_v.shape[0])


    def get(self, index):
        if index >= len(self):
            raise StopIteration

        idx1 = index
        theta1, beta1, gdr1 = self.get_data(idx1)

        tpose_v1, joints0_1 = self.smpl(np.zeros([1, 156]), beta1[None], [-1],
                                    ret_J=True)
        tpose_v1 = tpose_v1[0]
        joints0_1 = joints0_1[0]
        tpose_v1, joints0_1 = torch2np(tpose_v1, joints0_1)

        if self.simplify:
            tpose_v1 = self.smpl.simplify(tpose_v1)

        if self.no_hand:
            theta1 = theta1[:22*3]
            joints0_1 = joints0_1[:22]

        v1, joints_1, T1 = lbs(tpose_v1, theta1, joints0_1, self.parents, self.d_weights, verbose=True)

        # if self.simplify:
        #     v1 = self.smpl.simplify(v1)
        #     tpose_v1 = self.smpl.simplify(tpose_v1)

        center1 = (np.max(tpose_v1, 0, keepdims=True) + np.min(tpose_v1, 0, keepdims=True)) / 2
        scale1 = np.max(tpose_v1[:, 1], 0) - np.min(tpose_v1[:, 1], 0)

        v1 = (v1 - center1) / scale1
        tpose_v1 = (tpose_v1 - center1) / scale1

        v0_1 = torch.from_numpy(tpose_v1).float()
        v_1 = torch.from_numpy(v1).float()

        normal_v0_1 = get_normal(v0_1, self.canonical_f)
        normal_v1 = get_normal(v_1, self.canonical_f)

        return Data(
            v0=v0_1, v1=v_1,
            feat0=normal_v0_1, feat1=normal_v1,
            tpl_edge_index=self.tpl_edge_index, face=torch.from_numpy(self.canonical_f.T.astype(int)).long(),
            triangle=self.canonical_f[None],
            weights=self.d_weights,
            theta_1=torch.from_numpy(theta1[None]).float(), beta_1=torch.from_numpy(beta1[None]).float(),
            gender_1=gdr1, num_nodes=len(v1))

    def get_data(self, idx):
        gdr = self.data[idx]['f1']
        shape = self.data[idx]['f2']
        pose = self.data[idx]['f3']
        return pose, shape, gdr

    def len(self):
        return len(self.data)


if __name__ == '__main__':
    from global_var import *
    import cv2
    from models.smpl import SMPL2Mesh
    from utils.render import Renderer
    smpl = SMPL2Mesh(SMPLH_PATH)
    dataset = AmassDataset(AMASS_PATH, smpl, "train", part_augmentation=True)
    renderer = Renderer(400)

    def equal(a, b):
        print(np.max(np.abs(a-b)))

    for i in range(10):
        idx = np.random.randint(len(dataset))
        data, _ = dataset.get_uniform(idx)

        # equal(data.v0.numpy(), data2.v0.numpy())
        # equal(data.v1.numpy(), data.aug_v1.numpy())

        img = renderer(data.aug_v1.numpy(), data.triangle[0])
        cv2.imshow('a', np.concatenate((img, img), 1))
        cv2.waitKey()