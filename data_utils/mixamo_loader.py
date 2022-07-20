import os
import cv2
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_self_loops
from utils.lbs import lbs
from scipy.spatial.transform import Rotation
from utils.geometry import part_scaling, get_normal
from utils.bvh import Bvh
from utils.general import np2torch
from data_utils.common import get_geo_index
from scipy.sparse import csr_matrix
from global_var import *


def normalize_y_rotation(raw_theta):
    """
    rotate along y axis so that root rotation can always face the camera
    theta should be a [3] or [66] numpy array
    """
    only_global = True
    if raw_theta.shape == (66,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta


def extend_pose(joint_names, poses):
    poses = poses.reshape([-1, 3])
    num_joints = len(joint_names)
    ext_poses = np.zeros([num_joints, 3], dtype=np.float32)
    for i, name in enumerate(MIXAMO_JOINTS):
        assert name in joint_names, "{} not in {}".format(name, joint_names)
        ext_poses[joint_names.index(name)] = poses[i]
    return ext_poses.reshape([-1])


def sort_joints(joint_names, joints):
    sorted_joints = []
    for i, name in enumerate(MIXAMO_JOINTS):
        sorted_joints.append(joints[joint_names.index(name)])
    sorted_joints = np.array(sorted_joints).copy()
    return sorted_joints


def part_augment(v, tpl_edge_index, weights, joints, joint_names, parents, scale, log_prob=False):
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

    if log_prob:
        scale = np.log(scale)
    p = np.random.rand()
    if p < 1/3:
        neck = joint_names.index("Neck")
        head_joint_idxs = []
        add_children(neck, head_joint_idxs)
        joint_idxs = head_joint_idxs
        scale = np.random.uniform(scale[0][0], scale[0][1])
    elif p < 2/3:
        lshoulder = joint_names.index("LeftShoulder")
        rshoulder = joint_names.index("RightShoulder")
        arm_joint_idxs = []
        add_children(lshoulder, arm_joint_idxs)
        add_children(rshoulder, arm_joint_idxs)
        joint_idxs = arm_joint_idxs
        scale = np.random.uniform(scale[1][0], scale[1][1])
    else:
        lleg = joint_names.index("LeftUpLeg")
        rleg = joint_names.index("RightUpLeg")
        leg_joint_idxs = [joint_names.index("Hips")]
        add_children(lleg, leg_joint_idxs)
        add_children(rleg, leg_joint_idxs)
        joint_idxs = leg_joint_idxs
        scale = np.random.uniform(scale[2][0], scale[2][1])
    if log_prob:
        scale = np.exp(scale)
    deformed_v, deformed_joints = part_scaling(v, tpl_edge_index, weights, joints, joint_idxs, scale)
    center = (np.max(deformed_v, 0, keepdims=True) + np.min(deformed_v, 0, keepdims=True)) / 2
    scale = np.max(deformed_v[:, 1], 0) - np.min(deformed_v[:, 1], 0)
    deformed_v = (deformed_v - center) / scale
    deformed_joints = (deformed_joints - center) / scale
    if np.isnan(deformed_v).any() or np.isinf(deformed_v).any():
        return v, joints
    return deformed_v, deformed_joints


def part_augment2(v, tpl_edge_index, weights, joints, joint_names, parents, scale=(1., 1., 1.)):
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
    neck = joint_names.index("Neck")
    head_joint_idxs = []
    add_children(neck, head_joint_idxs)
    joint_idxs = head_joint_idxs
    deformed_v, deformed_joints = part_scaling(v, tpl_edge_index, weights, joints, joint_idxs, scale[0])

    lshoulder = joint_names.index("LeftShoulder")
    rshoulder = joint_names.index("RightShoulder")
    arm_joint_idxs = []
    add_children(lshoulder, arm_joint_idxs)
    add_children(rshoulder, arm_joint_idxs)
    joint_idxs = arm_joint_idxs
    deformed_v, deformed_joints = part_scaling(deformed_v, tpl_edge_index, weights,
                                               deformed_joints, joint_idxs, [[scale[1], 1., 1.]])

    lleg = joint_names.index("LeftUpLeg")
    rleg = joint_names.index("RightUpLeg")
    leg_joint_idxs = [joint_names.index("Hips")]
    add_children(lleg, leg_joint_idxs)
    add_children(rleg, leg_joint_idxs)
    joint_idxs = leg_joint_idxs
    deformed_v, deformed_joints = part_scaling(deformed_v, tpl_edge_index, weights,
                                               deformed_joints, joint_idxs, [[1., scale[2], 1.]])


    center = (np.max(deformed_v, 0, keepdims=True) + np.min(deformed_v, 0, keepdims=True)) / 2
    scale = np.max(deformed_v[:, 1], 0) - np.min(deformed_v[:, 1], 0)
    deformed_v = (deformed_v - center) / scale
    deformed_joints = (deformed_joints - center) / scale
    if np.isnan(deformed_v).any() or np.isinf(deformed_v).any():
        return v, joints
    return deformed_v, deformed_joints


def merge_skinning_weight(weight, joint_names, parents):
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
    ee_joint_idxs = [[] for k in range(5)]
    ee_names = ["Head", "LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]
    ee_ext_names = []
    for i, ee_name in enumerate(ee_names):
        joint_idx = joint_names.index(ee_name)
        add_children(joint_idx, ee_joint_idxs[i])
        for idx in ee_joint_idxs[i]:
            ee_ext_names.append(joint_names[idx])

    n_vertices = weight.shape[0]
    new_weight = np.zeros([n_vertices, len(MIXAMO_JOINTS)])
    for i, name in enumerate(MIXAMO_JOINTS):
        if name not in ee_ext_names:
            new_weight[:, i] = weight[:, joint_names.index(name)]
        else:
            if name in ee_names:
                new_weight[:, i] = np.sum(weight[:, ee_joint_idxs[ee_names.index(name)]], 1)
    return new_weight


class MixamoDataset(Dataset):
    # v1 and v2 are the same character with different pose!!!
    # if you want to have different characters with the same pose, use MixamoDatasetPaired
    def __init__(self, data_dir, flag=None, preload=True,
                 augmentation=False, part_augmentation=False, single_part=True,
                 part_aug_scale=((0.5, 2.2), (0.7, 1.5), (0.7, 1.5)),
                 log_prob=False):
        super(MixamoDataset, self).__init__()
        self.data_dir = data_dir
        self.single_part = single_part
        self.preload = preload
        self.augmentation = augmentation
        self.part_augmentation = part_augmentation
        self.part_aug_scale = part_aug_scale
        self.flag = flag
        self.log_prob = log_prob

        if flag:
            with open(os.path.join(data_dir, f'character_{flag}.txt')) as f:
                self.names = f.read().splitlines()
        else:
            self.names = os.listdir(os.path.join(data_dir, 'npz'))
            self.names = [k.replace('.npz', '') for k in self.names if k.endswith('.npz')]

        self.vs, self.fs = [], []
        self.joints, self.parents, self.weights = [], [], []
        self.joint_names = []
        self.tpl_edge_indexs, self.geo_edge_indexs = [], []
        self.poses = []

        if self.preload:
            self._preload()
            self._preload_motion()
        else:
            raise ValueError
        print('Mixamo dataset:', flag)
        print('Number of subjects:', len(self.vs))
        print('Number of poses:', len(self.poses))

    def get(self, index):
        # return two characters with the same pose. Both of them have a random augmentation
        index1 = index % len(self.vs)
        index2 = np.random.randint(0, len(self.vs))
        v0_1, f1, joints0_1, joint_names1, parents1, weights1, tpl_edge_index1, \
            geo_edge_index1, name1 = self.load(index1)
        v0_2, f2, joints0_2, joint_names2, parents2, weights2, tpl_edge_index2, \
            geo_edge_index2, name2 = self.load(index2)

        part_aug_scale = np.array(self.part_aug_scale)
        scale_min = np.log(part_aug_scale[:, 0])
        scale_max = np.log(part_aug_scale[:, 1])
        scale1 = np.exp(np.random.uniform(scale_min, scale_max))
        scale2 = np.exp(np.random.uniform(scale_min, scale_max))
        if not self.part_augmentation:
            scale1 = scale2 = np.ones([3])
        if self.single_part:
            scale1 = random_mask(scale1)
            scale2 = random_mask(scale2)

        weights1 = np.array(weights1.todense()).astype(np.float32)
        weights2 = np.array(weights2.todense()).astype(np.float32)
        if self.part_augmentation:
            aug_v0_1, aug_joints0_1 = part_augment2(v0_1, tpl_edge_index1, weights1,
                                                    joints0_1, joint_names1, parents1, scale1)
            aug_v0_2, aug_joints0_2 = part_augment2(v0_2, tpl_edge_index2, weights2,
                                                    joints0_2, joint_names2, parents2, scale2)
        else:
            aug_v0_1, aug_joints0_1 = v0_1, joints0_1
            aug_v0_2, aug_joints0_2 = v0_2, joints0_2

        theta1_ = self.poses[np.random.randint(0, len(self.poses))]
        theta1_ = normalize_y_rotation(theta1_)
        theta1 = extend_pose(joint_names1, theta1_)
        theta2 = extend_pose(joint_names2, theta1_)

        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])
        tpl_edge_index2 = torch.from_numpy(tpl_edge_index2).long()
        tpl_edge_index2, _ = add_self_loops(tpl_edge_index2, num_nodes=v0_2.shape[0])
        geo_edge_index1 = torch.from_numpy(geo_edge_index1).long()
        geo_edge_index2 = torch.from_numpy(geo_edge_index2).long()

        v1, joints1, T1 = lbs(v0_1, theta1, joints0_1, parents1, weights1, verbose=True)
        v2, joints2, T2 = lbs(v0_2, theta2, joints0_2, parents2, weights2, verbose=True)
        aug_v1, aug_joints1, aug_T1 = lbs(aug_v0_1, theta1, aug_joints0_1, parents1, weights1, verbose=True)
        aug_v2, aug_joints2, aug_T2 = lbs(aug_v0_2, theta2, aug_joints0_2, parents2, weights2, verbose=True)

        v0_1, v0_2, v1, v2, T1, T2 = np2torch(v0_1, v0_2, v1, v2, T1, T2)
        aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2 = np2torch(aug_v0_1, aug_v0_2, aug_v1, aug_v2, aug_T1, aug_T2)

        normal_v0_1 = get_normal(v0_1, f1)
        normal_v1 = get_normal(v1, f1)
        normal_v0_2 = get_normal(v0_2, f2)
        normal_v2 = get_normal(v2, f2)
        
        if joint_names1[0] in MIXAMO_JOINTS:
            new_weights1 = merge_skinning_weight(weights1, joint_names1, parents1)
            new_weights2 = merge_skinning_weight(weights2, joint_names2, parents2)
        else:
            # this is not real MIXAMO, just using the interface (e.g., result of NBS)
            new_weights1 = weights1
            new_weights2 = weights2

        return Data(v0=v0_1, v1=v1, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int), name=name1,
                    aug_v0=aug_v0_1, aug_v1=aug_v1, aug_T=aug_T1, aug_joints=aug_joints0_1[None],
                    feat0=normal_v0_1, feat1=normal_v1, geo_edge_index=geo_edge_index1,
                    joints=joints0_1[None], weights=new_weights1, parents=parents1[None], theta=theta1[None],
                    num_nodes=len(v0_1), dataset=1), \
               Data(v0=v0_2, v1=v2, tpl_edge_index=tpl_edge_index2, triangle=f2[None].astype(int), name=name2,
                    aug_v0=aug_v0_2, aug_v1=aug_v2, aug_T=aug_T2, aug_joints=aug_joints0_2[None],
                    feat0=normal_v0_2, feat1=normal_v2, geo_edge_index=geo_edge_index2,
                    joints=joints0_2[None], weights=new_weights2, parents=parents2[None], theta=theta2[None],
                    num_nodes=len(v0_2), dataset=1)

    def get_uniform(self, index):
        return self.get(index)

    def articulate(self, index_char, index_pose):
        v0, f, joints, joint_names, parents, weights, tpl_edge_index, name = self.load(index_char)
        theta1_ = self.poses[index_pose]
        theta1_ = normalize_y_rotation(theta1_)
        theta1 = extend_pose(joint_names, theta1_)
        v1 = lbs(v0, theta1, joints, parents, np.array(weights.todense()))
        return v1, f

    def animate(self, index, motion_path):
        v, f, joints, joint_names, parents, weights, tpl_edge_index, name = self.load(index)
        mocap = Bvh(motion_path)

        frames = mocap.frames
        n_frame, c_frame = frames.shape
        frames = frames.reshape([n_frame, -1, 3])
        poses = np.zeros([n_frame, len(joints), 3], dtype=np.float32)

        bvh_joint_names = []
        for name in mocap.get_joints_names():
            if ":" in name:
                name = name.split(":")[1]
            bvh_joint_names.append(name)

        for name in MIXAMO_JOINTS:
            assert name in joint_names, "{} not in {}".format(name, joint_names)
            poses[:, joint_names.index(name)] = frames[:, bvh_joint_names.index(name)+1]

        renderer = Renderer(400)
        for pose in poses:
            theta = Rotation.from_euler('XYZ', angles=pose, degrees=True).as_rotvec()
            v1 = lbs(v, theta, joints, parents, np.array(weights.todense()))
            img = renderer(v1, f)
            cv2.imshow("a", img)
            cv2.waitKey()

    def random_pose(self):
        pass

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get(idx)

    def load(self, index):
        if self.preload:
            return self.vs[index], self.fs[index], self.joints[index], self.joint_names[index], \
                   self.parents[index], self.weights[index], self.tpl_edge_indexs[index], \
                   self.geo_edge_indexs[index], self.names[index]

    def len(self):
        return 1000

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
            self.fs.append(c["f"].astype(int))
            self.joints.append(joints)
            self.parents.append(c["parents"])
            self.joint_names.append(c["joint_names"].tolist())
            self.weights.append(csr_matrix((c["weights_data"], (c["weights_rows"], c["weights_cols"])),
                                           shape=c["weights_shape"], dtype=np.float32))
            self.tpl_edge_indexs.append(c["tpl_edge_index"].astype(int).T)
            self.geo_edge_indexs.append(np.stack([c["geo_rows"].astype(np.int32), c["geo_cols"].astype(np.int32)], 0))

    def _preload_motion(self):
        motion_dir = os.path.join(self.data_dir, 'motion')
        if self.flag:
            with open(os.path.join(self.data_dir, f'motion_{self.flag}.txt')) as f:
                self.motion_names = f.read().splitlines()
        else:
            self.motion_names = [k for k in sorted(os.listdir(motion_dir)) if k.endswith('.npy')]
        self.motion_idxs = {}
        self.count = 0
        for name in self.motion_names:
            frames = np.load(os.path.join(motion_dir, name))
            self.motion_idxs[name] = (self.count, self.count+len(frames))
            self.count += len(frames)
            self.poses.append(frames)
        self.poses = np.concatenate(self.poses, 0).reshape([-1, 66])


def random_mask(scale):
    p = np.random.rand()
    if p < 1/3:
        scale[[1, 2]] = 1
    elif p < 2/3:
        scale[[0, 2]] = 1
    else:
        scale[[0, 1]] = 1
    return scale



class MixamoValidationSingleDataset(MixamoDataset):
    # used to validate single character, not pose transfer
    def __init__(self, data_dir, flag='test'):
        super(MixamoValidationSingleDataset, self).__init__(data_dir, flag=flag, part_augmentation=False)

    def get(self, index):
        # return two characters with the same pose
        index1 = index % len(self.vs)
        v0_1, f1, joints0_1, joint_names1, parents1, weights1, tpl_edge_index1, \
        geo_edge_index1, name1 = self.load(index1)

        part_aug_scale = np.array(self.part_aug_scale)
        scale_min = np.log(part_aug_scale[:, 0])
        scale_max = np.log(part_aug_scale[:, 1])
        scale1 = np.exp(np.random.uniform(scale_min, scale_max))
        if not self.part_augmentation:
            scale1 = scale2 = np.ones([3])
        if self.single_part:
            scale1 = random_mask(scale1)

        weights1 = np.array(weights1.todense()).astype(np.float32)
        if self.part_augmentation:
            aug_v0_1, aug_joints0_1 = part_augment2(v0_1, tpl_edge_index1, weights1,
                                                    joints0_1, joint_names1, parents1, scale1)
        else:
            aug_v0_1, aug_joints0_1 = v0_1, joints0_1

        theta1_ = self.poses[np.random.randint(0, len(self.poses))]
        theta1_ = normalize_y_rotation(theta1_)
        if joint_names1[0] in MIXAMO_JOINTS:
            theta1 = extend_pose(joint_names1, theta1_)
            v1, joints1, T1 = lbs(v0_1, theta1, joints0_1, parents1, weights1, verbose=True)
            aug_v1, aug_joints1, aug_T1 = lbs(aug_v0_1, theta1, aug_joints0_1, parents1, weights1, verbose=True)
        else:
            theta1 = theta1_.ravel()
            v1 = v0_1
            aug_v1 = aug_v0_1
            joints1 = joints0_1
            aug_joints1 = aug_joints0_1

        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])
        geo_edge_index1 = torch.from_numpy(geo_edge_index1).long()



        v0_1, v1 = np2torch(v0_1, v1)
        aug_v0_1, aug_v1 = np2torch(aug_v0_1, aug_v1)

        normal_v0_1 = get_normal(v0_1, f1)
        normal_v1 = get_normal(v1, f1)

        if joint_names1[0] in MIXAMO_JOINTS:
            new_weights1 = merge_skinning_weight(weights1, joint_names1, parents1)
        else:
            # this is not real MIXAMO, just using the interface (e.g., result of NBS)
            new_weights1 = weights1

        return Data(v0=v0_1, v1=v1, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int), name=name1,
                    aug_v0=aug_v0_1, aug_v1=aug_v1, aug_joints=aug_joints0_1[None],
                    feat0=normal_v0_1, feat1=normal_v1, geo_edge_index=geo_edge_index1,
                    joints=joints0_1[None], weights=new_weights1, parents=parents1[None], theta=theta1[None],
                    num_nodes=len(v0_1), dataset=1)

    def len(self):
        return len(self.names)



class MixamoValidationDataset(MixamoDataset):
    # validate pose transfer
    def __init__(self, data_dir, flag='validation'):
        super(MixamoValidationDataset, self).__init__(data_dir, flag='test', augmentation=False)
        with open(os.path.join(data_dir, f'{flag}.txt'), 'r') as f:
            c = f.read().splitlines()
        self.info = []
        for cc in c:
            d = cc.split('\t')
            d[2] = int(d[2])
            d[4] = int(d[4])
            self.info.append(d)

    def get(self, index):
        # return two characters with the same pose
        src_name, dst_name, pose_i, _, _ = self.info[index]
        index1 = self.names.index(src_name)
        index2 = self.names.index(dst_name)

        v0_1, f1, joints1, joint_names1, parents1, weights1, tpl_edge_index1, geo_edge_index1, name1 = self.load(index1)
        v0_2, f2, joints2, joint_names2, parents2, weights2, tpl_edge_index2, geo_edge_index2, name2 = self.load(index2)

        theta1_ = self.poses[pose_i]
        theta1_ = normalize_y_rotation(theta1_)
        theta1 = extend_pose(joint_names1, theta1_)
        theta2 = extend_pose(joint_names2, theta1_)

        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])
        tpl_edge_index2 = torch.from_numpy(tpl_edge_index2).long()
        tpl_edge_index2, _ = add_self_loops(tpl_edge_index2, num_nodes=v0_2.shape[0])
        geo_edge_index1 = torch.from_numpy(geo_edge_index1).long()
        geo_edge_index2 = torch.from_numpy(geo_edge_index2).long()

        weights1 = np.array(weights1.todense()).astype(np.float32)
        weights2 = np.array(weights2.todense()).astype(np.float32)
        v1, j_trans1, T1 = lbs(v0_1, theta1, joints1, parents1, weights1, verbose=True)
        v2, j_trans2, T2 = lbs(v0_2, theta2, joints2, parents2, weights2, verbose=True)

        v0_1 = torch.from_numpy(v0_1).float()
        v0_2 = torch.from_numpy(v0_2).float()
        v1 = torch.from_numpy(v1).float()
        v2 = torch.from_numpy(v2).float()
        T1 = torch.from_numpy(T1).float()
        T2 = torch.from_numpy(T2).float()

        normal_v0_1 = get_normal(v0_1, f1)
        normal_v1 = get_normal(v1, f1)
        normal_v0_2 = get_normal(v0_2, f2)
        normal_v2 = get_normal(v2, f2)

        return Data(v0=v0_1, v1=v1, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int),
                    name=name1, geo_edge_index=geo_edge_index1,
                    feat0=normal_v0_1, feat1=normal_v1,
                    joints=joints1[None], weights=weights1, parents=parents1[None],
                    num_nodes=len(v0_1), T=T1, dataset=1), \
               Data(v0=v0_2, v1=v2, tpl_edge_index=tpl_edge_index2, triangle=f2[None].astype(int),
                    name=name2, geo_edge_index=geo_edge_index2,
                    feat0=normal_v0_2, feat1=normal_v2,
                    joints=joints2[None], weights=weights2, parents=parents2[None],
                    num_nodes=len(v0_2), T=T2, dataset=1)

    def len(self):
        return len(self.info)


class MixamoValidationSeqDataset(MixamoDataset):
    def __init__(self, data_dir, flag='test'):
        super(MixamoValidationSeqDataset, self).__init__(data_dir, flag=flag, augmentation=False)

        self.character_index = 0
        self.cur_poses = []
        self.set_motion(0)
        self.motion_num = len(self.motion_names)

    def get(self, index):
        # return two characters with the same pose
        v0_1, f1, joints1, joint_names1, parents1, weights1, \
            tpl_edge_index1, geo_edge_index1, name1 = self.load(self.character_index)

        theta1_ = self.cur_poses[index]
        theta1_ = normalize_y_rotation(theta1_)
        theta1 = extend_pose(joint_names1, theta1_)

        tpl_edge_index1 = torch.from_numpy(tpl_edge_index1).long()
        tpl_edge_index1, _ = add_self_loops(tpl_edge_index1, num_nodes=v0_1.shape[0])
        geo_edge_index1 = torch.from_numpy(geo_edge_index1).long()

        weights1 = np.array(weights1.todense()).astype(np.float32)

        v1 = lbs(v0_1, theta1, joints1, parents1, weights1)

        v0_1 = torch.from_numpy(v0_1).float()
        v1 = torch.from_numpy(v1).float()

        normal_v0_1 = get_normal(v0_1, f1)
        normal_v1 = get_normal(v1, f1)

        return Data(v0=v0_1, v1=v1, tpl_edge_index=tpl_edge_index1, triangle=f1[None].astype(int),
                    name=name1, geo_edge_index=geo_edge_index1,
                    feat0=normal_v0_1, feat1=normal_v1,
                    joints=joints1[None], weights=weights1, parents=parents1[None],
                    num_nodes=len(v0_1), dataset=1)

    def set_character(self, index):
        if isinstance(index, str):
            index = self.names.index(index)
        self.character_index = index
        self.cur_char_name = self

    def set_motion(self, index):
        if isinstance(index, str):
            index = self.motion_names.index(index)
        l, r = self.motion_idxs[self.motion_names[index]]
        self.cur_poses = self.poses[l: r]
        self.cur_motion_name = self.motion_names[index]

    def len(self):
        return len(self.cur_poses)


if __name__ == '__main__':
    from global_var import *
    import cv2
    from utils.render import Renderer
    from utils.visualization import visualize_part

    # dataset0 = MixamoDataset(MIXAMO_SIMPLIFY_PATH, "test")
    dataset = MixamoDataset(MIXAMO_SIMPLIFY_PATH, "test", part_augmentation=True)
    for i in range(100):
        idx = np.random.randint(len(dataset))
        data1, data2 = dataset.get(idx)
        renderer = Renderer(400)

        img = renderer(*visualize_part(data1.v1.numpy(), data1.triangle[0], None, data1.weights))
        img2 = renderer(data1.v1.numpy(), data1.triangle[0])
        cv2.imshow('a', np.concatenate((img, img2), 1))
        cv2.waitKey()