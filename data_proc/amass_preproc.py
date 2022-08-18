from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, glob
import torch
import torch.utils.data
import numpy as np
import cv2
# import tables
from global_var import *
from scipy.spatial.transform import Rotation as R


amass_splits = {
    'train': ['BMLmovi', 'KIT', 'EKUT', 'TotalCapture', 'Eyes_Japan_Dataset', 'ACCAD', 'CMU'],
    'test': ['HumanEva', 'SFU', 'Transitions_mocap', 'SSM_synced', 'DanceDB'],
}

np_dtype = np.dtype('<U50, i4, (16)f4, (156)f4')
amass_dir = os.path.join(AMASS_PATH, '..')
processed_dir = os.path.join(amass_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)



def normalize_y_rotation(raw_theta):
    """
    rotate along y axis so that root rotation can always face the camera
    theta should be a [3] or [72] numpy array
    """
    only_global = True
    if raw_theta.shape == (72,):
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


def correct_global_rot(p):
    correction_mat = R.from_rotvec(np.array([-np.pi / 2, 0, 0])).as_matrix()
    rot_mat = cv2.Rodrigues(p[:3])[0]
    # correct the global rotation inconsistency
    rot_mat = correction_mat.dot(rot_mat)
    # convert to axis-angle
    th = R.from_matrix(rot_mat[None]).as_rotvec().reshape(-1)
    th = normalize_y_rotation(th)
    p[:3] = th
    return p


def process_split(split, step_size=100):
    datasets = amass_splits[split]
    np_data = []

    for ds_name in datasets:
        print('Processing dataset {}'.format(ds_name))
        subjects = glob.glob(os.path.join(amass_dir, ds_name, '*/'))
        for s in subjects:
            subj_name = ds_name + '/' + s
            print('\tProcessing subject {}'.format(subj_name))
            npz_fnames = glob.glob(os.path.join(s, '*_poses.npz'))

            for npz_fname in npz_fnames:
                print('\t\tProcessing motion sequence {}'.format(npz_fname))
                seq = process_single_seq(npz_fname, subj_name, step_size)
                if seq:
                    np_data.extend(seq)
    np_data = np.array(np_data, dtype=np_dtype)

    np.save(os.path.join(processed_dir, f"{split}.npy"), np_data)


def process_single_seq(npz_fname, subj_name, step_size, start=0.1, end=0.9):
    fdata = np.load(npz_fname)
    N = len(fdata['poses'])
    fids = list(range(int(start * N), int(end * N))[::step_size])
    if not fids:
        return None
    M = len(fids)
    gdr = np.array(gdr2num[str(fdata['gender'].astype(np.str))]).astype(np.int32)
    gdr = np.repeat(gdr[np.newaxis], repeats=M, axis=0)
    betas = np.repeat(fdata['betas'][np.newaxis], repeats=M, axis=0).astype(np.float32)
    pose = fdata['poses'][fids].astype(np.float32)
    pose = np.array([correct_global_rot(k) for k in pose], dtype=np.float32)
    # mesh = smpl2mesh(pose, betas, gdr).cpu().numpy().astype(np.float32).reshape([M, 6890*3])
    subj_name_np = np.tile(np.array(subj_name, dtype=np.dtype('U50'))[None], (M, 1))[:, 0]
    return list(zip(subj_name_np, gdr, betas, pose))



if __name__ == '__main__':
    process_split("train", 100)
    process_split("test", 100)
