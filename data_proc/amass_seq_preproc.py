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


test_sequence = [["seq_01", "MPI_HDM05/bk/HDM_bk_05-03_01_120", [0.1, 0.3]],
                 ["seq_02", "MPI_mosh/00096/stretches", [0.1, 0.5]],
                 ["seq_03", "Transitions_mocap/mazen_c3d/devishdance_kick", [0.2, 0.6]],
                 ["seq_04", "DanceDB/20180501_CliodelaVara/CLIO_Roditikos", [0.36, 0.4]],
                 ["seq_05", "CMU/01/01_10", [0.05, 0.3]],
                 ["seq_06", "CMU/15/15_10", [0.6, 0.75]],
                 ["seq_07", "CMU/29/29_15", [28/37, 34/37]],
                 ["seq_08", "CMU/54/54_01", [5/16, 13/16]],
                 ["seq_09", "CMU/54/54_12", [10/19, 16/19]],
                 ["seq_10", "CMU/64/64_05", [0, 1]],
                 ["seq_11", "CMU/64/64_23", [0, 1]],

                 ["seq_20", ["CMU/25/25_01", "CMU/26/26_05", "CMU/31/31_08", "CMU/105/105_05", "CMU/111/111_17", "CMU/142/142_14"],
                  [[0.72, 0.92], [0.65, 0.8], [0.02, 0.07], [0.36, 0.43], [0.37, 0.75], [0.54, 0.67]] ],

                 ["seq_30", "DanceDB/20120807_VasoAristeidou/Vasso_Reggaeton_01", [0.588, 0.696]],
                 ["seq_31", "DanceDB/20130216_AnnaCharalambous/Anna_Angry_C3D", [20/62, 35/62]],
                 ["seq_32", "SSM_synced/20160330_03333/ATU_001", [0, 0.8]],
                 ["seq_33", "Transitions_mocap/mazen_c3d/dance_running", [1/8, 5/8]],
                 ["seq_34", "MPI_HDM05/bk/HDM_bk_05-01_01_120", [17/46, 30/46]],
                 ["seq_35", "MPI_HDM05/bk/HDM_bk_03-02_01_120", [15/41, 30/41]],
                 ["seq_36", "DanceDB/20180501_AndreasdelaVara/ANDREAS_Hasaposerviko", [60/105, 75/105]],
                 ['seq_37', "DanceDB/20151003_ElenaKyriakou/Elena_Happy_v1_C3D", [2/17, 15/17]]
                 ]


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
    # process_split("test", 100)
    # process_split("test", 200)

    for save_name, input_name, interval in test_sequence[-4:]:
        if isinstance(input_name, list):
            np_data = []
            for name, intv in zip(input_name, interval):
                data = process_single_seq(os.path.join(amass_dir, name+'_poses.npz'), name, 2, intv[0], intv[1])
                np_data.extend(data)
        else:
            np_data = process_single_seq(os.path.join(amass_dir, input_name+'_poses.npz'), input_name, 2, interval[0], interval[1])
        if np_data:
            np_data = np.array(np_data, dtype=np_dtype)
            save_path = os.path.join(processed_dir, f'{save_name}.npy')
            np.save(save_path, np_data)
            print(f"{save_path} saved")
