import torch
import numpy as np
from smplx.lbs import batch_rigid_transform as batch_rigid_transform_torch
from scipy.spatial.transform import Rotation as rot


def batch_rigid_transform(
    rot_mats,
    joints,
    parents
):
    posed_joints, rel_transforms = batch_rigid_transform_torch(torch.from_numpy(rot_mats).float(),
                                torch.from_numpy(joints).float(),
                                torch.from_numpy(parents).float())
    return posed_joints.numpy(), rel_transforms.numpy()


def lbs_batch(v, rotations, J, parents, weights, input_mat=False):
    """
    Args:
        v: (B, V, 3)
        rotations: (B, J, 3), rotation vector
        J: (B, J, 3), joint positions
        parents: (J), kinematic chain indicator
        weights: (B, V, J), skinning weights

    Returns:
        articulated vertices: (B, V, 3)
    """
    B, num_joints, _ = J.shape
    if input_mat:
        rot_mats = rotations.reshape([B, num_joints, 3, 3])
    else:
        rot_mats = rot.from_rotvec(rotations.reshape([-1, 3])).as_matrix().reshape([B, num_joints, 3, 3])
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)

    W = np.tile(weights[None], [B, 1, 1])
    T = np.einsum("bnj,bjpq->bnpq", W, A)

    R = T[:, :, :3, :3]
    t = T[:, :, :3, 3]
    verts = np.einsum("bnpq,bnq->bnp", R, v) + t
    return verts, J_transformed, T


def lbs(v, rotations, J, parents, weights, input_mat=False, verbose=False):
    """
    Args:
        v: (V, 3)
        rotations: (J, 3), rotation vector
        J: (J, 3), joint positions
        parents: (J), kinematic chain indicator
        weights: (V, J), skinning weights
        input_mat: input rotation is rotation matrix (otherwise, axis angle)

    Returns:
        articulated vertices: (V, 3)
    """

    verts, J_transformed, T = lbs_batch(v[None], rotations[None], J[None], parents, weights[None], input_mat)
    if verbose:
        return verts[0], J_transformed[0], T[0]
    else:
        return verts[0]

