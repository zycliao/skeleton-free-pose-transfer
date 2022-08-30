import numpy as np
from tqdm import tqdm
from global_var import *
from utils.bvh import Bvh
from scipy.spatial.transform import Rotation


if __name__ == '__main__':
    motion_dir = os.path.join(MIXAMO_SIMPLIFY_PATH, 'BVH')
    save_dir = os.path.join(MIXAMO_SIMPLIFY_PATH, 'motion')
    os.makedirs(save_dir, exist_ok=True)
    motion_names = [k for k in os.listdir(motion_dir) if k != 'skin.bvh' and k.endswith('.bvh')]
    for name in tqdm(motion_names):
        save_path = os.path.join(save_dir, name.replace('.bvh', '.npy'))
        mocap = Bvh(os.path.join(motion_dir, name))
        frames = mocap.frames
        n_frame, c_frame = frames.shape
        frames = frames.reshape([n_frame, -1, 3])
        frames = frames[:, 1:]

        bvh_joint_names = []
        for name in mocap.get_joints_names():
            if ":" in name:
                name = name.split(":")[1]
            bvh_joint_names.append(name)
        joint_idxs = []
        for name in MIXAMO_JOINTS:
            joint_idxs.append(bvh_joint_names.index(name))
        frames = frames[:, joint_idxs]

        thetas = Rotation.from_euler('XYZ', angles=frames.reshape([-1, 3]), degrees=True).as_rotvec()
        thetas = thetas.reshape([n_frame, -1, 3])
        np.save(save_path, thetas)
