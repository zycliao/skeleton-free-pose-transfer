import cv2
import numpy as np
import trimesh
import torch
from scipy.spatial.transform import Rotation
from utils.render import merge_mesh
from utils.o3d_wrapper import Mesh


COLORS_HSV = np.zeros([64, 3])
ci = 0
for s in [170, 100]:
    for h in np.linspace(0, 179, 16):
        for v in [220, 128]:
            COLORS_HSV[ci] = [h, s, v]
            ci += 1
COLORS = cv2.cvtColor(COLORS_HSV[None].astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float) / 255
COLORS_ALPHA = np.concatenate((COLORS, np.ones_like(COLORS[:, :1])), 1)



def visualize_handle(v, f, handle_pos=None, heatmap=None, save_path=None, norm=True):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    if isinstance(handle_pos, torch.Tensor):
        handle_pos = handle_pos.detach().cpu().numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    cube_v = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1],
                       [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1],], dtype=np.float32) * 0.01
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [7, 4, 6],
                       [2, 5, 1], [2, 6, 5], [0, 4, 3], [4, 7, 3],
                       [1, 5, 0], [5, 4, 0], [3, 6, 2], [3, 7, 6]])
    cube_c = np.tile(np.array([[2, 0.3, 0.3]]), (len(cube_v), 1))
    if heatmap is None:
        base_c = np.ones([v.shape[0], 4])
        if handle_pos is not None:
            base_c[:, 3] = 0.5
    else:
        if norm:
            heatmap = heatmap / np.max(heatmap)
        heatmap = heatmap * 10 - 5
        heatmap = 1/(np.exp(-heatmap)+1) # sigmoid
        heatmap = (1 - heatmap) * 120
        heatmap_hsv = np.ones([heatmap.shape[0], 3], dtype=np.uint8) * 255
        heatmap_hsv[:, 0] = heatmap.astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_hsv[None], cv2.COLOR_HSV2BGR)[0]
        base_c = heatmap_bgr.astype(float) / 255
        base_c = np.concatenate((base_c, np.ones_like(base_c[:, :1])*0.5), 1)
    vv, ff, vcc = [v],  [f], [base_c]
    if handle_pos is not None:
        for i, hp in enumerate(handle_pos):
            vv.append(cube_v + hp[None])
            ff.append(cube_f)
            if len(handle_pos) > 1:
                vcc.append(COLORS_ALPHA[i%64])
            else:
                vcc.append(np.array([1., 1., 1., 1.]))
    if save_path is None:
        return vv, ff, vcc
    else:
        vv, ff, vcc = merge_mesh(vv, ff, vcc)
        Mesh(v=vv, f=ff, vc=vcc[:, :3]).write_ply(save_path)
        return vv, ff, vcc



def visualize_part(v, f, handle_pos, score, save_path=None):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    if isinstance(handle_pos, torch.Tensor):
        handle_pos = handle_pos.detach().cpu().numpy()
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    if not isinstance(score, np.ndarray):
        score = np.array(score.todense())
    cube_v = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1],
                       [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1],], dtype=np.float32) * 0.01
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [7, 4, 6],
                       [2, 5, 1], [2, 6, 5], [0, 4, 3], [4, 7, 3],
                       [1, 5, 0], [5, 4, 0], [3, 6, 2], [3, 7, 6]])
    cube_c = np.tile(np.array([[2, 0.3, 0.3]]), (len(cube_v), 1))

    num_part = score.shape[1]
    if num_part >= 64:
        colors = np.concatenate((COLORS_ALPHA, COLORS_ALPHA), 0)[:num_part]
    else:
        colors = COLORS_ALPHA[:num_part]
    base_c = np.sum(score[:, :, None] * colors[None], 1)
    # seg = np.argmax(score, 1)
    # base_c = COLORS_ALPHA[seg]
    base_c[:, 3] = 0.5

    vv, ff, vcc = [v],  [f], [base_c]
    if handle_pos is not None:
        for i, hp in enumerate(handle_pos):
            vv.append(cube_v + hp[None])
            ff.append(cube_f)
            if len(handle_pos) > 1:
                vcc.append(COLORS_ALPHA[i%64])
            else:
                vcc.append(np.array([1., 1., 1., 1.]))
    if save_path is None:
        return vv, ff, vcc
    else:
        vv, ff, vcc = merge_mesh(vv, ff, vcc)
        Mesh(v=vv, f=ff, vc=vcc[:, :3]).write_ply(save_path)
        return vv, ff, vcc



def visualize_gaussian(v, f, param, score=None, save_path=None):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    if isinstance(param, torch.Tensor):
        param = param.detach().cpu().numpy()
    if score is not None and isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    param = param.reshape([-1, 10])
    pos_ = param[:, :3]
    quat = param[:, 3:7]
    scale = param[:, 7:10]
    sphere = trimesh.creation.uv_sphere(radius=0.05, count=[8, 8])
    num_part = param.shape[0]

    if score is None:
        base_c = np.ones([v.shape[0], 4])
        base_c[:, 3] = 0.4
    else:
        seg = np.argmax(score, 1)
        base_c = COLORS_ALPHA[seg]
        base_c[:, 3] = 0.4
    vv, ff, vcc = [],  [], []
    for i in range(num_part):
        sphere_v = np.copy(sphere.vertices)
        sphere_v = sphere_v / np.exp(0.5*scale[i][None])
        sphere_v = np.einsum("ab,nb->na", Rotation.from_quat(quat[i]).as_matrix().T, sphere_v)
        sphere_v = sphere_v + pos_[i][None]
        vv.append(sphere_v)
        ff.append(sphere.faces)
        vcc.append(np.tile(COLORS_ALPHA[i][None], (len(sphere_v), 1)))
    vv.append(v)
    ff.append(f)
    vcc.append(base_c)
    if save_path is None:
        return vv, ff, vcc
    else:
        vv, ff, vcc = merge_mesh(vv, ff, vcc)
        Mesh(v=vv, f=ff, vc=vcc[:, :3]).write_ply(save_path)
        return vv, ff, vcc
