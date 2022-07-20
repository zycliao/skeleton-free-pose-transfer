import numpy as np
import torch
from torch import Tensor
from smplx.lbs import batch_rodrigues, batch_rigid_transform


def fk(
    pose: Tensor,
    v_template: Tensor,
    J: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
):
    batch_size = 1
    pose = pose[None]
    v_template = v_template[None]
    J = J[None]

    device, dtype = pose.device, pose.dtype

    # 3. rotation matrices
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J.shape[1]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_template.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_template, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed

def ik(target_v, v_template, joints, parents, weights,
       device, pose=None, debug=False, max_iter=300):
    # device = torch.device("cpu")
    with torch.set_grad_enabled(True):
        num_joints = joints.shape[0]

        if isinstance(v_template, np.ndarray):
            v_template = torch.from_numpy(v_template)
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints)
        if isinstance(parents, np.ndarray):
            parents = torch.from_numpy(parents)
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        if isinstance(target_v, np.ndarray):
            target_v = torch.from_numpy(target_v)

        v_template = v_template.detach().to(device).float()
        target_v = target_v.detach().to(device).float()
        joints = joints.detach().to(device).float()
        parents = parents.detach().to(device).long()
        weights = weights.detach().to(device).float()

        # parents_np = parents.detach().cpu().numpy()
        # depth = np.zeros_like(parents_np)
        # for idx, p_idx in enumerate(parents_np):
        #     if p_idx == -1:
        #         depth[idx] = 0
        #     else:
        #         depth[idx] = depth[p_idx] + 1
        # max_depth = np.max(depth)

        if pose is None:
            pose = torch.zeros(num_joints, 3, dtype=torch.float32, device=device, requires_grad=True)
        else:
            pose = torch.tensor(torch.from_numpy(pose), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([pose], lr=3e-2)
        mse = torch.nn.MSELoss()
        l1 = torch.nn.L1Loss()
        iter_num = 300
        # for d in range(max_depth+1):
        verts_record = []
        min_i, min_loss = -1, 1000000
        for i in range(max_iter):
            optimizer.zero_grad()
            verts, _ = fk(pose, v_template, joints, parents, weights)
            verts = verts[0]
            data_loss = l1(verts, target_v)
            loss = data_loss
            loss.backward()
            loss_ = loss.detach().cpu().item()
            # print(f"iter {i}, data loss {data_loss.detach().cpu().item():.6f}")
            optimizer.step()
            if loss_ < min_loss:
                min_loss = loss_
                min_i = i
            if i - min_i >= 10:
                break
            if debug:
                verts_record.append(verts.detach().cpu().numpy())
    if debug:
        return pose.detach().cpu().numpy(), verts.detach().cpu().numpy(), verts_record
    else:
        return pose.detach().cpu().numpy(), verts.detach().cpu().numpy()


def ik_single(data, handle_idx, handle_pos, max_iter=300, refine=True, j=0, pose=None, device=None):
    '''
    data: data loaded by RigNet dataset
    handle_pos: (K, 3)
    batch: if true, data is loaded by DataLoader as a batch
    '''
    if device is None:
        device = data.v0.device

    if pose is not None:
        pose = pose[None]
    v0 = data.v0[data.ptr[j]: data.ptr[j+1]]
    pose, verts = ik(v0, data.joints[j][0], data.parents[j][0], data.weights[j].todense(),
                     handle_idx, handle_pos[None], device=device, pose=pose, max_iter=max_iter, refine=refine)
    return verts[0], pose[0]


def ik_verts(v_template, joints, parents, weights,
             target_v, device, pose=None, debug=False, max_iter=300, refine=True):
    # device = torch.device("cpu")
    with torch.set_grad_enabled(True):
        B = 1
        num_joints = joints.shape[0]

        if isinstance(v_template, np.ndarray):
            v_template = torch.from_numpy(v_template)
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints)
        if isinstance(parents, np.ndarray):
            parents = torch.from_numpy(parents)
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        if isinstance(target_v, np.ndarray):
            target_v = torch.from_numpy(target_v)

        v_template = v_template.detach().to(device)
        joints = joints.detach().to(device)
        parents = parents.detach().to(device).long()
        weights = weights.detach().to(device)
        target_v = target_v.detach().to(device)

        # parents_np = parents.detach().cpu().numpy()
        # depth = np.zeros_like(parents_np)
        # for idx, p_idx in enumerate(parents_np):
        #     if p_idx == -1:
        #         depth[idx] = 0
        #     else:
        #         depth[idx] = depth[p_idx] + 1
        # max_depth = np.max(depth)

        if pose is None:
            pose = torch.zeros(B, num_joints, 3, dtype=torch.float32, device=device, requires_grad=True)
        else:
            pose = torch.tensor(torch.from_numpy(pose), dtype=torch.float32, device=device, requires_grad=True)
        # pose = torch.from_numpy(np.random.normal(scale=0.1, size=(B, num_joints, 3))).float()
        optimizer = torch.optim.Adam([pose], lr=1e-1)
        mse = torch.nn.MSELoss()
        l1 = torch.nn.L1Loss()
        iter_num = 300
        # for d in range(max_depth+1):
        verts_record = []
        min_i, min_loss = -1, 1000000
        for i in range(max_iter):
            optimizer.zero_grad()
            verts, _ = fk(pose, v_template, joints, parents, weights)
            data_loss = mse(verts, target_v)
            reg_loss = 0.01*l1(pose, 0*pose)
            loss = data_loss + reg_loss
            loss.backward()
            loss_ = loss.detach().cpu().item()
            # print(f"iter {i}, data loss {data_loss.detach().cpu().item():.6f}, reg {reg_loss.detach().cpu().item():.8f}")
            optimizer.step()
            if loss_ < min_loss:
                min_loss = loss_
                min_i = i
            if i - min_i >= 10:
                break
            if debug:
                verts_record.append(verts.detach().cpu().numpy())
        if refine:
            optimizer.param_groups[0]['lr'] = 1e-2
            for i in range(100):
                optimizer.zero_grad()
                verts, _ = fk(pose, v_template, joints, parents, weights)
                data_loss = l1(verts, target_v)
                reg_loss = 0.03*l1(pose, 0*pose)
                loss = data_loss + reg_loss
                loss.backward()
                loss_ = loss.detach().cpu().item()
                # print(f"iter {i}, data loss {data_loss.detach().cpu().item():.6f}, reg {reg_loss.detach().cpu().item():.8f}")
                optimizer.step()
                if loss_ < min_loss:
                    min_loss = loss_
                    min_i = i
                if i - min_i >= 10:
                    break
                if debug:
                    verts_record.append(verts.detach().cpu().numpy())
    if debug:
        return pose.detach().cpu().numpy(), verts.detach().cpu().numpy(), verts_record
    else:
        return pose.detach().cpu().numpy(), verts.detach().cpu().numpy()


def ik_single_verts(data, target_v, max_iter=300, refine=True, j=0, pose=None, device=None):
    '''
    data: data loaded by RigNet dataset
    handle_pos: (K, 3)
    batch: if true, data is loaded by DataLoader as a batch
    '''
    if device is None:
        device = data.v0.device

    if pose is not None:
        pose = pose[None]
    v0 = data.v0[data.ptr[j]: data.ptr[j+1]]
    pose, verts = ik_verts(v0, data.joints[j][0], data.parents[j][0], data.weights[j].todense(),
                     target_v, device=device, pose=pose, max_iter=max_iter, refine=refine)
    return verts[0], pose[0]

