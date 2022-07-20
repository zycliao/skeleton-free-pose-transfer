import torch
import numpy as np
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, QuaternionCoeffOrder
wxyz = QuaternionCoeffOrder.WXYZ
from utils.geometry import fps, arap_loss
from torch_scatter import scatter_sum
from torch.nn.functional import kl_div


def distChamfer(a, b, norm="L2"):
    """
    a: (B, N1, 3)
    b: (B, N2, 3)
    """
    x = a[:, :, None]
    y = b[:, None]
    if norm == "L1":
        dist = torch.abs(x - y)
    else:
        dist = torch.square(x - y)
    dist = torch.sum(dist, -1)
    return torch.mean(torch.min(dist, 2)[0]) + torch.mean(torch.min(dist, 1)[0])


def selfChamfer(a, norm="L2"):
    """
    a: (B, N1, 3)
    """
    N = a.shape[1]
    x = a[:, :, None]
    y = a[:, None]
    if norm == "L1":
        dist = torch.abs(x - y)
    else:
        dist = torch.square(x - y)
    dist = torch.sum(dist, -1)
    for i in range(N):
        dist[:, i, i] += 10000
    closest1 = torch.min(dist, 2)[0]
    closest2 = torch.min(dist, 1)[0]
    closest1[closest1>0.05] = 0
    closest2[closest2>0.05] = 0
    return torch.mean(closest1) + torch.mean(closest2)



def sw_loss(region_score, gt_weights, batch):
    """
    use GT skinning weights as supervision
    region_score: (B*V, K)
    gt_weights: list (V, K2) length: B
    """
    B = len(gt_weights)
    K = region_score.shape[1]
    all_loss = []
    for i in range(B):
        scores = region_score[batch==i] + 1e-6
        seg = torch.argmax(scores, 1).detach().cpu().numpy()
        gt_weight = gt_weights[i]
        if not isinstance(gt_weight, np.ndarray):
            gt_weight = np.array(gt_weight.todense())
        gt_seg = np.argmax(gt_weight, 1)
        gt_valid = np.where(np.max(gt_weight, 1) > 0.9)[0]
        p1, p2 = [], []
        for j in range(K):
            seg_idxs = np.where(seg == j)[0]
            np.random.shuffle(seg_idxs)
            if len(seg_idxs) < 2:
                continue
            half_num = len(seg_idxs) // 2
            p1.append(seg_idxs[:half_num])
            p2.append(seg_idxs[half_num: 2*half_num])
        p1 = np.concatenate(p1, 0)
        p2 = np.concatenate(p2, 0)
        valid_pidx = np.logical_and(np.isin(p1, gt_valid), np.isin(p2, gt_valid))
        p1 = p1[valid_pidx]
        p2 = p2[valid_pidx]
        same = gt_seg[p1] == gt_seg[p2]
        diff = np.logical_not(same)
        same = np.where(same)[0]
        diff = np.where(diff)[0]
        p1_score = scores[p1]
        p2_score = scores[p2]
        loss = kl_div(p1_score[same].log(), p2_score[same]) - kl_div(p1_score[diff].log(), p2_score[diff])
        if torch.isnan(loss):
            continue
        all_loss.append(loss)
    if len(all_loss) == 0:
        return torch.tensor(0).float().to(region_score.device)
    else:
        return torch.mean(torch.stack(all_loss))


def quaternion_loss(q1, q2):
    l1 = (q1 - q2).abs().mean(-1)
    l2 = (q1 + q2).abs().mean(-1)
    mask = (l1 < l2).float()
    loss = l1 * mask + l2 * (1-mask)
    return torch.mean(loss)


def transformation_loss(transformation, hm, region_score, batch, v0, v1, criterion):
    hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
    hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)  # (B, 40, 3)
    r_weight = 0.5
    with torch.no_grad():
        B, K, _ = hd0.shape
        seg = torch.max(region_score, 1)[1]  # (B*V,)
        R_batch = []
        for i in range(B):
            center0 = hd0[i]  # (40, 3)
            center1 = hd1[i]  # (40, 3)
            seg_single = seg[batch==i]  # (V,)
            v0_single = v0[batch==i]  # (V, 3)
            v1_single = v1[batch==i]  # (V, 3)
            seg_max = torch.max(seg_single) + 1

            scatter_center0 = center0[seg_single]
            scatter_center1 = center1[seg_single]

            v0_single = v0_single - scatter_center0
            v1_single = v1_single - scatter_center1

            M = torch.bmm(v0_single[:, :, None], v1_single[:, None])  # (V, 3, 3)
            M_sum = scatter_sum(M, seg_single, dim=0)  # (40, 3, 3)
            try:
                U, S, Vh = torch.linalg.svd(M_sum)
                # u, s, vh = np.linalg.svd(M_sum.cpu().numpy())
                # U = torch.from_numpy(u).float().to(M_sum.device)
                # S = torch.from_numpy(s).float().to(M_sum.device)
                # Vh = torch.from_numpy(vh).float().to(M_sum.device)
            except RuntimeError as e:
                print(e)
                r_weight = 0
                R_batch.append(torch.eye(3)[None].repeat(K, 1, 1).float().to(M_sum.device))
                continue
            R_temp = torch.bmm(U, Vh).permute(0, 2, 1)  # (40, 3, 3)
            R = torch.eye(3)[None].repeat(K, 1, 1).float().to(R_temp.device)
            R[:seg_max] = R_temp
            R = rotation_matrix_to_quaternion(R, order=wxyz)
            R_batch.append(R)

        R_batch = torch.stack(R_batch, 0)  # (B, 40, 3, 3)

        # R_batch = R_batch.detach().cpu().numpy()
        # Rs = R_batch.reshape([-1, 3, 3])
        # R_batch = []
        # for R in Rs:
        #     R_batch.append(cv2.Rodrigues(R)[0])
        # R_batch = np.stack(R_batch, 0)
        # R_batch = R_batch.reshape([B, K, 3])
        # R_batch = torch.from_numpy(R_batch).float().to(hd0.device)

        t_batch = hd1 - hd0
    pred_t = transformation[:, :, :3]
    pred_R = transformation[:, :, 3:]
    if r_weight == 0:
        return criterion(t_batch, pred_t)
    else:
        return quaternion_loss(R_batch, pred_R) * r_weight + criterion(t_batch, pred_t)


def get_transformation(hm, region_score, batch, v0, v1):
    with torch.no_grad():
        hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
        hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)  # (B, 40, 3)

        B, K, _ = hd0.shape
        seg = torch.max(region_score, 1)[1]  # (B*V,)
        R_batch = []
        for i in range(B):
            center0 = hd0[i]  # (40, 3)
            center1 = hd1[i]  # (40, 3)
            seg_single = seg[batch==i]  # (V,)
            v0_single = v0[batch==i]  # (V, 3)
            v1_single = v1[batch==i]  # (V, 3)
            seg_max = torch.max(seg_single) + 1

            scatter_center0 = center0[seg_single]
            scatter_center1 = center1[seg_single]

            v0_single = v0_single - scatter_center0
            v1_single = v1_single - scatter_center1

            M = torch.bmm(v0_single[:, :, None], v1_single[:, None])  # (V, 3, 3)
            M_sum = scatter_sum(M, seg_single, dim=0)  # (40, 3, 3)
            U, S, Vh = torch.linalg.svd(M_sum)
            R_temp = torch.bmm(U, Vh).permute(0, 2, 1)  # (40, 3, 3)
            R = torch.eye(3)[None].repeat(K, 1, 1).float().to(R_temp.device)
            R[:seg_max] = R_temp
            R = rotation_matrix_to_quaternion(R, order=wxyz)
            R_batch.append(R)

        R_batch = torch.stack(R_batch, 0)  # (B, 40, 4)
        t_batch = hd1 - hd0
    return torch.cat((t_batch, R_batch), 2)


def handle2mesh(transformation, handle_pos, region_score, batch, v0):
    """
    use per-part trans+rot to reconstruct the mesh
    transformation: (B, 40, 3+4)
    handle_pos: handle position of T-pose
    handle_pos: (B, 40, 3)
    region_score: (B*V, 40)
    v0: (B*V, 3)
    """
    B, K, _ = handle_pos.shape
    disp = transformation[:, :, :3]
    rot = transformation[:, :, 3:]
    rot = quaternion_to_rotation_matrix(rot.view(B*K, 4).contiguous(), order=wxyz).view(B, K, 3, 3).contiguous()
    hd_disp = torch.repeat_interleave(disp, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    hd_rot = torch.repeat_interleave(rot, torch.bincount(batch), dim=0)  # (B*V, 40, 3, 3)
    hd_pos = torch.repeat_interleave(handle_pos, torch.bincount(batch), dim=0)  # (B*V, 40, 3)
    per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
    v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
    return v


def torch_geometric_fps(x, batch, K):
    with torch.no_grad():
        N = torch.max(batch)+1 # batch size
        N = N.cpu().item()
        batch_pts = []
        batch_idx = []
        for i in range(N):
            v = x[batch == i]
            pts, idx = fps((v, K, None))
            batch_pts.append(pts)
            batch_idx.append(idx)
    batch_pts = torch.stack(batch_pts, 0)
    return batch_pts


def gt_trans_mesh(pred_disp, hm, region_score, batch, v0, v1):
    with torch.no_grad():
        hd0 = scatter_sum(hm[:, :, None] * v0[:, None], batch, dim=0)
        hd1 = scatter_sum(hm[:, :, None] * v1[:, None], batch, dim=0)
        gt_trans = hd1 - hd0  # (B, K, 3)
    pred_rot = pred_disp[:, :, 3:]  # (B, K, 4)
    return handle2mesh(torch.cat((gt_trans, pred_rot), 2), hd0, region_score, batch, v0)


def arap_smooth(trans, center, sw, batch, v0, tpl_edge_index, arap_weight=-1):
    if arap_weight <= 0:
        return trans
    torch.set_grad_enabled(True)
    device = trans.device
    pred_v0 = handle2mesh(trans, center, sw, batch, v0)
    pred_trans = trans.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pred_trans], lr=3e-2)
    l1 = torch.nn.L1Loss()

    max_iter = 300
    min_i, min_loss = -1, 1000000
    for i in range(max_iter):
        optimizer.zero_grad()
        pred_v = handle2mesh(pred_trans, center, sw, batch, v0)
        loss_data = l1(pred_v, pred_v0)
        loss_arap = arap_loss(tpl_edge_index, pred_v, v0) * 100
        loss = loss_data + loss_arap * arap_weight
        loss.backward()
        loss_ = loss.detach().cpu().item()
        # print(f"iter {i}, data loss {loss_data.detach().cpu().item():.6f}"
        #       f", arap loss {loss_arap.detach().cpu().item():.6f}")
        optimizer.step()
        if loss_ < min_loss:
            min_loss = loss_
            min_i = i
        if i - min_i >= 10:
            break
    torch.set_grad_enabled(False)
    return pred_trans