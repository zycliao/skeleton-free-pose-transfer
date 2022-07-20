import igl
import numpy as np
import open3d as o3d
import torch
from torch_scatter import scatter_add
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay
import torch_geometric as tg
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
from torch.nn.functional import kl_div
from networkx.algorithms.components import connected_components
from utils.o3d_wrapper import MeshO3d


calc_distances = lambda p0, pts: ((p0 - pts) ** 2).sum(dim=1)


def fps_np(pts, K, init_pts=None):
    pts = torch.from_numpy(pts)
    pts_sampled, farthest_idx = fps((pts, K, init_pts))
    return pts_sampled.numpy(), farthest_idx.numpy()


def fps(x):
    pts, K, init_pts = x
    farthest_idx = torch.LongTensor(K)
    farthest_idx.zero_()
    if init_pts is None:
        farthest_idx[0] = np.random.randint(len(pts))
    else:
        farthest_idx[0] = init_pts
    distances = calc_distances(pts[farthest_idx[0]], pts)
    for i in range(1, K):
        farthest_idx[i] = torch.max(distances, dim=0)[1]
        farthest_pts = pts[farthest_idx[i]]
        distances = torch.min(distances, calc_distances(farthest_pts, pts))
    pts_sampled = pts[farthest_idx, :]
    return pts_sampled, farthest_idx


def batch_fps(pts, K, init_pts=None):
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    fps_res = list(map(fps, [(pts[i].to('cpu'), K, init_pts[i]) for i in range(len(pts))]))
    batch_pts = [i[0] for i in fps_res]
    batch_pts = torch.stack(batch_pts, dim=0).to(pts[0].device)
    batch_id = [i[1] for i in fps_res]
    batch_id = torch.stack(batch_id, dim=0).long().to(pts[0].device)
    return batch_pts, batch_id


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


def geodesic_fps(v, f, init_pts, K, geo_dist=None):
    """
    fps on the mesh (based on geodesic distance)
    Args:
        v: vertices (V, 3)
        f: faces (F, 3)
        init_pts: index of initial points for farthest point sampling (N,)
        K: int. how many points to sample
    """
    N = v.shape[0]
    if geo_dist is None:
        conn_matrix = lil_matrix((N, N), dtype=np.float32)
        for tri in f:
            conn_matrix[tri[0], tri[1]] = 1
            conn_matrix[tri[1], tri[2]] = 1
            conn_matrix[tri[2], tri[0]] = 1
        [geo_dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                                   return_predecessors=True, unweighted=True)
    inf_pos = np.argwhere(np.isinf(geo_dist))
    if len(inf_pos) > 0:
        geo_dist[inf_pos[:, 0], inf_pos[:, 1]] = 1e10
    # farthest point sampling
    if not init_pts:
        init_pts = [np.random.randint(N)]
    min_dist = np.min(geo_dist[init_pts], 0)
    sample_idx = []
    for i in range(K):
        sample_idx.append(np.argmax(min_dist))
        min_dist = np.minimum(geo_dist[sample_idx[-1]], min_dist)
    return sample_idx, geo_dist



def merge_mesh(vs, fs, vcs):
    v_num = 0
    new_fs = [fs[0]]
    new_vcs = []
    for i in range(len(vs)):
        if i >= 1:
            v_num += vs[i-1].shape[0]
            new_fs.append(fs[i]+v_num)
        if vcs is not None:
            if vcs[i].ndim == 1:
                new_vcs.append(np.tile(np.expand_dims(vcs[i], 0), [vs[i].shape[0], 1]))
            else:
                new_vcs.append(vcs[i])
    vs = np.concatenate(vs, 0)
    new_fs = np.concatenate(new_fs, 0)
    if vcs is not None:
        new_vcs = np.concatenate(new_vcs, 0)
    return vs, new_fs, new_vcs


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_normal_np(v, f):
    m = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v.astype(np.float32)),
                                             triangles=o3d.utility.Vector3iVector(f.astype(np.int32)))
    m.compute_vertex_normals()
    vn = np.asarray(m.vertex_normals)
    return vn


def arap(v, f, handle_idx, handle_pos, backend='o3d'):
    if backend == 'o3d':
        m = MeshO3d(v=v, f=f)
        constraint_ids = o3d.utility.IntVector(handle_idx.astype(np.int32))
        constraint_pos = o3d.utility.Vector3dVector(handle_pos.astype(np.float32))
        m.m = m.m.deform_as_rigid_as_possible(
            constraint_ids, constraint_pos, max_iter=10)
        return m.v
    elif backend == 'igl':
        solver = igl.ARAP(v, f, 3, handle_idx)
        return solver.solve(handle_pos, v)


def count_component(v, f):
    # return value: a list containing the indexes in each component
    edge_idx = get_tpl_edges(v, f).T
    d = tg.data.Data(x=torch.from_numpy(v), edge_index=torch.from_numpy(edge_idx).long())
    graph = tg.utils.to_networkx(d, to_undirected=True, remove_self_loops=True)
    comps = connected_components(graph)
    return [k for k in comps]


def keep_larget_component(v, f):
    # return value: a list containing the indexes in each component
    edge_idx = get_tpl_edges(v, f).T
    d = tg.data.Data(x=torch.from_numpy(v), edge_index=torch.from_numpy(edge_idx).long())
    graph = tg.utils.to_networkx(d, to_undirected=True, remove_self_loops=True)
    comps = connected_components(graph)
    return [k for k in comps]


def point2mesh_dist(v0, v, f):
    # the distance between points (N1,3) and all faces of a mesh (N2,3) (F,3)
    # return (N1, F)
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]
    n = np.cross(p1-p0, p2-p0)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    dist = np.abs(np.sum((v0[:, None] - p0[None]) * n[None], 2))
    return dist


# def get_nearest_face(v0, v, f):
#     # v0 (N1, 3)
#     # v (N2, 3), f (F, 3)
#     a = [[] for k in range(len(v))]
#     for fi, tri in enumerate(f):
#         for t in tri:
#             a[t].append(fi)
#
#     p2p_dist = np.sum(np.square(v0[:, None] - v[None]), 2) # (N1, N2)
#     nearest_v = np.argsort(p2p_dist, 1)[:, :3] # (N1, 3)
#     ret = np.zeros([len(v0)], dtype=int)
#     for vi, nv in enumerate(nearest_v):
#         nearest_f = []
#         for ni in nv:
#             nearest_f.extend(a[ni])
#         nearest_f_cnt = {}
#         for fi in nearest_f:
#             if fi in nearest_f_cnt:
#                 nearest_f_cnt[fi] += 1
#             else:
#                 nearest_f_cnt[fi] = 1
#         max_cnt = 0
#         max_idx = -1
#         for fi, cnt in nearest_f_cnt.items():
#             if cnt > max_cnt:
#                 max_cnt = cnt
#                 max_idx = fi
#         ret[vi] = max_idx
#     return ret


def get_nearest_face(v0, v, f):
    tri_verts = v[f.reshape(-1)].reshape([-1, 3, 3])
    tri_centroid = np.mean(tri_verts, 1)
    dist = np.sum(np.square(v0[:, None] - tri_centroid[None]), 2)
    return np.argmin(dist, 1)


def barycentric(p, a, b, c):
    # p (N, 3)
    # a, b, c (N, 3), three vertices of the triangle
    # return (N, 3), the barycentric coordinates
    v0, v1, v2 = b-a, c-a, p-a
    d00 = np.sum(v0*v0, 1)
    d01 = np.sum(v0*v1, 1)
    d11 = np.sum(v1*v1, 1)
    d20 = np.sum(v2*v0, 1)
    d21 = np.sum(v2*v1, 1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    u = np.clip(1.0 - v - w, 0, 1)
    v = 1.0 - u - w
    return np.stack((u, v, w), 1)


def part_scaling(v, tpl_edge_index, weights, joints, joint_idxs, scale):
    edges = tpl_edge_index.T
    edges = edges[edges[:, 0]<edges[:, 1]]  # remove repeat

    v_idx = np.where(np.sum(weights[:, joint_idxs], 1) > 0.1)[0]
    is_edge_in = np.isin(edges, v_idx)
    connect_edge = np.logical_xor(is_edge_in[:, 0], is_edge_in[:, 1])
    edges = edges[connect_edge]  # edges of interest
    is_edge_in = is_edge_in[connect_edge]

    edges[is_edge_in[:, 1]] = edges[is_edge_in[:, 1]][:, ::-1]  # swap the order, so that edges[:, 0] are transformed vertices
    if edges.size == 0:
        return v, joints
    gt_mean_disp = np.mean(v[edges[:, 0]], 0, keepdims=True)

    scaled_v = v * scale
    disp = gt_mean_disp - np.mean(scaled_v[edges[:, 0]], 0, keepdims=True)
    deformed = v.copy()
    deformed[v_idx] = scaled_v[v_idx] + disp

    if joints is not None:
        scaled_joints = joints * scale
        deformed_joints = joints.copy()
        deformed_joints[joint_idxs] = scaled_joints[joint_idxs] + disp
    else:
        deformed_joints = None
    return deformed, deformed_joints


def arap_loss(edge_index, x1, x2, weights=None):
    # edge_index: (2, num_edge)
    # x1, x2: (B*V, 3)
    # weights: a list of (V, J)
    if weights is not None:
        with torch.no_grad():
            rigid = []
            for w in weights:
                w = np.max(w.todense(), 1) > 0.9
                rigid.append(w)
            rigid = np.concatenate(rigid, 0)  # (B*V)
            rigid = torch.from_numpy(rigid).to(edge_index.device)
            rigid = torch.nonzero(rigid)
            edge_mask = torch.isin(edge_index, rigid).all(dim=0)
            edge_index = edge_index[:, edge_mask]

    diff1 = (x1[edge_index[0]] - x1[edge_index[1]]).pow(2).sum(1)
    diff2 = (x2[edge_index[0]] - x2[edge_index[1]]).pow(2).sum(1)
    diff = (diff1-diff2).abs()
    if diff.size() == 0:
        return torch.tensor([0]).to(x1.device).float()
    loss = torch.mean(diff)
    return loss


def laplacian_loss(v, edge_index):
    device = v.device
    with torch.no_grad():
        edge_index, _ = remove_self_loops(edge_index)
        edge_weight = torch.ones(edge_index.size(1), device=device).float()
        rows, cols = edge_index[0], edge_index[1]
        num_nodes = torch.max(edge_index) + 1
        deg = scatter_add(edge_weight, rows, dim=0, dim_size=num_nodes)
        vals = 1 / deg[rows]
        seqs = torch.arange(0, num_nodes).to(device)
        rows = torch.cat((rows, seqs), 0)
        cols = torch.cat((cols, seqs), 0)
        vals = torch.cat((vals, -1 * torch.ones(num_nodes).to(device).float()), 0)
        L = torch.sparse_coo_tensor(torch.stack((rows, cols), 0), vals, (num_nodes, num_nodes))
    disp = torch.sparse.mm(L, v)
    # return loss
    return torch.mean(torch.square(disp))



def kl_divergence(a, b, **args):
    return kl_div((a+1e-7).log(), b+1e-7, **args)


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
        gt_weight = np.array(gt_weights[i].todense())
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
        loss = kl_divergence(p1_score[same], p2_score[same]) - kl_divergence(p1_score[diff], p2_score[diff])
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        all_loss.append(loss)
    if len(all_loss) == 0:
        return torch.tensor(0).float().to(region_score.device)
    else:
        return torch.mean(torch.stack(all_loss))


def sw_loss2(region_score, gt_weights, batch):
    """
    use GT skinning weights as supervision
    region_score: (B*V, K)
    gt_weights: list (V, K2) length: B
    """
    B = len(gt_weights)
    dev = region_score.device
    all_loss = []
    for i in range(B):

        scores = region_score[batch==i] + 1e-7

        gt_weight = np.array(gt_weights[i].todense())
        gt_weight = torch.from_numpy(gt_weight).float().to(dev)

        scores = scores / torch.sum(scores, 0, keepdim=True)  # (V, K1)
        gt_weight = gt_weight / torch.sum(gt_weight, 0, keepdim=True)  # (V, K2)
        K1 = scores.shape[1]
        K2 = gt_weight.shape[1]
        scores = scores.T[None].repeat(K2, 1, 1)  # (K2, K1, V)
        gt_weight = gt_weight.T[:, None].repeat(1, K1, 1)  # (K2, K1, V)
        div = kl_divergence(scores, gt_weight, reduction="none").mean(2)  # (K2, K1)
        loss = torch.mean(torch.min(div, dim=0)[0]) + torch.mean(torch.min(div, dim=1)[0])

        if torch.isnan(loss) or torch.isinf(loss):
            continue
        all_loss.append(loss)
    if len(all_loss) == 0:
        return torch.tensor(0).float().to(region_score.device)
    else:
        return torch.mean(torch.stack(all_loss))



def sample_skeleton_points(joints, parents, sample_num=150):
    points = []
    for i, j in enumerate(parents):
        # i: child, j parent joint index
        if i == 0:
            continue
        joint_i = joints[i]
        joint_j = joints[j]
        length = np.linalg.norm(joint_i - joint_j)
        if length <= 0.01:
            continue
        samples = np.arange(0, length+0.01, 0.01)
        direction = (joint_i - joint_j) / length
        points.append(joint_j[None] + direction[None] * samples[:, None])
    points = np.concatenate(points, 0)
    # assert len(points) >= sample_num
    if len(points) >= sample_num:
        idxs = np.arange(len(points))
        np.random.shuffle(idxs)
    else:
        idxs = np.arange(5*sample_num)
        np.random.shuffle(idxs)
        idxs = idxs % len(points)
    return points[idxs[:sample_num]]


def get_part_mesh(vidx, f):
    # given a subset of vertex indices vidx, return a new face containing those vertices
    isin_f = np.isin(f, vidx)
    new_f = f[isin_f.all(1)]
    vidx = np.sort(vidx)
    mapping = np.zeros([np.max(vidx)+1], dtype=int)
    mapping[vidx] = np.arange(len(vidx))
    f_num = new_f.shape[0]
    new_f = mapping[new_f.reshape([-1])].reshape([f_num, 3])
    return new_f


def get_normal(v, f):
    f = f.T.astype(int)
    vec1 = v[f[1]] - v[f[0]]
    vec2 = v[f[2]] - v[f[0]]
    face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

    idx = np.concatenate([f[0], f[1], f[2]], axis=0)
    idx = torch.from_numpy(idx).long().to(v.device)
    face_norm = face_norm.repeat(3, 1)

    norm = scatter_add(face_norm, idx, dim=0, dim_size=v.size(0))
    norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]
    return norm


def get_normal_batch(vs, triangle, batch):
    B = torch.max(batch).detach().cpu().item() + 1
    normals = []
    for i in range(B):
        v = vs[batch==i]
        f = triangle[i][0]
        normals.append(get_normal(v, f))
    normals = torch.cat(normals, 0)
    return normals


def calc_surface_geodesic(mesh, threshold=0.06):
    # mesh: Open3d mesh
    # We denselu sample 4000 points to be more accuracy.
    if len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    # time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                    return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    surface_geodesic = surface_geodesic.astype(np.float32)

    rows, cols = [], []
    max_neighbor = 8
    for i in range(len(mesh.vertices)):
        js = np.where(surface_geodesic[i] < threshold)[0]
        if len(js) > max_neighbor:
            idxs = np.arange(len(js))
            np.random.shuffle(idxs)
            idxs = idxs[:max_neighbor]
            js = js[idxs]
        rows.extend(len(js) * [i])
        cols.extend(js)

    rows = np.array(rows).astype(np.int32)
    cols = np.array(cols).astype(np.int32)
    vals = surface_geodesic[rows, cols]
    # time2 = time.time()
    # print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return rows, cols, vals


if __name__ == '__main__':
    from utils.o3d_wrapper import Mesh
    a = Mesh(filename=r"C:\data_adobe\custom\lamp.obj")
    n1 = get_normal_np(a.v, a.f)
    n2 = get_normal(torch.from_numpy(a.v), a.f)
    b = 1
