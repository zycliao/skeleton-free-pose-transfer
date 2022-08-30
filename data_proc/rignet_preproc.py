import numpy as np
from queue import Queue
from scipy.sparse import csr_matrix
from utils.lbs import lbs
from utils.geometry import get_tpl_edges, calc_surface_geodesic
from utils.o3d_wrapper import Mesh, MeshO3d


class Node(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


class TreeNode(Node):
    def __init__(self, name, pos):
        super(TreeNode, self).__init__(name, pos)
        self.children = []
        self.parent = None


class Info:
    """
    Wrap class for rig information
    """
    def __init__(self, filename=None):
        self.joint_pos = {}
        self.joint_skin = []
        self.root = None
        self.parents = []
        self.joints = None
        self.weights = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as f_txt:
            lines = f_txt.readlines()
        for line in lines:
            word = line.split()
            if word[0] == 'joints':
                self.joint_pos[word[1]] = [float(word[2]), float(word[3]), float(word[4])]
            elif word[0] == 'root':
                root_pos = self.joint_pos[word[1]]
                self.root = TreeNode(word[1], (root_pos[0], root_pos[1], root_pos[2]))
            elif word[0] == 'skin':
                skin_item = word[1:]
                self.joint_skin.append(skin_item)
        self.loadHierarchy_recur(self.root, lines, self.joint_pos)

        #
        joint_pos_arr = [self.root.pos]
        joint_names = [self.root.name]
        joint_idx_dict = {self.root.name: 0}
        q = Queue()
        q.put(self.root)
        self.parents.append(-1)
        while not q.empty():
            node = q.get()
            parent_idx = joint_idx_dict[node.name]
            for child in node.children:
                self.parents.append(parent_idx)
                q.put(child)
                joint_pos_arr.append(child.pos)
                joint_idx_dict[child.name] = len(joint_names)
                joint_names.append(child.name)
        self.joints = np.array(joint_pos_arr)
        self.parents = np.array(self.parents)

        num_joint = len(joint_names)
        num_verts = len(self.joint_skin)
        assert num_joint == len(self.parents) == len(self.joint_pos)
        rows, cols, data = [], [], []
        for i, skin_per_vertex in enumerate(self.joint_skin):
            for j in range(len(skin_per_vertex)//2):
                rows.append(i)
                jn = skin_per_vertex[2 * j + 1]
                val = float(skin_per_vertex[2 * j + 2])
                cols.append(joint_idx_dict[jn])
                data.append(val)
        self.weights = csr_matrix((data, (rows, cols)), shape=(num_verts, num_joint), dtype=np.float32)
        self.sparse_weights = {"data": self.weights.data, "rows": rows, "cols": cols, "shape": self.weights.shape}

    def loadHierarchy_recur(self, node, lines, joint_pos):
        for li in lines:
            if li.split()[0] == 'hier' and li.split()[1] == node.name:
                pos = joint_pos[li.split()[2]]
                ch_node = TreeNode(li.split()[2], tuple(pos))
                node.children.append(ch_node)
                ch_node.parent = node
                self.loadHierarchy_recur(ch_node, lines, joint_pos)

    def save(self, filename, **kwargs):
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
        c = {"joints": self.joints.astype(np.float32), "parents": self.parents,
             "weights_data": self.sparse_weights["data"], "weights_rows": self.sparse_weights["rows"],
             "weights_cols": self.sparse_weights["cols"], "weights_shape": self.sparse_weights["shape"]}
        for k, v in kwargs.items():
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            c[k] = v
        np.savez(filename, **c)

    def save_as_skel_format(self, filename):
        fout = open(filename, 'w')
        this_level = [self.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else 'None'
                line = '{0} {1} {2:8f} {3:8f} {4:8f} {5}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2],
                                                                   parent)
                fout.write(line)
                for c_node in p_node.children:
                    next_level.append(c_node)
            this_level = next_level
            hier_level += 1
        fout.close()

    def normalize(self, scale, trans):
        for k, v in self.joint_pos.items():
            self.joint_pos[k] /= scale
            self.joint_pos[k] -= trans

        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                node.pos /= scale
                node.pos = (node.pos[0] - trans[0], node.pos[1] - trans[1], node.pos[2] - trans[2])
                for ch in node.children:
                    next_level.append(ch)
            this_level = next_level

    def get_joint_dict(self):
        joint_dict = {}
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                joint_dict[node.name] = node.pos
                next_level += node.children
            this_level = next_level
        return joint_dict

    def adjacent_matrix(self):
        joint_pos = self.get_joint_dict()
        joint_name_list = list(joint_pos.keys())
        num_joint = len(joint_pos)
        adj_matrix = np.zeros((num_joint, num_joint))
        this_level = [self.root]
        while this_level:
            next_level = []
            for p_node in this_level:
                for c_node in p_node.children:
                    index_parent = joint_name_list.index(p_node.name)
                    index_children = joint_name_list.index(c_node.name)
                    adj_matrix[index_parent, index_children] = 1.
                next_level += p_node.children
            this_level = next_level
        adj_matrix = adj_matrix + adj_matrix.transpose()
        return adj_matrix


if __name__ == '__main__':
    import os
    import sys
    from tqdm import tqdm
    from global_var import *

    job_id = int(sys.argv[-1]) - 1
    # job_id = 0
    job_num = 100

    data_dir = RIGNET_PATH
    mesh_dir = os.path.join(data_dir, 'obj_remesh')
    rig_dir = os.path.join(data_dir, 'rig_info_remesh')
    save_dir = os.path.join(data_dir, 'npz')
    os.makedirs(save_dir, exist_ok=True)
    fnames = [k for k in os.listdir(mesh_dir) if k.endswith('.obj')]
    data_num = len(fnames)
    l = int(data_num * job_id / job_num)
    r = int(data_num * (job_id+1) / job_num)
    fnames = fnames[l: r]

    for fname in tqdm(fnames):
        save_path = os.path.join(save_dir, fname.replace('.obj', '.npz'))
        if os.path.exists(save_path):
            continue
        m = Mesh(filename=os.path.join(mesh_dir, fname))
        try:
            rig_info = Info(os.path.join(rig_dir, fname.replace('.obj', '.txt')))
        except KeyError:
            print(fname)
            continue
        try:
            tpl_edge_index = get_tpl_edges(m.v, m.f)
            v = m.v
            scale = np.max(v[:, 1]) - np.min(v[:, 1])
            v = v * 2 / scale
            geo_rows, geo_cols, geo_vals = calc_surface_geodesic(MeshO3d(v=v, f=m.f).m)
        except ValueError:
            continue

        rig_info.save(save_path, v=m.v, f=m.f, tpl_edge_index=tpl_edge_index,
                      geo_rows=geo_rows, geo_cols=geo_cols, geo_vals=geo_vals)
