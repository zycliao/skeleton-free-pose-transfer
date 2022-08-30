from utils.geometry import calc_surface_geodesic
from utils.o3d_wrapper import MeshO3d
from global_var import *



if __name__ == '__main__':
    save_path = os.path.join(SMPLH_PATH, 'geodesic_simplify.npz')
    simplify_path = r'D:\data\v3\smpl\simplify.obj'
    mesh = MeshO3d(filename=simplify_path)
    v = mesh.v
    scale = np.max(v[:, 1]) - np.min(v[:, 1])
    v = v * 2 / scale
    mesh.v = v
    rows, cols, vals = calc_surface_geodesic(mesh.m)
    np.savez(save_path, rows=rows, cols=cols, vals=vals)
