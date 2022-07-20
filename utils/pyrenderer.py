import os
if 'SSH_CONNECTION' in os.environ:
    print('You are in an SSH connection')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import trimesh
import numpy as np
import torch
import cv2
from OpenGL.error import GLError


COLORS_HSV = np.zeros([64, 3])
ci = 0
for s in [170, 100]:
    for h in np.linspace(0, 179, 16):
        for v in [220, 128]:
            COLORS_HSV[ci] = [h, s, v]
            ci += 1
COLORS = cv2.cvtColor(COLORS_HSV[None].astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float) / 255
COLORS_ALPHA = np.concatenate((COLORS, np.ones_like(COLORS[:, :1])), 1)


class Renderer(object):
    """
    This is a wrapper of pyrender
    see documentation of __call__ for detailed usage
    """

    def __init__(self, img_size, bg_color=None):
        if bg_color is None:
            bg_color = np.array([0.1, 0.1, 0.1, 1.])
        self.scene = pyrender.Scene(bg_color=bg_color)
        self.focal_len = 5.
        camera = pyrender.PerspectiveCamera(yfov=np.tan(1 / self.focal_len) * 2, aspectRatio=1.0)
        camera_pose = np.eye(4, dtype=np.float32)
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0,
                                          )
        self.scene.add(light, pose=camera_pose)
        if not hasattr(img_size, '__iter__'):
            img_size = [img_size, img_size]
        self.r = pyrender.OffscreenRenderer(*img_size)
        self.null_img = np.ones([img_size[1], img_size[0], 3]) * bg_color[:3][None][None]
        self.null_img = (self.null_img * 255).astype(np.uint8)

    def __call__(self, vs, fs, vcs=None, trans=(1.4, 0., 0.), euler=(0., 0., 0.), center=True, wire=None):
        """
        This function will put the center of objects at origin point.
        vs, fs, vcs:
            vertices, faces, colors of vertices.
            They are numpy array or list of numpy array (multiple meshes)
        trans:
            It is a 3 element tuple. The first is scale factor. The last two is x,y translation
        euler:
            euler angle of objects (degree not radian). It follows the order of YXZ,
            which means Y-axis, X-axis, Z-axis are yaw, pitch, roll respectively.
        """
        if isinstance(vs, np.ndarray) or isinstance(vs, torch.Tensor):
            vs = [vs]
            fs = [fs]
            vcs = [vcs]
        for i, v in enumerate(vs):
            if isinstance(v, torch.Tensor):
                vs[i] = v.detach().cpu().numpy()
        for i, f in enumerate(fs):
            if isinstance(f, torch.Tensor):
                fs[i] = f.detach().cpu().numpy()
        if wire is None:
            wire = [False] * len(vs)
        if vcs is None:
            vcs = [None] * len(vs)
        ms = []
        mnodes = []
        vss = np.concatenate(vs, 0)
        cen = (np.max(vss, 0, keepdims=True) + np.min(vss, 0, keepdims=True)) / 2.
        rotmat = self.euler2rotmat(euler)
        for v, f, vs in zip(vs, fs, vcs):
            trans_v = v - cen if center else v
            trans_v = np.einsum('pq,nq->np', rotmat, trans_v)
            trans_v[:, :2] += np.expand_dims(np.array(trans[1:]), 0)
            trans_v[:, 2] -= self.focal_len / trans[0]
            ms.append(trimesh.Trimesh(vertices=trans_v, faces=f, vertex_colors=vs))
        for m, w in zip(ms, wire):
            mnode = self.scene.add(pyrender.Mesh.from_trimesh(m, wireframe=w))
            mnodes.append(mnode)
        try:
            img, depth = self.r.render(self.scene)
        except GLError:
            print("Rendering failed")
            img = self.null_img
        for mnode in mnodes:
            self.scene.remove_node(mnode)
        return img

    def render_cage(self, vs, fs, vcs=None, trans=(1.4, 0., 0.), euler=(0., 0., 0.), center=True):
        """
        This function will put the center of objects at origin point.
        vs, fs, vcs:
            vertices, faces, colors of vertices.
            They are numpy array or list of numpy array (multiple meshes)
        trans:
            It is a 3 element tuple. The first is scale factor. The last two is x,y translation
        euler:
            euler angle of objects (degree not radian). It follows the order of YXZ,
            which means Y-axis, X-axis, Z-axis are yaw, pitch, roll respectively.
        """
        if isinstance(vs, np.ndarray):
            vs = [vs]
            fs = [fs]
            vcs = [vcs]
        ms = []
        wires = []
        mnodes = []
        vss = np.concatenate(vs, 0)
        cen = (np.max(vss, 0, keepdims=True) + np.min(vss, 0, keepdims=True)) / 2.
        rotmat = self.euler2rotmat(euler)
        for v, f, vs in zip(vs, fs, vcs):
            trans_v = v - cen if center else v
            trans_v = np.einsum('pq,nq->np', rotmat, trans_v)
            trans_v[:, :2] += np.expand_dims(np.array(trans[1:]), 0)
            trans_v[:, 2] -= self.focal_len / trans[0]
            ms.append(trimesh.Trimesh(vertices=trans_v, faces=f, vertex_colors=vs))
            wires.append(trimesh.Trimesh(vertices=trans_v, faces=f, vertex_colors=vs))
        for m, w in zip(ms, wires):
            # mnode = self.scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
            mnode2 = self.scene.add(pyrender.Mesh.from_trimesh(w, wireframe=True, smooth=False))
            # mnodes.append(mnode)
            mnodes.append(mnode2)
        img, depth = self.r.render(self.scene)
        for mnode in mnodes:
            self.scene.remove_node(mnode)
        return img

    def viewer(self, vs, fs, vcs=None, trans=(1.4, 0., 0.), euler=(0., 0., 0.), center=True):
        """
        This function will put the center of objects at origin point.
        vs, fs, vcs:
            vertices, faces, colors of vertices.
            They are numpy array or list of numpy array (multiple meshes)
        trans:
            It is a 3 element tuple. The first is scale factor. The last two is x,y translation
        euler:
            euler angle of objects (degree not radian). It follows the order of YXZ,
            which means Y-axis, X-axis, Z-axis are yaw, pitch, roll respectively.
        """
        if isinstance(vs, np.ndarray):
            vs = [vs]
            fs = [fs]
            vcs = [vcs]
        ms = []
        mnodes = []
        vss = np.concatenate(vs, 0)
        cen = (np.max(vss, 0, keepdims=True) + np.min(vss, 0, keepdims=True)) / 2.
        rotmat = self.euler2rotmat(euler)
        for v, f, vs in zip(vs, fs, vcs):
            trans_v = v - cen if center else v
            trans_v = np.einsum('pq,nq->np', rotmat, trans_v)
            trans_v[:, :2] += np.expand_dims(np.array(trans[1:]), 0)
            trans_v[:, 2] -= self.focal_len / trans[0]
            ms.append(trimesh.Trimesh(vertices=trans_v, faces=f, vertex_colors=vs))
        for m in ms:
            mnode = self.scene.add(pyrender.Mesh.from_trimesh(m))
            mnodes.append(mnode)
        pyrender.Viewer(self.scene)
        for mnode in mnodes:
            self.scene.remove_node(mnode)

    @staticmethod
    def euler2rotmat(euler):
        euler = np.array(euler)*np.pi/180.
        se, ce = np.sin(euler), np.cos(euler)
        s1, c1 = se[0], ce[0]
        s2, c2 = se[1], ce[1]
        s3, c3 = se[2], ce[2]
        return np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])


if __name__ == '__main__':
    renderer = Renderer(512)