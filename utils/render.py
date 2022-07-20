'''
functions about rendering mesh(from 3d obj to 2d image).
only use rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light(I write these in other files)
2. Generally, the input vertices are normalized to [-1,1] and cetered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [w/2, h/2] with the y-axis pointing to oppisite direction.
 Means: render here only conducts interpolation.(I just want to make the input flexible)

Author: Yao Feng
Mail: yaofeng1995@gmail.com
'''
import numpy as np
from time import time

from utils.render_lib import mesh_core_cython
import math
from math import cos, sin


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)


## ------------------------------------------ 1. transform(transform, project, camera).
## ---------- 3d-3d transform. Transform obj in world space
def rotate(vertices, angles):
    ''' rotate vertices.
    X_new = R.dot(X). X: 3 x 1
    Args:
        vertices: [nver, 3].
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices


def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3].
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype=np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


## -------------- Camera. from world space to camera space
# Ref: https://cs184.eecs.berkeley.edu/lecture/transforms-2
def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    norm = np.maximum(norm, epsilon)
    return x / norm


def lookat_camera(vertices, eye, at=None, up=None):
    """ 'look at' transformation: from world space to camera space
    standard camera space:
        camera located at the origin.
        looking down negative z-axis.
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    Args:
      vertices: [nver, 3]
      eye: [3,] the XYZ world space position of the camera.
      at: [3,] a position along the center of the camera's gaze.
      up: [3,] up direction
    Returns:
      transformed_vertices: [nver, 3]
    """
    if at is None:
        at = np.array([0, 0, 0], np.float32)
    if up is None:
        up = np.array([0, 1, 0], np.float32)

    eye = np.array(eye).astype(np.float32)
    at = np.array(at).astype(np.float32)
    z_aixs = -normalize(at - eye)  # look forward
    x_aixs = normalize(np.cross(up, z_aixs))  # look right
    y_axis = np.cross(z_aixs, x_aixs)  # look up

    R = np.stack((x_aixs, y_axis, z_aixs))  # , axis = 0) # 3 x 3
    transformed_vertices = vertices - eye  # translation
    transformed_vertices = transformed_vertices.dot(R.T)  # rotation
    return transformed_vertices


## --------- 3d-2d project. from camera space to image plane
# generally, image plane only keeps x,y channels, here reserve z channel for calculating z-buffer.
def orthographic_project(vertices):
    ''' scaled orthographic projection(just delete z)
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
        x -> x*f/z, y -> x*f/z, z -> f.
        for point i,j. zi~=zj. so just delete z
        ** often used in face
        Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    Args:
        vertices: [nver, 3]
    Returns:
        projected_vertices: [nver, 3] if isKeepZ=True. [nver, 2] if isKeepZ=False.
    '''
    return vertices.copy()


def perspective_project(vertices, fovy, aspect_ratio=1., near=0.1, far=1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3]
    '''
    fovy = np.deg2rad(fovy)
    top = near * np.tan(fovy)
    bottom = -top
    right = top * aspect_ratio
    left = -right

    # -- homo
    P = np.array([[near / right, 0, 0, 0],
                  [0, near / top, 0, 0],
                  [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                  [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # [nver, 4]
    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices / projected_vertices[:, 3:]
    projected_vertices = projected_vertices[:, :3]
    projected_vertices[:, 2] = -projected_vertices[:, 2]

    # -- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices


def to_image(vertices, h, w, is_perspective=False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:, 0] = image_vertices[:, 0] * w / 2
        image_vertices[:, 1] = image_vertices[:, 1] * h / 2
    # move to center of image
    image_vertices[:, 0] = image_vertices[:, 0] + w / 2
    image_vertices[:, 1] = image_vertices[:, 1] + h / 2
    # flip vertices along y-axis.
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1
    return image_vertices


#### -------------------------------------------2. estimate transform matrix from correspondences.
def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[1], 1])))  # n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T  # Affine matrix. 3 x 4
    return P


def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T;
    x = x.T
    assert (x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert (n >= 4)

    # --- 1. normalization
    # 2d points
    mean = np.mean(x, 1)  # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x ** 2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean * scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1)  # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3, :] - X
    average_norm = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4, 4), dtype=np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean * scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n * 2, 8), dtype=np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])

    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype=np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine


def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


# Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert (isRotationMatrix)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz


def rasterize_triangles(vertices, triangles, h, w):
    '''
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle).
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''

    # initial
    depth_buffer = np.zeros([h, w]) - 999999.  # set the initial z to the farest position
    triangle_buffer = np.zeros([h, w], dtype=np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([h, w, 3], dtype=np.float32)  #

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()

    mesh_core_cython.rasterize_triangles_core(
        vertices, triangles,
        depth_buffer, triangle_buffer, barycentric_weight,
        vertices.shape[0], triangles.shape[0],
        h, w)


def render_colors(vertices, triangles, colors, alpha, h, w, c=3, BG=None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    '''

    # initial
    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    alpha = alpha.astype(np.float32).copy()
    ###
    st = time()
    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, c=3, mapping_type='nearest', BG=None):
    ''' render mesh with texture map
    Args:
        vertices: [3, nver]
        triangles: [3, ntri]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        h: height of rendering
        w: width of rendering
        c: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    # initial
    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG

    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)

    # -> C order
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    texture = texture.astype(np.float32).copy()
    tex_coords = tex_coords.astype(np.float32).copy()
    tex_triangles = tex_triangles.astype(np.int32).copy()

    mesh_core_cython.render_texture_core(
        image, vertices, triangles,
        texture, tex_coords, tex_triangles,
        depth_buffer,
        vertices.shape[0], tex_coords.shape[0], triangles.shape[0],
        h, w, c,
        tex_h, tex_w, tex_c,
        mt)
    return image


def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :] # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :] # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :] # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices, dtype = np.float32).copy() # [nver, 3]
    # for i in range(triangles.shape[0]):
    #     normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
    #     normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
    #     normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
    mesh_core_cython.get_normal_core(normal, tri_normal.astype(np.float32).copy(), triangles.copy(), triangles.shape[0])

    # normalize to unit length
    mag = np.sum(normal**2, 1) # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal/np.sqrt(mag[:,np.newaxis])

    return normal


def add_light(vertices, triangles, colors, light_positions=0, light_intensities=0):
    ''' Gouraud shading. add point lights.
    In 3d face, usually assume:
    1. The surface of face is Lambertian(reflect only the low frequencies of lighting)
    2. Lighting can be an arbitrary combination of point sources
    3. No specular (unless skin is oil, 23333)

    Ref: https://cs184.eecs.berkeley.edu/lecture/pipeline
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        light_positions: [nlight, 3]
        light_intensities: [nlight, 3]
    Returns:
        lit_colors: [nver, 3]
    '''
    nver = vertices.shape[0]
    normals = get_normal(vertices, triangles)  # [nver, 3]

    # ambient
    # La = ka*Ia

    # diffuse
    # Ld = kd*(I/r^2)max(0, nxl)
    direction_to_lights = vertices[np.newaxis, :, :] - light_positions[:, np.newaxis, :]  # [nlight, nver, 3]
    direction_to_lights_n = np.sqrt(np.sum(direction_to_lights ** 2, axis=2))  # [nlight, nver]
    direction_to_lights = direction_to_lights / (direction_to_lights_n[:, :, np.newaxis] + 1e-6)
    normals_dot_lights = normals[np.newaxis, :, :] * direction_to_lights  # [nlight, nver, 3]
    normals_dot_lights = np.abs(np.sum(normals_dot_lights, axis=2))  # [nlight, nver]
    diffuse_output = colors[np.newaxis, :, :] * normals_dot_lights[:, :, np.newaxis] * light_intensities[:, np.newaxis,
                                                                                       :]
    diffuse_output = np.sum(diffuse_output, axis=0)  # [nver, 3]

    # specular
    # h = (v + l)/(|v + l|) bisector
    # Ls = ks*(I/r^2)max(0, nxh)^p
    # increasing p narrows the reflectionlob

    lit_colors = diffuse_output  # only diffuse part here.
    lit_colors = np.minimum(np.maximum(lit_colors, 0), 1)
    return lit_colors


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
        vcs = np.concatenate(new_vcs, 0)

    return vs, new_fs, vcs



class Renderer(object):
    def __init__(self, img_size, bg_color=None, ambient=0.1):
        if bg_color is None:
            self.bg_color = np.array([0.1, 0.1, 0.1])
        else:
            self.bg_color = bg_color
        self.bg_color = (self.bg_color * 255 + 0.5).astype(np.uint8)
        self.w = img_size
        self.h = img_size
        self.ambient = ambient
        self.dtype = np.float32

    def __call__(self, vs, fs, vcs=None, trans=(1.5, 0., 0.), euler=(0., 0., 0.), center=True):
        if isinstance(vs, list):
            vs, fs, vcs = merge_mesh(vs, fs, vcs)
        else:
            vs = np.array(vs)
        if vcs is None:
            vcs = np.ones_like(vs)
        if center:
            cen = (np.max(vs, 0, keepdims=True) + np.min(vs, 0, keepdims=True)) / 2.
            vs -= cen

        rotmat = self.euler2rotmat(euler)
        vs = np.einsum('pq,nq->np', rotmat, vs)
        # trans
        vs[:, :2] += np.expand_dims(np.array(trans[1:]), 0)
        vs *= trans[0]


        vertices = vs.astype(np.float32)
        faces = fs.astype(np.int32)
        # face_colors = vcs / np.max(vcs)
        verts_colors = vcs / 1


        s = np.minimum(self.h, self.w) / 2
        R = angle2matrix([0, 0, 0])
        t = [0, 0, 0]
        vertices = similarity_transform(vertices, s, R, t)
        # h, w of rendering
        image_vertices = to_image(vertices, self.h, self.w)

        light_positions = np.array([[-0, -0, -300]])
        light_intensities = np.array([[1, 1, 1]])
        shaded_colors = add_light(vertices, faces, verts_colors[:, :3], light_positions, light_intensities)
        if verts_colors.shape[1] == 4:
            alpha = verts_colors[:, 3].copy()
        else:
            alpha = np.ones_like(verts_colors[:, 0])
        img = render_colors(image_vertices, faces, shaded_colors, alpha, self.h, self.w)
        img = np.around(img * 255).astype(np.uint8)
        return img

    @staticmethod
    def euler2rotmat(euler):
        euler = np.array(euler) * np.pi / 180.
        se, ce = np.sin(euler), np.cos(euler)
        s1, c1 = se[0], ce[0]
        s2, c2 = se[1], ce[1]
        s3, c3 = se[2], ce[2]
        return np.array([[c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                         [c2 * s3, c2 * c3, -s2],
                         [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2]])


if __name__ == '__main__':
    import cv2
    from utils.o3d_wrapper import Mesh
    m = Mesh(filename="/Users/zliao/Data/log_02_rignet/log_1012_3/val_180/0/loss1_gt.ply")
    renderer = Renderer(512)
    img = renderer(m.v, m.f, trans=(1.5, 0, 0))
    cv2.imshow("a", img)
    cv2.waitKey()

