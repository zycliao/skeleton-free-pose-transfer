# -*- coding: UTF-8 -*-
import bpy
import bmesh
import numpy as np
import os


def rotmatx(p, q):
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    d = np.array([(p * q).sum()])[:, None, None]
    c = (p.dot(eijk) @ q[..., None]).squeeze()  # cross product (optimized)
    cx = c.dot(eijk)
    return np.eye(3) + cx + cx @ cx / (1 + d)


class SkinMaterial(object):
    def __init__(self, name='Material skin'):
        mat = bpy.data.materials.new(name)
        self.name = name
        self.mat = mat

        # mat.diffuse_color = (1, 1, 1)
        # mat.specular_color = (0, 0, 0)
        mat.use_nodes = True
        self.node_tree = mat.node_tree
        self.nodes = self.node_tree.nodes
        self.bsdf = self.nodes.get("Principled BSDF")

        self.bsdf.inputs['Roughness'].default_value = 0.5
        self.bsdf.inputs['Specular'].default_value = 0.5

        hue_node = self.nodes.new(type='ShaderNodeHueSaturation')
        hue_node.inputs['Saturation'].default_value = 0

        bump_node = self.nodes.new(type='ShaderNodeBump')
        bump_node.inputs['Strength'].default_value = 0.05

        self.tex_node = self.nodes.new(type='ShaderNodeTexImage')
        self.node_tree.links.new(self.tex_node.outputs['Color'], self.bsdf.inputs['Base Color'])
        self.node_tree.links.new(self.tex_node.outputs['Color'], hue_node.inputs['Color'])
        self.node_tree.links.new(hue_node.outputs['Color'], bump_node.inputs['Height'])
        self.node_tree.links.new(bump_node.outputs['Normal'], self.bsdf.inputs['Normal'])

    def add_texture(self, img):
        self.img = img
        self.tex_node.image = img


class ClothMaterial(object):
    def __init__(self, name='Material cloth'):
        mat = bpy.data.materials.new(name)
        self.name = name
        self.mat = mat

        # mat.diffuse_color = (1, 1, 1)
        # mat.specular_color = (0, 0, 0)
        mat.use_nodes = True
        self.node_tree = mat.node_tree
        self.nodes = self.node_tree.nodes
        self.bsdf = self.nodes.get("Principled BSDF")

        self.bsdf.inputs['Roughness'].default_value = 0.5
        self.bsdf.inputs['Specular'].default_value = 0.

        self.tex_node = self.nodes.new(type='ShaderNodeTexImage')
        self.node_tree.links.new(self.tex_node.outputs['Color'], self.bsdf.inputs['Base Color'])

    def add_texture(self, img):
        self.img = img
        self.tex_node.image = img


def compute_euler_from_matrix(matrix, seq='xyz', extrinsic=False):
    # copy from sklearn source codes
    # The algorithm assumes intrinsic frame transformations. The algorithm
    # in the paper is formulated for rotation matrices which are transposition
    # rotation matrices used within Rotation.
    # Adapt the algorithm for our case by
    # 1. Instead of transposing our representation, use the transpose of the
    #    O matrix as defined in the paper, and be careful to swap indices
    # 2. Reversing both axis sequence and angles for extrinsic rotations

    _AXIS_TO_IND = {'x': 0, 'y': 1, 'z': 2}

    def _elementary_basis_vector(axis):
        b = np.zeros(3)
        b[_AXIS_TO_IND[axis]] = 1
        return b

    if extrinsic:
        seq = seq[::-1]

    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
    num_rotations = matrix.shape[0]

    # Step 0
    # Algorithm assumes axes as column vectors, here we use 1D vectors
    n1 = _elementary_basis_vector(seq[0])
    n2 = _elementary_basis_vector(seq[1])
    n3 = _elementary_basis_vector(seq[2])

    # Step 2
    sl = np.dot(np.cross(n1, n2), n3)
    cl = np.dot(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    offset = np.arctan2(sl, cl)
    c = np.vstack((n2, np.cross(n1, n2), n1))

    # Step 3
    rot = np.array([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ])
    res = np.einsum('...ij,...jk->...ik', c, matrix)
    matrix_transformed = np.einsum('...ij,...jk->...ik', res, c.T.dot(rot))

    # Step 4
    angles = np.empty((num_rotations, 3))
    # Ensure less than unit norm
    positive_unity = matrix_transformed[:, 2, 2] > 1
    negative_unity = matrix_transformed[:, 2, 2] < -1
    matrix_transformed[positive_unity, 2, 2] = 1
    matrix_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = np.arccos(matrix_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = (np.abs(angles[:, 1]) >= eps)
    safe2 = (np.abs(angles[:, 1] - np.pi) >= eps)

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = np.logical_and(safe1, safe2)
    angles[safe_mask, 0] = np.arctan2(matrix_transformed[safe_mask, 0, 2],
                                      -matrix_transformed[safe_mask, 1, 2])
    angles[safe_mask, 2] = np.arctan2(matrix_transformed[safe_mask, 2, 0],
                                      matrix_transformed[safe_mask, 2, 1])

    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = np.arctan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 2] = -np.arctan2(matrix_transformed[~safe2, 1, 0]
                                        + matrix_transformed[~safe2, 0, 1],
                                        matrix_transformed[~safe2, 0, 0]
                                        - matrix_transformed[~safe2, 1, 1])
    else:
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = np.arctan2(matrix_transformed[~safe1, 1, 0]
                                       - matrix_transformed[~safe1, 0, 1],
                                       matrix_transformed[~safe1, 0, 0]
                                       + matrix_transformed[~safe1, 1, 1])
        # 6c
        angles[~safe2, 0] = np.arctan2(matrix_transformed[~safe2, 1, 0]
                                       + matrix_transformed[~safe2, 0, 1],
                                       matrix_transformed[~safe2, 0, 0]
                                       - matrix_transformed[~safe2, 1, 1])

    # Step 7
    if seq[0] == seq[2]:
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = np.logical_or(angles[:, 1] < -np.pi / 2,
                                    angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = np.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    if extrinsic:
        angles = angles[:, ::-1]
    return angles


class Renderer(object):
    def __init__(self, w, h, spp, shadow):
        # enable CUDA
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
            try:
                cprefs.compute_device_type = compute_device_type
                break
            except TypeError:
                pass
        for device in cprefs.devices:
            device.use = True

        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        # scene.render.engine = 'BLENDER_EEVEE'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = spp
        scene.cycles.use_adaptive_sampling = True


        scene.render.resolution_percentage = 100
        scene.render.resolution_x = w
        scene.render.resolution_y = h
        self.w = w
        self.h = h
        # scene.render.use_border = True
        # scene.render.border_min_y = 0.125
        # scene.render.border_max_y = 0.875

        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.compression = 50
        scene.render.film_transparent = True

        scene.cursor.location = (0, 0, 0)

        if shadow:
            scene.objects['Plane'].cycles.is_shadow_catcher = True
        else:
            scene.objects['Plane'].cycles.is_shadow_catcher = False
            objs = bpy.data.objects
            objs.remove(objs["Plane"], do_unlink=True)


        self.scene = scene
        # self.grid_object = grid_object

    def render(self, out_path, mesh_paths, color=None):
        names = []
        mesh_objects = []

        for idx, mesh_path in enumerate(mesh_paths):
            if mesh_path.endswith('.ply'):
                imported_object = bpy.ops.import_mesh.ply(filepath=mesh_path)
                mesh_object = bpy.context.selected_objects[0]  ####<--Fix
                mesh_object.rotation_euler[0] = np.pi / 2
            else:
                imported_object = bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward='-Z', axis_up='Y')
                mesh_object = bpy.context.selected_objects[0]  ####<--Fix
            mesh_objects.append(mesh_object)
            mesh_object.select_set(True)  # origin set requires selecting the object
            bpy.context.view_layer.objects.active = mesh_object
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
            names.append(mesh_object.name)

            if color in ['Red', 'Green', 'Gray', 'Ply']:
                material = bpy.data.materials[color]
                mesh_object.active_material = material

            # mesh_object.cycles.is_shadow_catcher = True  # for plane

            bpy.ops.object.shade_smooth()
            # bpy.ops.object.shade_flat()
            mesh_object.select_set(False)

        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        # bpy.ops.wm.save_as_mainfile(filepath='/CT/lbcs/nobakup/tmp/temp.blend')

        for obj in mesh_objects:
            bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    import sys
    print(sys.argv)
    for i, p in enumerate(sys.argv):
        if p == '--':
            sys.argv = sys.argv[i:]
            break
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath')
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--spp', type=int)
    parser.add_argument('--color', type=str)
    parser.add_argument('--shadow', action="store_true")
    parser.add_argument('--meshes', nargs='+')

    args = parser.parse_args()
    print(args)

    # assert args.h <= args.w
    renderer = Renderer(args.img_size*2, args.img_size, args.spp, args.shadow)
    renderer.render(args.outpath, args.meshes, args.color)
