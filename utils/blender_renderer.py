import time
import numpy as np
import os
import copy
from utils.o3d_wrapper import Mesh
from global_var import *


height_box = 1.3


def calc_img_width(widths, img_height):
    ortho = np.sum(widths) * 1.1
    img_width = (img_height * ortho) / height_box
    img_width = int(img_width)
    img_width = np.maximum(img_height, img_width)
    return img_width


def render(outpath, v, f, img_size=512, spp=128, shadow=True, color=None, tranlation=None):
    # raise NotImplementedError
    bpy_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'blender_bpy.py')
    cmd = "{} --background {} --python {} -- --outpath {} --meshes ".format(  # or blender-softwaregl
        BLENDER_PATH,
        BLEND_FILE,
        bpy_path,
        outpath,
    )

    prefix = str(time.time_ns() // 1000)
    v = v.copy()
    if tranlation is not None:
        v[:, 1] += tranlation
    else:
        v[:, 1] -= np.min(v[:, 1])
    path = TEMP_DIR + f"/{prefix}.obj"
    if not isinstance(color, str):
        path = path.replace('.obj', '.ply')
        Mesh(v=v, f=f, vc=color).write_ply(path)
    else:
        Mesh(v=v, f=f).write_obj(path)
    print(f"Write to {path}")
    cmd += path
    cmd += " "

    cmd += "--img_size {} {} ".format(img_size, "--shadow" if shadow else "")
    cmd += "--spp {} ".format(spp)
    if color is not None:
        if not isinstance(color, str):
            cmd += "--color Ply"
        else:
            cmd += "--color {}".format(color)

    print(cmd)
    os.system(cmd)
    print("Done")


def render_path(outpath, input_path, img_size=512, spp=128, shadow=True, color=None):
    bpy_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'blender_bpy.py')
    cmd = "{} --background {} --python {} -- --outpath {} --meshes ".format(  # or blender-softwaregl
        BLENDER_PATH,
        BLEND_FILE,
        bpy_path,
        outpath,
    )

    print(f"Rendering path: {input_path}")
    cmd += input_path
    cmd += " "

    cmd += "--img_size {} {} ".format(img_size, "--shadow" if shadow else "")
    cmd += "--spp {} ".format(spp)
    if color is not None:
        if not isinstance(color, str):
            cmd += "--color Ply"
        else:
            cmd += "--color {}".format(color)

    print(cmd)
    os.system(cmd)
    print("Done")


if __name__ == '__main__':
    m = Mesh(filename='C:/data_adobe/mixamo_simplify/watertight/aj.obj')
    render('C:/data_adobe/test.png', m.v, m.f, 512, shadow=True, color="Red")