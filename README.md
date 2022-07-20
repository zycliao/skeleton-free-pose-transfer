# Skeleton-free Pose Transfer for Stylized 3D Characters

This is the official repository for ECCV 2022 paper _Skeleton-free Pose Transfer for Stylized 3D Characters_.  

More detailed documentation coming soon!

## Prerequisites
- Python >= 3.7
- [Pytorch](https://pytorch.org/) >= 1.4
- [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Pymesh](https://pymesh.readthedocs.io/en/latest/installation.html) (optional)

```
pip install opencv-python tensorboardx smplx pyrender open3d cython kornia
```

## Demo
Download our demo data and pretrained model from [here](https://drive.google.com/file/d/1k0Vg1N6xlLoPGG3Lrpa5Ly5ThLEqAmUg/view?usp=sharing).
Unzip it to the project root directory.

Then,
```
python demo.py
```

Checkt the results in `./demo/results` and they should be the same as meshes in `./demo/results_reference`.  

To try with your own data, make sure the number of triangles is around 5K (not a strict requirement) and the orientation of the character is the same as demo data (front: +Z, up: +Y)

## Training
```
python train.py
```

More documentations about training will come soon.

## Citation
Please cite our paper if you use this repository:
```
@inproceedings{liao2022pose,
    title = {Skeleton-free Pose Transfer for Stylized 3D Characters},
    author = {Liao, Zhouyingcheng and Yang, Jimei and Saito, Jun and Pons-Moll, Gerard and Zhou, Yang},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {October},
    organization = {{Springer}},
    year = {2022},
}
```

## Credit
We borrowed part of the codes from the following projects:  

https://github.com/zycliao/TailorNet_dataset  
https://github.com/zhan-xu/RigNet  
https://github.com/YadiraF/face3d  
https://github.com/kzhou23/shape_pose_disent  

