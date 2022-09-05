import os

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
from kornia.geometry.conversions import QuaternionCoeffOrder

import global_var

wxyz = QuaternionCoeffOrder.WXYZ
from models.networks import PerPartEncoderTpl, PerPartDecoder, HandlePredictorSWTpl
from models.ops import handle2mesh, get_transformation, arap_smooth
from models.smpl import SMPL2Mesh
from utils.o3d_wrapper import Mesh
from data_utils.custom_loader import CustomDataset, CustomMotionDataset


if __name__ == '__main__':
    use_smooth = False
    save_dir = './demo/results_smpl'
    ckpt_path = "./demo/ckpt.pth"
    src_data_dir = "./demo/src_data_smpl"
    dst_data_dir = "./demo/dst_data_smpl"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(src_data_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_data_dir, 'obj_remesh'), exist_ok=True)

    smpl = SMPL2Mesh(global_var.SMPLH_PATH)
    pose0 = np.zeros([1, 156])
    pose1 = np.zeros([1, 156])
    pose1[0, 18*3+1] = -np.pi/2
    shape1 = np.random.normal(loc=0, scale=1, size=[1, 16])
    shape2 = np.random.normal(loc=0, scale=1, size=[1, 16])
    src0 = smpl(pose0, shape1, [1])
    src1 = smpl(pose1, shape1, [1])
    dst = smpl(pose0, shape2, [1])
    Mesh(v=src0[0].numpy(), f=smpl.f).write_obj(os.path.join(src_data_dir, 'rest.obj'))
    Mesh(v=src1[0].numpy(), f=smpl.f).write_obj(os.path.join(src_data_dir, '1.obj'))
    Mesh(v=dst[0].numpy(), f=smpl.f).write_obj(os.path.join(dst_data_dir, 'obj_remesh', 'dst1.obj'))

    input_dim = 6
    num_feature = 128*2+7
    encoder_shape = PerPartEncoderTpl(input_dim, 128)
    encoder_pose = PerPartEncoderTpl(input_dim, 128)
    predictor = HandlePredictorSWTpl(input_dim, 40)
    decoder = PerPartDecoder(num_feature)

    device = torch.device("cuda:0")
    encoder_shape.to(device)
    encoder_pose.to(device)
    predictor.to(device)
    decoder.to(device)

    checkpoint = torch.load(ckpt_path)
    encoder_shape.load_state_dict(checkpoint['encoder_shape'])
    if 'encoder_pose' in checkpoint:
        encoder_pose.load_state_dict(checkpoint['encoder_pose'])
    else:
        encoder_pose.load_state_dict(checkpoint['encoder_shape'])
    decoder.load_state_dict(checkpoint['decoder'])
    predictor.load_state_dict(checkpoint['predictor'])

    encoder_pose.eval()
    encoder_shape.eval()
    predictor.eval()
    decoder.eval()

    torch.set_grad_enabled(False)

    src_set = CustomMotionDataset(src_data_dir)
    src_loader = DataLoader(src_set, batch_size=1,
                            shuffle=False, pin_memory=False, drop_last=False)
    dst_set = CustomDataset(dst_data_dir)
    dst_loader = DataLoader(dst_set, batch_size=1,
                            shuffle=False, pin_memory=False, drop_last=False)

    for i_s, src_data, in enumerate(tqdm(src_loader)):
        for i_d, dst_data in enumerate(dst_loader):
            src_data.to(device)
            dst_data.to(device)
            hm0, hd0_mean, _, region_score0 = predictor(torch.cat((dst_data.v0, dst_data.feat0), 1)
                                                        , data=dst_data, verbose=True)
            hm1, hd1_mean, _, region_score1 = predictor(torch.cat((src_data.v0, src_data.feat0), 1)
                                                        , data=src_data, verbose=True)
            trans1 = get_transformation(hm1, region_score1, src_data.batch, src_data.v0, src_data.v1)

            ap1 = encoder_pose(src_data.v1, hm1, data=src_data, feat=src_data.feat1)
            as0 = encoder_shape(src_data.v0, hm1, data=src_data, feat=src_data.feat0)
            rs0 = encoder_shape(dst_data.v0, hm0, data=dst_data, feat=dst_data.feat0)
            pred_disp = decoder(ap1-as0, rs0, trans1)
            if use_smooth:
                pred_disp = arap_smooth(pred_disp, hd0_mean, region_score0, dst_data.batch,
                                        dst_data.v0, dst_data.tpl_edge_index, 0.5)

            pred_v = handle2mesh(pred_disp, hd0_mean, region_score0, dst_data.batch, dst_data.v0)
            r_f = dst_data.triangle[0][0]
            v0 = dst_data.v0.cpu().numpy()
            pred_v = pred_v.cpu().numpy()
            Mesh(v=pred_v, f=r_f).write_obj(os.path.join(save_dir, f'{dst_data.name[0]}_{src_data.name[0]}.obj'))

