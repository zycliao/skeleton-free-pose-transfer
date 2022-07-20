import os, sys
import shutil

import cv2
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')

from data_utils.multi_loader import MultiDataset
from data_utils.amass_loader import AmassDataset
from data_utils.rignet_loader import RignetDataset
from data_utils.mixamo_loader import MixamoValidationDataset
from models.networks import PerPartEncoderTpl, PerPartDecoder, HandlePredictorSWTpl
from models.smpl import SMPL2Mesh
from utils.training import LossManager
from utils.pyrenderer import Renderer
from utils.visualization import visualize_handle, visualize_part
from utils.o3d_wrapper import Mesh
from utils.geometry import arap_loss, fps, get_normal_batch
from models.ops import *
from global_var import *


def visualize_data(vs, fs, save_path):
    renderer = Renderer(400)
    imgs = []
    for v, f in zip(vs, fs):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(f, list):
            f = f[0][0]
        imgs.append(renderer(v, f))
    imgs = np.concatenate(imgs, 1)
    cv2.imwrite(save_path, imgs)


"""
Shape and Handle in the same encoder
"""
def main():
    # import warnings
    # warnings.filterwarnings("error")
    bs = 4
    lr = 1e-4
    exp_name = 'exp'
    exp_dir = f"{LOG_DIR}/{exp_name}"

    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)
    ckpt_path = os.path.join(exp_dir, "latest.pth")
    # shutil.copy(f"{LOG_DIR}/log_1126/epoch_300.pth", ckpt_path)  # pretrained
    smpl = SMPL2Mesh(SMPLH_PATH)
    num_workers = 4
    part_aug_scale = ((0.5, 3), (0.7, 1), (0.5, 1.5))
    train_set = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, part_aug_scale=part_aug_scale,
                             part_augmentation=(False, True, False), prob=(0.2, 0.6, 0.2),
                             single_part=True, simplify=True, new_rignet=True)
    train_loader = DataLoader(train_set, batch_size=bs,
                              shuffle=True, pin_memory=False, drop_last=True,
                              num_workers=num_workers)

    train_set2 = MultiDataset(AMASS_PATH, MIXAMO_SIMPLIFY_PATH, RIGNET_PATH, smpl, single_part=True,
                              part_aug_scale=part_aug_scale,
                              part_augmentation=(False, True, False), prob=(0.3, 0.4, 0.3),
                              preload=train_set.database(), simplify=True, new_rignet=True)
    train_loader2 = DataLoader(train_set2, batch_size=bs,
                               shuffle=True, pin_memory=False, drop_last=True,
                               num_workers=num_workers)

    test_mixamo_set = MixamoValidationDataset(MIXAMO_SIMPLIFY_PATH)
    test_mixamo_loader = DataLoader(test_mixamo_set, batch_size=1,
                                    shuffle=False, pin_memory=False, drop_last=False,
                                    num_workers=num_workers)


    test_amass_set = AmassDataset(AMASS_PATH, smpl, "train", simplify=True)
    test_amass_loader = DataLoader(test_amass_set, batch_size=1,
                                   shuffle=True, pin_memory=False, drop_last=True,
                                   num_workers=num_workers)

    test_rignet_set = RignetDataset(RIGNET_PATH, "humanoid_test_new")
    test_rignet_loader = DataLoader(test_rignet_set, batch_size=1,
                                    shuffle=True, pin_memory=False, drop_last=True,
                                    num_workers=num_workers)

    num_handle = 40
    encoder_shape = PerPartEncoderTpl(3+3, 128)
    predictor = HandlePredictorSWTpl(3+3, num_handle)
    decoder = PerPartDecoder(128*2+7)

    encoder_shape.cuda()
    predictor.cuda()
    decoder.cuda()

    device = torch.device("cuda:0")
    # reference SMPL template
    ref_bm_path = os.path.join(SMPLH_PATH, 'neutral/model.npz')
    ref_bm = np.load(ref_bm_path)
    ref_mesh = Mesh(v=ref_bm['v_template'], f=ref_bm['f'])

    optimizer = optim.AdamW(list(encoder_shape.parameters()) +
                            list(decoder.parameters()) +
                            list(predictor.parameters()),
                            lr=lr)
    criterion = torch.nn.L1Loss()

    no_render = False
    if not no_render:
        renderer = Renderer(400)

    # predictor.load_state_dict(torch.load(predictor_path)['predictor'])
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        encoder_shape.load_state_dict(checkpoint['encoder_shape'])
        decoder.load_state_dict(checkpoint['decoder'])
        predictor.load_state_dict(checkpoint['predictor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully load {ckpt_path}")
    else:
        start_epoch = 0

    loss_m = LossManager(['loss1', 'loss2', 'loss_arap', 'loss_fps', 'loss_cross',
                          'loss_sw', 'loss_sw_sim'], writer=writer, epoch=start_epoch)
    mixamo_loss_m = LossManager(['mixamo_dist', 'mixamo_loss'], writer=writer, epoch=start_epoch)

    for i_epoch in range(start_epoch, 501):
        encoder_shape.train()
        decoder.train()
        predictor.train()
        # encoder_shape.eval()
        # decoder.eval()
        # predictor.eval()
        loss_m.update_epoch(i_epoch)
        mixamo_loss_m.update_epoch(i_epoch)
        print(f"=====Experiment {exp_name}\tEpoch {i_epoch}=====")
        for i_iter, ((data1, data2), (data3, _)) in enumerate(zip(train_loader, train_loader2)):
            optimizer.zero_grad()

            data1.to(device)
            data2.to(device)
            data3.to(device)

            # if torch.isnan(data2.aug_v1).any() or torch.isinf(data2.aug_v1).any():
            #     import IPython; IPython.embed()
            # if i_iter >= 10:
            #     exit()

            hm1, hd_pos1, _, region_score1 = predictor(torch.cat((data1.aug_v0, data1.feat0), 1),
                                                       data=data1, verbose=True)
            _, _, _, region_score2_noaug = predictor(torch.cat((data2.v0, data2.feat0), 1),
                                                     data=data2, verbose=True)
            hm2, hd_pos2, _, region_score2 = predictor(torch.cat((data2.aug_v0, data2.feat0), 1),
                                                       data=data2, verbose=True)
            hm3, hd_pos3, _, region_score3 = predictor(torch.cat((data3.aug_v0, data3.feat0), 1),
                                                       data=data3, verbose=True)

            hm1, hm2, hm3 = hm1.detach(), hm2.detach(), hm3.detach()

            pose_enc_0 = encoder_shape(data1.aug_v0, hm1, data=data1, feat=data1.feat0)
            pose_enc = encoder_shape(data1.aug_v1, hm1, data=data1, feat=data1.feat1)
            shape_enc = encoder_shape(data2.aug_v0, hm2, data=data2, feat=data2.feat0)
            trans1 = get_transformation(hm1, region_score1, data1.batch, data1.aug_v0, data1.aug_v1)
            pred_disp = decoder(pose_enc-pose_enc_0, shape_enc, trans1.detach())
            pred_v = handle2mesh(pred_disp, hd_pos2.detach(), region_score2, data2.batch, data2.aug_v0)
            pred_v_gtt = gt_trans_mesh(pred_disp, hm2, region_score2, data2.batch, data2.aug_v0, data2.v1)
            loss1 = criterion(data2.aug_v1, pred_v) + criterion(data2.aug_v1, pred_v_gtt)
            loss1 = loss1 * 0.5

            pose_enc2 = encoder_shape(data2.aug_v1, hm2, data=data2, feat=data2.feat1)
            pose_enc2_0 = shape_enc
            shape_enc2 = encoder_shape(data3.aug_v0, hm3, data=data3, feat=data3.feat0)
            trans2 = get_transformation(hm2, region_score2, data2.batch, data2.aug_v0, data2.aug_v1)
            cross_disp = decoder(pose_enc2-pose_enc2_0, shape_enc2, trans2.detach())
            cross_v = handle2mesh(cross_disp, hd_pos3.detach(), region_score3.detach(), data3.batch, data3.aug_v0)
            cross_superv = handle2mesh(trans2, hd_pos3.detach(), region_score3, data3.batch, data3.aug_v0).detach()
            cross_v_gtt = gt_trans_mesh(cross_disp, hm3, region_score3, data3.batch, data3.aug_v0, cross_superv)

            loss_cross = criterion(cross_superv, cross_v) + criterion(cross_superv, cross_v_gtt)
            loss_cross = loss_cross * 0.5

            with torch.no_grad():
                cross_norm = get_normal_batch(cross_v, data3.triangle, data3.batch)
            pose_enc_cross = encoder_shape(cross_v, hm3, data=data3, feat=cross_norm)
            pose_enc_cross_0 = shape_enc2
            trans_cross = get_transformation(hm3, region_score3, data3.batch, data3.aug_v0, cross_v)
            pred_disp2 = decoder(pose_enc_cross-pose_enc_cross_0, shape_enc, trans_cross.detach())
            pred_v2 = handle2mesh(pred_disp2, hd_pos2.detach(), region_score2, data2.batch, data2.aug_v0)
            pred_v2_gtt = gt_trans_mesh(pred_disp2, hm2, region_score2, data2.batch, data2.aug_v0, data2.aug_v1)
            loss2 = criterion(data2.aug_v1, pred_v2) + criterion(data2.aug_v1, pred_v2_gtt)
            loss2 = loss2 * 0.5

            # visualize_data([data1.aug_v1, data2.aug_v0, data2.aug_v1, data3.v0, pred_v, cross_v, pred_v2],
            #                [data1.triangle, data2.triangle, data2.triangle, data3.triangle, data2.triangle, data3.triangle, data2.triangle],
            #                os.path.join(exp_dir, f'debug_{i_iter}.jpg'))
            # continue

            loss3 = transformation_loss(pred_disp, hm2, region_score2, data2.batch, data2.aug_v0, data2.aug_v1, criterion) + \
                    transformation_loss(pred_disp2, hm2, region_score2, data2.batch, data2.aug_v0, data2.aug_v1, criterion)

            loss_arap = arap_loss(data2.tpl_edge_index, data2.aug_v0, pred_v) * 100 + \
                        arap_loss(data3.tpl_edge_index, data3.aug_v0, cross_v) * 100 + \
                        arap_loss(data2.tpl_edge_index, data2.aug_v0, pred_v2) * 100
            fps_points0 = torch_geometric_fps(data2.aug_v0, data2.batch, num_handle*4)
            fps_points2 = torch_geometric_fps(data3.aug_v0, data3.batch, num_handle*4)
            # loss_hd_reg = -selfChamfer(hd_pos2, norm='L1') -selfChamfer(hd_pos3, norm='L1')
            loss_fps = distChamfer(fps_points0, hd_pos2, norm='L2') + distChamfer(fps_points2, hd_pos3, norm='L2')
            loss_sw = sw_loss(region_score2, data2.weights, data2.batch) + \
                      sw_loss(region_score3, data3.weights, data3.batch)
            loss_sw = loss_sw * 0.03

            loss_sw_sim = kl_div((region_score2+1e-6).log(), region_score2_noaug+1e-6)


            loss = loss1 + loss2 + loss3 + loss_cross * 0.3 + loss_sw + loss_sw_sim + loss_fps + loss_arap
            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Iter {i_iter}, invalid loss")
                # import IPython; IPython.embed()
                continue
            loss.backward()
            optimizer.step()

            loss_m.add_loss(loss1=loss1, loss2=loss2, loss_cross=loss_cross, loss_sw=loss_sw, loss_sw_sim=loss_sw_sim,
                            loss_fps=loss_fps, loss_arap=loss_arap)
            # loss_m.add_loss(loss1=loss1, loss2=loss2)
            if i_iter % 20 == 0:
                loss_m.print_latest()

        loss_m.epoch_summary()
        checkpoint = {"encoder_shape": encoder_shape.state_dict(),
                      "decoder": decoder.state_dict(),
                      "predictor": predictor.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      'torch_rnd': torch.get_rng_state(), 'numpy_rnd': np.random.get_state(), 'epoch': i_epoch}
        torch.save(checkpoint, os.path.join(exp_dir, 'latest.pth'))
        if i_epoch % 50 == 0:
            shutil.copy(os.path.join(exp_dir, 'latest.pth'), os.path.join(exp_dir, 'epoch_{}.pth'.format(i_epoch)))

        # validation
        if i_epoch % 20 != 0 and i_epoch != 1:
            continue

        render_args = {"trans": (1.5, 0, 0), "center": False}
        save_dir = os.path.join(exp_dir, 'val_{}'.format(i_epoch))
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            encoder_shape.eval()
            decoder.eval()
            predictor.eval()

            for i_iter, ((a_data1, a_data2), (r_data, _)) in enumerate(zip(test_amass_loader, test_rignet_loader)):
                if i_iter >= 10:
                    break
                a_data1.to(device)
                a_data2.to(device)
                r_data.to(device)

                hm0, hd0, _, region_score0 = predictor(torch.cat((r_data.v0, r_data.feat0), 1),
                                                       data=r_data, verbose=True)
                hm1, hd1, _, region_score1 = predictor(torch.cat((a_data1.v0, a_data1.feat0), 1),
                                                       data=a_data1, verbose=True)
                trans1 = get_transformation(hm1, region_score1, a_data1.batch, a_data1.v0, a_data1.v1)
                pose_enc_0 = encoder_shape(a_data1.v0, hm1, data=a_data1, feat=a_data1.feat0)
                pose_enc = encoder_shape(a_data1.v1, hm1, data=a_data1, feat=a_data1.feat1)
                shape_enc = encoder_shape(r_data.v0, hm0, data=r_data, feat=r_data.feat0)
                pred_disp = decoder(pose_enc-pose_enc_0, shape_enc, trans1)
                pred_v = handle2mesh(pred_disp, hd0, region_score0, r_data.batch, r_data.v0)

                # visualization
                if not no_render:
                    hd0 = hd0.cpu().numpy()[0]
                    v, f, vc = visualize_handle(r_data.v0.cpu().numpy(), r_data.triangle[0][0],
                                                save_path=os.path.join(save_dir, f'rignet_{i_iter}_input.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'rignet_{i_iter}_input.jpg'),
                                renderer(v, f, vc, **render_args))

                    Mesh(v=a_data1.v1.cpu().numpy(), f=a_data1.triangle[0][0]).write_ply(
                        os.path.join(save_dir, f'rignet_{i_iter}_pose.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'rignet_{i_iter}_pose.jpg'),
                                renderer(a_data1.v1.cpu().numpy(), a_data1.triangle[0][0], **render_args))

                    v, f, vc = visualize_handle(pred_v.cpu().numpy(), r_data.triangle[0][0],
                                                save_path=os.path.join(save_dir, f'rignet_{i_iter}_pred.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'rignet_{i_iter}_pred.jpg'),
                                renderer(v, f, vc, **render_args))

                    cv2.imwrite(os.path.join(save_dir, f'rignet_{i_iter}_seg.jpg'),
                                renderer(*visualize_part(r_data.v0, r_data.triangle[0][0], hd0, region_score0,
                                                         save_path=os.path.join(save_dir, f'rignet_{i_iter}_seg.ply'))))

            for i_iter, (data1, data2) in enumerate(test_mixamo_loader):
                data1.to(device)
                data2.to(device)
                hm0, hd0, _, region_score0 = predictor(torch.cat((data2.v0, data2.feat0), 1),
                                                       data=data2, verbose=True)
                hm1, hd1, _, region_score1 = predictor(torch.cat((data1.v0, data1.feat0), 1),
                                                       data=data1, verbose=True)
                trans1 = get_transformation(hm1, region_score1, data1.batch, data1.v0, data1.v1)
                pose_enc_0 = encoder_shape(data1.v0, hm1, data=data1, feat=data1.feat0)
                pose_enc = encoder_shape(data1.v1, hm1, data=data1, feat=data1.feat1)
                shape_enc = encoder_shape(data2.v0, hm0, data=data2, feat=data2.feat0)

                pred_disp = decoder(pose_enc-pose_enc_0, shape_enc, trans1)
                pred_v = handle2mesh(pred_disp, hd0, region_score0, data2.batch, data2.v0)

                dist = torch.mean(torch.sqrt(torch.sum((pred_v - data2.v1)**2, -1)))
                loss = criterion(pred_v, data2.v1)
                mixamo_loss_m.add_loss(mixamo_dist=dist, mixamo_loss=loss)

                # visualization
                if not no_render and i_iter < 10:
                    hd0 = hd0.cpu().numpy()[0]
                    v, f, vc = visualize_handle(data2.v1.cpu().numpy(), data2.triangle[0][0],
                                                save_path=os.path.join(save_dir, f'mixamo_{i_iter}_input.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter}_input.jpg'),
                                renderer(v, f, vc, **render_args))

                    Mesh(v=data1.v1.cpu().numpy(), f=data1.triangle[0][0]).write_ply(
                        os.path.join(save_dir, f'mixamo_{i_iter}_pose.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter}_pose.jpg'),
                                renderer(data1.v1.cpu().numpy(), data1.triangle[0][0], **render_args))


                    v, f, vc = visualize_handle(pred_v.cpu().numpy(), data2.triangle[0][0],
                                                save_path=os.path.join(save_dir, f'mixamo_{i_iter}_pred.ply'))
                    cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter}_pred.jpg'),
                                renderer(v, f, vc, **render_args))

                    cv2.imwrite(os.path.join(save_dir, f'mixamo_{i_iter}_seg.jpg'),
                                renderer(*visualize_part(data2.v0, data2.triangle[0][0], hd0, region_score0,
                                                         save_path=os.path.join(save_dir, f'mixamo_{i_iter}_seg.ply'))))

            mixamo_loss_m.epoch_summary()


if __name__ == '__main__':
    import signal
    try:
        main()
    except RuntimeError as e:
        print(e)
        os.kill(os.getpid(), signal.SIGINT)