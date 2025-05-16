
import os
import random
from tqdm import tqdm
import imageio
import numpy as np

from opt import config_parser
import torch
import torch.nn as nn
from utils import N_to_reso, rotation_ray, visualize_depth_numpy
# ----------------------------------------
# use this if loaded checkpoint is generate from single-light or rotated multi-light setting 
from models.tensoRF_rotated_lights import *

from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = GGX_specular
from utils import rgb_ssim, rgb_lpips
from models.relight_utils import Environment_Light
from renderer import compute_rescale_ratio, fill_ret


def relight(dataset, args):
    if not os.path.exists(args.ckpt):
        print('the checkpoint path for TensoLight does not exists!!!')
        return
        

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    TensoLight = eval(args.model_name)(dataset=dataset,**kwargs)
    TensoLight.load(ckpt)

    W, H = dataset.img_wh
    near_far = dataset.near_far
    
    rgb_frames_list = []

    aligned_albedo_list = []
    roughness_list = []

    envir_light = Environment_Light(args.hdrdir)
    light_rotation_idx = 0

    if 'real' not in args.expname:
        rescale_value = compute_rescale_ratio(TensoLight, envir_light, dataset, args)
    else:
        rescale_value = 2*torch.ones((3,3), device=device)
    

    relight_psnr = dict()
    relight_l_alex, relight_l_vgg, relight_ssim = dict(), dict(), dict() 
    for cur_light_name in dataset.light_names:
        relight_psnr[f'{cur_light_name}'] = []
        relight_l_alex[f'{cur_light_name}'] = []
        relight_l_vgg[f'{cur_light_name}'] = []
        relight_ssim[f'{cur_light_name}'] = []


    for idx in tqdm(range(len(dataset))):
        relight_pred_img_with_bg, relight_pred_img_without_bg, relight_gt_img = dict(), dict(), dict()
        for cur_light_name in dataset.light_names:
            relight_pred_img_with_bg[f'{cur_light_name}'] = []
            relight_pred_img_without_bg[f'{cur_light_name}'] = []
            relight_gt_img[f'{cur_light_name}'] = []

        cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)
        if 'real' not in args.expname:
            item = dataset[idx]
            rays_ = item['rays'].squeeze(0).to(device) # [H*W, 6]
            gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
            gt_mask = gt_mask > 0.5
            masks = gt_mask.view(H, W)
            frame_rays = rays_[gt_mask, :]
            gt_rgb = item['rgbs'].squeeze(0).view(len(light_name_list), -1, 3).cpu()  # [N, H, W, 3]
            light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)
        else:
            item = dataset[idx]
            rays_ = item['rays'].squeeze(0).to(device) # [H*W, 6]
            light_idx_ = item['light_idx'].to(torch.int32)
            rotation_matrix = torch.index_select(TensoLight.camera_rotation_matrix, 0, light_idx_.squeeze()) # [bs, 3, 3]
            rays_ = rotation_ray(rotation_matrix, rays_)
            gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
            gt_mask = gt_mask > 0.5
            masks = gt_mask.view(H, W)
            frame_rays = rays_[gt_mask, :]
            gt_rgb = item['rgbs'].squeeze(0).view(1, H*W, 3).expand(len(light_name_list), H*W, 3).cpu()  # [N, H, W, 3]
            light_idx = light_idx_[gt_mask, :].to(device)

        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], []
        acc_map = []

        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.no_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = TensoLight(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)


            relight_rgb_chunk = torch.ones_like(rgb_chunk)
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
            
            masked_normal_chunk = normal_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask] # [surface_point_num, 1]

            ## Get incident light directions
            for idx_, cur_light_name in enumerate(dataset.light_names):
                relight_rgb_chunk.fill_(1.0)
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 2048) # [bs, envW * envH, 3]
                surf2l = masked_light_dir                   # [surface_point_num, envW * envH, 3]
                surf2c = -rays_d_chunk[acc_chunk_mask]      # [surface_point_num, 3]
                surf2c = safe_l2_normalize(surf2c, dim=-1)  # [surface_point_num, 3]


                cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
                cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
                visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
                
                masked_surface_xyz = masked_surface_pts[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]

                cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
                indirect_light = torch.zeros((*cosine_mask.shape, 3), device=device)   # [bs, envW * envH, 3]
                masked_light_idx_ = masked_light_idx_chunk.reshape(-1, 1, 1).expand((*cosine_mask.shape, 1))
                with torch.no_grad():
                    visibility[cosine_mask], \
                        indirect_light[cosine_mask] = compute_secondary_shading_effects(
                                                                        TensoLight=TensoLight,
                                                                        surface_pts=cosine_masked_surface_pts,
                                                                        surf2light=cosine_masked_surf2l,
                                                                        light_idx=masked_light_idx_[cosine_mask],
                                                                        nSample=args.second_nSample,
                                                                        vis_near=args.second_near,
                                                                        vis_far=args.second_far,
                                                                        chunk_size=65536,
                                                                        device=device,
                                                                    )

                ## Get BRDF specs
                nlights = surf2l.shape[1]
                
                # relighting
                specular_relighting = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
                masked_albedo_chunk_rescaled = (masked_albedo_chunk * rescale_value[idx_, :])
                surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_relighting # [surface_point_num, envW * envH, 3]
               
                direct_light = masked_light_rgb 
                light_rgbs = visibility * direct_light
                light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] / masked_light_pdf
                surface_relight_rgb_chunk  = torch.mean(light_pix_contrib, dim=1)  # [bs, 3]

                ### Colorspace transform
                if surface_relight_rgb_chunk.shape[0] > 0:
                    surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
                relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk

                relight_without_bg = torch.ones_like(rays_d_chunk)
                relight_with_bg = torch.ones_like(rays_d_chunk)
                relight_without_bg[acc_chunk_mask] = relight_rgb_chunk[acc_chunk_mask]
                acc_temp = acc_chunk[..., None]
                acc_temp[acc_temp <= 0.9] = 0.0
                relight_with_bg = relight_without_bg.clone()

                relight_pred_img_with_bg[cur_light_name].append(relight_with_bg.detach().clone().cpu())
                relight_pred_img_without_bg[cur_light_name].append(relight_without_bg.detach().clone().cpu())


            rgb_map.append(rgb_chunk.cpu().detach())
            depth_map.append(depth_chunk.cpu().detach())
            acc_map.append(acc_chunk.cpu().detach())
            normal_map.append(normal_chunk.cpu().detach())
            albedo_map.append(albedo_chunk.cpu().detach())
            roughness_map.append(roughness_chunk.cpu().detach())

        rgb_map = torch.cat(rgb_map, dim=0)
        depth_map = torch.cat(depth_map, dim=0)
        acc_map = torch.cat(acc_map, dim=0)
        normal_map = torch.cat(normal_map, dim=0)
        acc_map_mask = (acc_map > args.acc_mask_threshold)
        albedo_map = torch.cat(albedo_map, dim=0)
        roughness_map = torch.cat(roughness_map, dim=0)
        
        rgb_map = fill_ret(rgb_map, rays_.shape[:-1], gt_mask, 1)
        depth_map = fill_ret(depth_map, rays_.shape[:-1], gt_mask, 0)
        normal_map = fill_ret(normal_map, rays_.shape[:-1], gt_mask, 0)
        albedo_map = fill_ret(albedo_map, rays_.shape[:-1], gt_mask, 0)
        roughness_map = fill_ret(roughness_map, rays_.shape[:-1], gt_mask, 0)
        acc_map = fill_ret(acc_map, rays_.shape[:-1], gt_mask, 0)
        acc_map_mask = fill_ret(acc_map_mask.float(), rays_.shape[:-1], gt_mask, 0) > 0
        os.makedirs(os.path.join(cur_dir_path, 'relighting_with_bg'), exist_ok=True)
        os.makedirs(os.path.join(cur_dir_path, 'relighting_without_bg'), exist_ok=True)
        os.makedirs(os.path.join(cur_dir_path, 'relighting_gt'), exist_ok=True)
        mask = acc_map_mask[:,None]
        masks = masks.to(acc_map_mask.device)
        for light_name_idx, cur_light_name in enumerate(dataset.light_names):
            relight_map_with_bg = torch.cat(relight_pred_img_with_bg[cur_light_name], dim=0)
            relight_map_with_bg = fill_ret(relight_map_with_bg, rays_.shape[:-1], gt_mask, 1)
            bg_color = envir_light.get_light(cur_light_name, rays_[:, 3:])
            bg_color = linear2srgb_torch(bg_color).cpu()
            relight_map_with_bg = relight_map_with_bg * mask + (~mask) * bg_color
            relight_map_with_bg = relight_map_with_bg.reshape(H, W, 3).numpy()
            relight_map_without_bg = torch.cat(relight_pred_img_without_bg[cur_light_name], dim=0)
            relight_map_without_bg = fill_ret(relight_map_without_bg, rays_.shape[:-1], gt_mask, 1).reshape(H, W, 3)
            acc_map_mask = acc_map_mask.reshape(H, W)
            gt_img_map = torch.ones(H*W,3)
            gt_img_map[gt_mask, :] = gt_rgb[light_name_idx, gt_mask, :]
            gt_img_map = gt_img_map.reshape(H, W, 3).numpy()

            relight_map_without_bg = relight_map_without_bg.numpy()

            if args.if_save_relight_rgb:
                imageio.imwrite(os.path.join(cur_dir_path, 'relighting_with_bg', f'{cur_light_name}.png'), (relight_map_with_bg * 255).astype('uint8'))
                imageio.imwrite(os.path.join(cur_dir_path, 'relighting_without_bg', f'{cur_light_name}.png'), (relight_map_without_bg * 255).astype('uint8'))

            # change the background color to white before computing metrics
            
            imageio.imwrite(os.path.join(cur_dir_path, 'relighting_gt', f'{cur_light_name}.png'), (gt_img_map * 255).astype('uint8'))
            loss_relight = np.mean((relight_map_without_bg[masks, :] - gt_img_map[masks, :]) ** 2)
            cur_psnr = -10.0 * np.log(loss_relight) / np.log(10.0)

            ssim_relight = rgb_ssim(relight_map_without_bg, gt_img_map, 1)
            l_a_relight = rgb_lpips(gt_img_map, relight_map_without_bg, 'alex', TensoLight.device)
            l_v_relight = rgb_lpips(gt_img_map, relight_map_without_bg, 'vgg', TensoLight.device)

            relight_psnr[cur_light_name].append(cur_psnr)
            relight_ssim[cur_light_name].append(ssim_relight)
            relight_l_alex[cur_light_name].append(l_a_relight)
            relight_l_vgg[cur_light_name].append(l_v_relight)


        # write relight image psnr to a txt file
        with open(os.path.join(cur_dir_path, 'relighting_without_bg', 'relight_psnr.txt'), 'w') as f:
            for cur_light_name in dataset.light_names:
                f.write(f'{cur_light_name}: PNSR {relight_psnr[cur_light_name][-1]}; SSIM {relight_ssim[cur_light_name][-1]}; L_Alex {relight_l_alex[cur_light_name][-1]}; L_VGG {relight_l_vgg[cur_light_name][-1]}\n')

        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgb_frames_list.append(rgb_map)
        depth_map, _ = visualize_depth_numpy(depth_map.reshape(H, W, 1).numpy(), near_far)
        acc_map = (acc_map.reshape(H, W, 1).numpy() * 255).astype('uint8')

        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgb_map)


        if args.if_save_depth:
            imageio.imwrite(os.path.join(cur_dir_path, 'depth.png'), depth_map)

        if args.if_save_albedo:
            albedo_map = albedo_map.reshape(H, W, 3)

            
            albedo_map_to_save = (albedo_map * 255).numpy().astype('uint8')
            albedo_map_to_save = np.concatenate([albedo_map_to_save, acc_map], axis=2).astype('uint8')
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo.png'), albedo_map_to_save)
            if args.if_save_albedo_gamma_corrected:
                to_save_albedo = (albedo_map ** (1/2.2) * 255).numpy().astype('uint8')
                to_save_albedo = np.concatenate([to_save_albedo, acc_map], axis=2)
                # gamma cororection
                imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_corrected.png'), to_save_albedo)

            aligned_albedo_list.append(((albedo_map ** (1.0/2.2)) * 255).numpy().astype('uint8'))

            roughness_map = roughness_map.reshape(H, W, 1)
            # expand to three channels
            roughness_map = (roughness_map.expand(-1, -1, 3) * 255)
            roughness_map = np.concatenate([roughness_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'roughness.png'), (roughness_map).astype('uint8'))
            roughness_list.append((roughness_map).astype('uint8'))
        if args.if_render_normal:
            normal_map = F.normalize(normal_map, dim=-1)
            normal_rgb_map = normal_map * 0.5 + 0.5
            normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            normal_rgb_map = np.concatenate([normal_rgb_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'normal.png'), normal_rgb_map)


    # write relight image psnr to a txt file
    m1=[]
    m2=[]
    m3=[]
    with open(os.path.join(args.geo_buffer_path, 'relight_psnr.txt'), 'w') as f:
        for cur_light_name in dataset.light_names:
            f.write(f'{cur_light_name}:  PSNR {np.mean(relight_psnr[cur_light_name])}; SSIM {np.mean(relight_ssim[cur_light_name])}; L_Alex {np.mean(relight_l_alex[cur_light_name])}; L_VGG {np.mean(relight_l_vgg[cur_light_name])}\n')
            m1.append(relight_psnr[cur_light_name])
            m2.append(relight_ssim[cur_light_name])
            m3.append(relight_l_alex[cur_light_name])
        f.write(f'all:  PSNR {np.mean(m1)}; SSIM {np.mean(m2)}; L_Alex {np.mean(m3)}\n')



if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))

    
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)
    random.seed(20211202)

    os.environ['PYTHONHASHSEED'] = str(20211202)


    # The following args are not defined in opt.py
    args.if_save_rgb = False
    args.if_save_depth = False
    args.if_save_acc = True
    args.if_save_rgb_video = False
    args.if_save_relight_rgb = True
    args.if_save_albedo = True
    args.if_save_albedo_gamma_corrected = True
    args.acc_mask_threshold = 0.5
    args.if_render_normal = True

    dataset = dataset_dict[args.dataset_name]

    test_dataset = dataset(                            
                            args.datadir, 
                            split='test', 
                            )
    light_name_list= test_dataset.light_names
    relight(test_dataset , args)