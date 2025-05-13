import numpy as np
import random
import os, imageio
from tqdm.auto import tqdm
from utils import *
from models.relight_utils import compute_tensolight_shading_effects, linear2srgb_torch, render_with_BRDF
import torch
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.relight_utils import *
brdf_specular = GGX_specular

def compute_rescale_ratio(TensoLight, envir_light, dataset, args, sampled_num=20, batch_size=2048):
    W, H = dataset.img_wh
    light_name_list= dataset.light_names
    #### 
    light_rotation_idx = 0
    log_scale = torch.nn.Parameter(torch.zeros(3, 3, device=device))  
    
    optimizer = torch.optim.Adam([log_scale], lr=0.005)
    for idx in tqdm(range(sampled_num)):
        relight_pred_img_without_bg, relight_gt_img = dict(), dict()
        for cur_light_name in dataset.light_names:
            relight_pred_img_without_bg[f'{cur_light_name}'] = []
            relight_gt_img[f'{cur_light_name}'] = []

        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(device) # [H*W, 6]
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).to(device) # [H*W]
        gt_rgb = item['rgbs'].squeeze(0).view(len(light_name_list), H*W, 3).to(device) # [N, H, W, 3]
        gt_mask = gt_mask > 0
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)
        frame_rays = frame_rays[gt_mask, :]
        gt_rgb = gt_rgb[:, gt_mask, :]


        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), batch_size) # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.no_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = TensoLight(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)


            gt_rgb_chunk = gt_rgb[:, chunk_idx]
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
            
            masked_normal_chunk = normal_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask] # [surface_point_num, 1]

            for idx_, cur_light_name in enumerate(dataset.light_names):
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 1024) # [bs, envW * envH, 3]
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
                rescale_value = torch.exp(log_scale)
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

                loss = torch.mean((surface_relight_rgb_chunk - gt_rgb_chunk[idx_, acc_chunk_mask]) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(f'loss: {loss.detach().cpu()}')

    rescale_value = torch.exp(log_scale)
    print(rescale_value.detach())
    return rescale_value



def Renderer_TensoLight_train(  
                            rays=None, 
                            normal_gt=None, 
                            light_idx=None, 
                            TensoLight=None, 
                            N_samples=-1,
                            ndc_ray=False, 
                            white_bg=True, 
                            is_train=False,
                            is_relight=True,
                            train_light=False,
                            sample_method='fixed_envirmap',
                            chunk_size=15000,
                            device='cuda',      
                            args=None,
                        ):

   
    rays = rays.to(device)
    light_idx = light_idx.to(device, torch.int32)
    if not train_light:
        rgb_map, depth_map, normal_map, albedo_map, roughness_map, \
            fresnel_map, acc_map, normals_diff_map, normals_orientation_loss_map, \
            acc_mask, albedo_smoothness_loss, roughness_smoothness_loss \
            = TensoLight(rays, light_idx, is_train=is_train, white_bg=white_bg, is_relight=is_relight, ndc_ray=ndc_ray, N_samples=N_samples)

    else:
        rgb_map = TensoLight.get_tensolight_rgbs(rays[:, :3].contiguous(), rays[:, 3:].contiguous(), light_idx, device, True)

        ### Tonemapping
        rgb_map = torch.clamp(rgb_map, min=0.0, max=1.0)  
        ### Colorspace transform
        if rgb_map.shape[0] > 0:
            rgb_map = linear2srgb_torch(rgb_map)

    # Physically-based Rendering(Relighting)
    if is_relight:
        rgb_with_brdf_masked = render_with_BRDF(   
                                                depth_map[acc_mask],
                                                normal_map[acc_mask],
                                                albedo_map[acc_mask],
                                                roughness_map[acc_mask].repeat(1, 3),
                                                fresnel_map[acc_mask],
                                                rays[acc_mask],
                                                TensoLight,
                                                light_idx[acc_mask],
                                                sample_method,
                                                chunk_size=chunk_size,
                                                device=device,
                                                args=args
                                               )




        rgb_with_brdf = torch.ones_like(rgb_map) # background default to be white
        rgb_with_brdf[acc_mask] = rgb_with_brdf_masked

    else:
        rgb_with_brdf = torch.ones_like(rgb_map)


    if not train_light:
        ret_kw = {
            "rgb_map": rgb_map,
            "depth_map": depth_map,
            "normal_map": normal_map,
            "albedo_map": albedo_map,
            "acc_map": acc_map,
            "roughness_map": roughness_map,
            "fresnel_map": fresnel_map,
            'rgb_with_brdf_map': rgb_with_brdf,
            'normals_diff_map': normals_diff_map,
            'normals_orientation_loss_map': normals_orientation_loss_map,
            'albedo_smoothness_loss': albedo_smoothness_loss,
            'roughness_smoothness_loss': roughness_smoothness_loss,
        }
    else:
        ret_kw = {"rgb_map": rgb_map}
        
    return ret_kw






@torch.no_grad()
def evaluation_iter_TensoLight(
        test_dataset,
        TensoLight,
        args,
        renderer,
        savePath=None,
        prtx='',
        N_samples=-1,
        white_bg=False,
        ndc_ray=False,
        compute_extra_metrics=True,
        device='cuda',
        logger=None,
        step=None,
        test_all=False,
):

    
    PSNRs_rgb, rgb_maps, depth_maps, gt_maps, gt_rgb_brdf_maps = [], [], [], [], []
    PSNRs_rgb_brdf = []
    rgb_with_brdf_maps, normal_rgb_maps, normal_rgb_vis_maps= [], [], []
    albedo_maps, single_aligned_albedo_maps, three_aligned_albedo_maps, roughness_maps, fresnel_maps, normals_diff_maps  =  [], [], [], [], [], []
    normal_raw_list = []

    ssims, l_alex, l_vgg = [], [], []
    ssims_rgb_brdf, l_alex_rgb_brdf, l_vgg_rgb_brdf = [], [], []


    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/nvs_with_radiance_field", exist_ok=True)
    os.makedirs(savePath + "/nvs_with_brdf", exist_ok=True)
    os.makedirs(savePath + "/normal", exist_ok=True)
    os.makedirs(savePath + "/normal_vis", exist_ok=True)
    os.makedirs(savePath + "/brdf", exist_ok=True)
    os.makedirs(savePath + "/envir_map/", exist_ok=True)
    os.makedirs(savePath + "/acc_map", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh
    
    num_test = len(test_dataset) if test_all else min(args.N_vis, len(test_dataset))

    gt_envir_map = None
    envH = 256  # TensoLight.envmap_h
    envW = 512  # TensoLight.envmap_w
    if test_dataset.lights_probes is not None:
        gt_envir_map = test_dataset.lights_probes.reshape(test_dataset.envir_map_h, test_dataset.envir_map_w, 3).numpy()
        # gt_envir_map = linear2srgb_torch(gt_envir_map)
        gt_envir_map = np.uint8(gt_envir_map * 255.)
        
        # resize to envW * 512
        gt_envir_map = cv2.resize(gt_envir_map, (envW, envH), interpolation=cv2.INTER_CUBIC)

    view_dirs = TensoLight.generate_envir_map_dir(envW, envH)
    v = view_dirs.reshape(-1, 3)
    o = torch.zeros_like(v, dtype=v.dtype)
    idx = torch.zeros((v.shape[0], 1), dtype=torch.int)
    predicted_envir_map = compute_tensolight_shading_effects(TensoLight, o, v, idx, args.relight_chunk_size, device)
    predicted_envir_map = linear2srgb_torch(predicted_envir_map)

    predicted_envir_map = predicted_envir_map.reshape(envH, envW, 3).cpu().detach().numpy()
    predicted_envir_map = np.clip(predicted_envir_map, a_min=0, a_max=1.)

    predicted_envir_map = np.uint8(predicted_envir_map * 255.)
    if gt_envir_map is not None:
        envirmap = np.concatenate((gt_envir_map, predicted_envir_map), axis=1)
    else:
        envirmap = predicted_envir_map
    # save predicted envir map
    imageio.imwrite(f'{savePath}/envir_map/{prtx}envirmap.png', envirmap)
    test_duration = int(len(test_dataset) / num_test)

    for idx in range(num_test):
        if test_all:
            print(f"test {idx} / {num_test}")
        item = test_dataset.__getitem__(idx * test_duration)
        rays_ = item['rays']                 # [H*W, 6]
        gt_rgb_ = item['rgbs']            # [H*W, 3]
        light_idx_ = item['light_idx']   # [H*W, 1]

        rotation_matrix = torch.index_select(TensoLight.camera_rotation_matrix, 0, light_idx_.squeeze().to(torch.int32)) # [bs, 3, 3]
        rays_ = rotation_ray(rotation_matrix, rays_)

        gt_mask = item['rgbs_mask']         # [H*W, 1]
        gt_masks = (gt_mask > 0.5).squeeze()
        mask = gt_masks.view(H, W)
        
        gt_white = torch.ones_like(gt_rgb_).to(gt_rgb_.device)
        gt_rgb = torch.where(gt_masks.unsqueeze(-1), gt_rgb_, gt_white)
        gt_rgb_wirh_brdf = gt_rgb           # [H*W, 3]

        rays = rays_[gt_masks, :]
        light_idx = light_idx_[gt_masks, :]

        rgb_map, acc_map, depth_map, normal_map, albedo_map, roughness_map = [], [], [], [], [], []
        fresnel_map, rgb_with_brdf_map, normals_diff_map = [], [], []
        chunk_idxs = torch.split(torch.arange(rays.shape[0]), args.batch_size_test)
        for chunk_idx in tqdm(chunk_idxs):
            ret_kw= renderer(   
                                rays[chunk_idx], 
                                None, # not used
                                light_idx[chunk_idx],
                                TensoLight, 
                                N_samples=N_samples,
                                ndc_ray=ndc_ray,
                                white_bg=white_bg,
                                sample_method=args.light_sample_train,
                                chunk_size=args.relight_chunk_size,  
                                device=device,
                                args=args
                            )
            
            rgb_map.append(ret_kw['rgb_map'].detach().cpu())
            depth_map.append(ret_kw['depth_map'].detach().cpu())
            normal_map.append(ret_kw['normal_map'].detach().cpu())
            albedo_map.append(ret_kw['albedo_map'].detach().cpu())
            roughness_map.append(ret_kw['roughness_map'].detach().cpu())
            fresnel_map.append(ret_kw['fresnel_map'].detach().cpu())
            rgb_with_brdf_map.append(ret_kw['rgb_with_brdf_map'].detach().cpu())
            normals_diff_map.append(ret_kw['normals_diff_map'].detach().cpu())

            acc_map.append(ret_kw['acc_map'].detach().cpu())

        
        rgb_map = torch.cat(rgb_map)
        depth_map = torch.cat(depth_map)
        normal_map = torch.cat(normal_map)
        albedo_map = torch.cat(albedo_map)
        roughness_map = torch.cat(roughness_map)
        fresnel_map = torch.cat(fresnel_map)
        rgb_with_brdf_map = torch.cat(rgb_with_brdf_map)
        normals_diff_map = torch.cat(normals_diff_map)

        acc_map = torch.cat(acc_map)
        
        rgb_map = fill_ret(rgb_map, rays_.shape[:-1], gt_masks, 1)
        rgb_with_brdf_map = fill_ret(rgb_with_brdf_map, rays_.shape[:-1], gt_masks, 1)
        depth_map = fill_ret(depth_map, rays_.shape[:-1], gt_masks, 0)
        normal_map = fill_ret(normal_map, rays_.shape[:-1], gt_masks, 0)
        albedo_map = fill_ret(albedo_map, rays_.shape[:-1], gt_masks, 1)
        roughness_map = fill_ret(roughness_map, rays_.shape[:-1], gt_masks, 1)
        fresnel_map = fill_ret(fresnel_map, rays_.shape[:-1], gt_masks, 0)
        normals_diff_map = fill_ret(normals_diff_map , rays_.shape[:-1], gt_masks, 0)
        acc_map = fill_ret(acc_map, rays_.shape[:-1], gt_masks, 0)


        acc_map = acc_map.reshape(H, W).detach().cpu()

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).detach().cpu(), depth_map.reshape(H, W).detach().cpu()
        rgb_with_brdf_map = rgb_with_brdf_map.reshape(H, W, 3).detach().cpu()
        albedo_map = albedo_map.reshape(H, W, 3).detach().cpu()

        single_aligned_albedo_map = torch.ones_like(albedo_map)
        three_aligned_albedo_map = torch.ones_like(albedo_map)

        roughness_map = roughness_map.reshape(H, W, 1).repeat(1, 1, 3).detach().cpu()
        fresnel_map = fresnel_map.reshape(H, W, 3).detach().cpu()
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        mask = mask.to(rgb_map.device)
        # Store loss and images
        if test_dataset.__len__():
            gt_rgb = gt_rgb.view(H, W, 3)
            gt_rgb = (gt_rgb * 255).to(torch.uint8)
            gt_rgb = gt_rgb.float() / 255.
            gt_rgb_wirh_brdf = gt_rgb
            rgb_with_brdf_map = (rgb_with_brdf_map * 255).to(torch.uint8)
            rgb_with_brdf_map = rgb_with_brdf_map.float() / 255.

            loss_rgb = torch.mean((rgb_map[mask, :] - gt_rgb[mask, :]) ** 2)
            loss_rgb_brdf = torch.mean((rgb_with_brdf_map[mask, :] - gt_rgb_wirh_brdf[mask, :]) ** 2)
            if loss_rgb_brdf < 1e-6:
                continue
            PSNRs_rgb.append(-10.0 * np.log(loss_rgb.item()) / np.log(10.0))
            PSNRs_rgb_brdf.append(-10.0 * np.log(loss_rgb_brdf.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', TensoLight.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', TensoLight.device)

                ssim_rgb_brdf = rgb_ssim(rgb_with_brdf_map, gt_rgb_wirh_brdf, 1)
                l_a_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'alex', TensoLight.device)
                l_v_rgb_brdf = rgb_lpips(gt_rgb_wirh_brdf.numpy(), rgb_with_brdf_map.numpy(), 'vgg', TensoLight.device)

                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

                ssims_rgb_brdf.append(ssim_rgb_brdf)
                l_alex_rgb_brdf.append(l_a_rgb_brdf)
                l_vgg_rgb_brdf.append(l_v_rgb_brdf)


        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_with_brdf_map = (rgb_with_brdf_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')
        gt_rgb_wirh_brdf = (gt_rgb_wirh_brdf.numpy() * 255).astype('uint8')
        albedo_map = (albedo_map.numpy() * 255).astype('uint8')
        roughness_map = (roughness_map.numpy() * 255).astype('uint8')
        fresnel_map = (fresnel_map.numpy() * 255).astype('uint8')
        acc_map = (acc_map.numpy() * 255).astype('uint8')

        normal_map = F.normalize(normal_map, dim=-1)
        normal_raw_list.append(normal_map)

        normal_rgb_map = normal_map * 0.5 + 0.5 # map from [-1, 1] to [0, 1] to visualize
        normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).cpu().numpy() * 255).astype('uint8')
        normal_rgb_vis_map = (normal_rgb_map * (acc_map[:, :, None] / 255.0) + (1 -(acc_map[:, :, None] / 255.0)) * 255).astype('uint8') # white background

        # difference between the predicted normals and derived normals
        normals_diff_map = (torch.clamp(normals_diff_map, 0.0, 1.0).reshape(H, W, 1).repeat(1, 1, 3).numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        rgb_with_brdf_maps.append(rgb_with_brdf_map)
        depth_maps.append(depth_map)
        gt_maps.append(gt_rgb)
        gt_rgb_brdf_maps.append(gt_rgb_wirh_brdf)
        normal_rgb_maps.append(normal_rgb_map)
        normal_rgb_vis_maps.append(normal_rgb_vis_map)
 
        if not test_all:
            normals_diff_maps.append(normals_diff_map)


        albedo_maps.append(albedo_map)
        single_aligned_albedo_maps.append((single_aligned_albedo_map.numpy())**(1/2.2))
        three_aligned_albedo_maps.append((three_aligned_albedo_map.numpy())**(1/2.2))

        roughness_maps.append(roughness_map)
        fresnel_maps.append(fresnel_map)

        normal_map = (normal_map.numpy() * 255).astype(np.uint8)
        if savePath is not None:
            rgb_map = np.concatenate((rgb_map, gt_rgb, depth_map), axis=1)
            rgb_with_brdf_map = np.concatenate((rgb_with_brdf_map, gt_rgb_wirh_brdf), axis=1)

            normal_map = np.concatenate((normal_rgb_map, normals_diff_map), axis=1)
            brdf_map = np.concatenate((albedo_map, roughness_map, fresnel_map), axis=1)
           
            imageio.imwrite(f'{savePath}/nvs_with_radiance_field/{prtx}{idx:03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/nvs_with_brdf/{prtx}{idx:03d}.png', rgb_with_brdf_map)
            imageio.imwrite(f'{savePath}/normal/{prtx}{idx:03d}.png', normal_map)
            imageio.imwrite(f'{savePath}/normal_vis/{prtx}{idx:03d}.png', normal_rgb_vis_map)
            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}.png', brdf_map)

            imageio.imwrite(f'{savePath}/brdf/{prtx}{idx:03d}_roughness.png', roughness_map)
            imageio.imwrite(f'{savePath}/acc_map/{prtx}{idx:03d}.png', acc_map)


    # Randomly select a prediction to visualize
    if logger and step and not test_all:
        vis_idx = random.choice(range(len(rgb_maps)))
        vis_rgb = torch.from_numpy(rgb_maps[vis_idx])
        vis_rgb_brdf_rgb = torch.from_numpy(rgb_with_brdf_maps[vis_idx])
        vis_depth = torch.from_numpy(depth_maps[vis_idx])
        vis_rgb_gt = torch.from_numpy(gt_maps[vis_idx])

        vis_albedo = torch.from_numpy(albedo_maps[vis_idx])

        vis_roughness = torch.from_numpy(roughness_maps[vis_idx])
        vis_fresnel = torch.from_numpy(fresnel_maps[vis_idx])
        vis_rgb_grid = torch.stack([vis_rgb, vis_rgb_brdf_rgb, vis_rgb_gt, vis_depth]).permute(0, 3, 1, 2).to(float)
   
        vis_brdf_grid = torch.stack([vis_albedo, vis_roughness, vis_fresnel]).permute(0, 3, 1, 2).to(float)
        vis_envir_map_grid = torch.from_numpy(envirmap).unsqueeze(0).permute(0, 3, 1, 2).to(float)


        logger.add_image('test/rgb',
                            vutils.make_grid(vis_rgb_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/brdf',
                            vutils.make_grid(vis_brdf_grid, padding=0, normalize=True, value_range=(0, 255)), step)
        logger.add_image('test/envir_map',
                            vutils.make_grid(vis_envir_map_grid, padding=0, normalize=True, value_range=(0, 255)), step)


    # Compute metrics
    if PSNRs_rgb:
        psnr = np.mean(np.asarray(PSNRs_rgb))
        psnr_rgb_brdf = np.mean(np.asarray(PSNRs_rgb_brdf))
       
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))

            ssim_rgb_brdf = np.mean(np.asarray(ssims_rgb_brdf))
            l_a_rgb_brdf = np.mean(np.asarray(l_alex_rgb_brdf))
            l_v_rgb_brdf = np.mean(np.asarray(l_vgg_rgb_brdf))



            saved_message = f'Iteration:{prtx[:-1]}: \n' \
                            + f'\tPSNR_nvs: {psnr:.4f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.4f}' \
                            + f'\tSSIM_rgb: {ssim:.4f}, L_Alex_rgb: {l_a:.4f}, L_VGG_rgb: {l_v:.4f}\n' \
                            + f'\tSSIM_rgb_brdf: {ssim_rgb_brdf:.4f}, L_Alex_rgb_brdf: {l_a_rgb_brdf:.4f}, L_VGG_rgb_brdf: {l_v_rgb_brdf:.4f}\n' \
                        

        else:
            saved_message = f'Iteration:{prtx[:-1]}, PSNR_nvs: {psnr:.4f}, PSNR_nvs_brdf: {psnr_rgb_brdf:.4f}\n'

        with open(f'{savePath}/metrics_record.txt', 'a') as f:
            f.write(saved_message)

    return psnr, psnr_rgb_brdf