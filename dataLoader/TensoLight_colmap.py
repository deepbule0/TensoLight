import os, random
import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import imageio.v3 as iio
import mitsuba as mi

from models.relight_utils import linear2srgb_torch
from utils import parse_camera_file, parse_images_file
mi.set_variant('scalar_rgb')

from skimage.transform import resize
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *



class TensoLight_Dataset_colmap(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 random_test=False,
                 N_vis=-1,
                 train_ratio=0.9,
                 downsample=4.0,
                 sub=0,
                 **temp,
                 ):
        """
        @param root_dir: str | Root path of dataset folder
        @param hdr_dir: str | Root path for HDR folder
        @param split: str | e.g. 'train' / 'test'
        @param random_test: bool | Whether to randomly select a test view and a lighting
        else [frames, h*w, 6]
        @param N_vis: int | If N_vis > 0, select N_vis frames from the dataset, else (-1) import entire dataset
        @param downsample: float | Downsample ratio for input rgb images
        """
        assert split in ['train', 'test']
        self.N_vis = N_vis
        obj = root_dir.split('obj')[-1]
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_path = os.path.join(self.root_dir, 'undistort')
        self.mask_path = os.path.join(self.root_dir, 'masks')
        self.poses, self.image_names = parse_images_file(os.path.join(self.root_dir, 'images.txt'))
        img_num = len([x for x in os.listdir(self.mask_path) if '.jpg' in x])
        split_num = int(img_num * train_ratio)
        lst = [[int(k), self.image_names[k]] for k in self.image_names.keys()]
        mask_names = os.listdir(self.mask_path)
        lst = [x[0] for x in lst if x[1] in mask_names]
        lst = sorted(lst)
        if split == 'train':
            if obj == '2':
                self.split_list = lst[:39]
            elif obj == '3':
                self.split_list = lst[:47]
            elif obj == '1':
                self.split_list = lst[:30]
        else:
            self.split_list = lst[::10] #[16:17]

        self.white_bg = False
        self.downsample = downsample
        self.transform = self.define_transforms()
        self.near_far = [0.05, 100]  
        self.scene_scale = 1.         
        if obj == '1':
            self.scene_bbox = torch.tensor([[-0.9, -1.2, 0.1],[0.5,  0.5,  1.5]])
        elif obj == '2':
            self.scene_bbox = torch.tensor([[-0.8, -1.33, -0.5],[0.5,  0.23,  0.8]])
        elif obj == '3':
            self.scene_bbox = torch.tensor([[-1., -1.55, -0.2],[0.2,  0.2,  1.]])

        self.AABB = torch.tensor([[-1., -1., -1.], [1., 1., 1.]]) * 18

        
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

        self.light_rotation = [0]
        self.light_num = -1
        self.light_num = 3
        self.light_names = ['lythwood', 'ballroom', 'cathedral']
        
        
        
        w, h, fx, fy, cx, cy, k1, k2, p1, p2 = parse_camera_file(os.path.join(self.root_dir, 'cameras.txt'))
        self.dist_coeffs = np.array([k1, k2, p1, p2])
        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        _, _, w_new, h_new = self.roi

        self.img_wh = (int(w_new / downsample), int(h_new / downsample))
        fx = self.new_camera_matrix[0, 0]
        fy = self.new_camera_matrix[1, 1]
        cx = self.new_camera_matrix[0, 2]
        cy = self.new_camera_matrix[1, 2]

        fx = fx / downsample
        fy = fy / downsample
        cx = cx / downsample
        cy = cy / downsample
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [fx, fy], [cx, cy]) 
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        
        ## Load light data
        self.read_lights()

        self.use_hdr = False
        # when trainning, we will load all the rays and rgbs
        if split == 'train':
            self.read_all_frames()    
                
    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms

    def read_lights(self):
        """
        Read hdr file from local path
        """
        self.lights_probes = None

        hdr_path = 'data/syn/teapot-unirough-0.2_in_bedroom_v2_rotated/env.exr'
        if os.path.exists(hdr_path):
            light_rgb = np.asarray(mi.Bitmap(str(hdr_path)))
            light_rgb[~np.isfinite(light_rgb)] = 0
            light_rgb = linear2srgb_torch(light_rgb)
            self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
            light_rgb = light_rgb.reshape(-1, 3)
            light_rgb = torch.from_numpy(light_rgb).float()
            self.lights_probes = light_rgb

    def read_all_frames(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_light_idx = []
        
        for idx in tqdm(range(self.__len__()), desc=f'Loading {self.split} data, view number: {self.__len__()}'):
            img_idx = self.split_list[idx]
            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            pose = self.poses[img_idx]
            pose = np.linalg.inv(pose)
            c2w = torch.from_numpy(pose)  # [4, 4]
            # Read ray data
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

            relight_img_path = os.path.join(self.img_path, self.image_names[img_idx])
            relight_img = np.array(Image.open(relight_img_path))
            mask_path = os.path.join(self.mask_path, self.image_names[img_idx])
            mask = Image.open(mask_path).convert('L')
            # print(mask_path)
            mask = np.array(mask)
            
            if self.downsample != 1.0:
                relight_img = resize(relight_img, (self.img_wh[1], self.img_wh[0], 3))
                mask = resize(mask, (self.img_wh[1], self.img_wh[0]))

            relight_img = self.transform(relight_img)  # [4, H, W]
            relight_rgbs = relight_img.view(3, -1).permute(1, 0)  # [H*W, 3]
            mask = self.transform(mask)  # [4, H, W]
            mask = mask.view(-1).unsqueeze(-1)

            light_idx = torch.zeros(self.img_wh[0] * self.img_wh[1], 1).to(torch.int8) # [H*W, 1], transform to in8 to save memory

            self.all_rays.append(rays)
            self.all_rgbs.append(relight_rgbs.to(torch.float))
            # self.all_masks.append(relight_mask)
            self.all_light_idx.append(light_idx)
            self.all_masks.append(mask.to(torch.float))

        self.all_rays = torch.cat(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        # self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]
        self.all_masks = torch.cat(self.all_masks, dim=0)


    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_stack(self):
        for idx in range(self.__len__()):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs]
        self.all_rays = torch.stack(self.all_rays, 0)  # [len(self), H*W, 6]
        self.all_rgbs = torch.stack(self.all_rgbs, 0)  # [len(self), H*W, 3]

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        img_idx = self.split_list[idx]
        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        pose = self.poses[img_idx]

        pose = np.linalg.inv(pose)

        c2w = torch.from_numpy(pose)  # [4, 4]
        w2c = torch.linalg.inv(c2w)
        # Read ray data
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

        relight_img_path = os.path.join(self.img_path, self.image_names[img_idx])
        relight_img = np.array(Image.open(relight_img_path))
        mask_path = os.path.join(self.mask_path, self.image_names[img_idx])
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        if self.downsample != 1.0:
            relight_img = resize(relight_img, (self.img_wh[1], self.img_wh[0], 3))
            mask = resize(mask, (self.img_wh[1], self.img_wh[0]))


        save_img = relight_img * 255
        # save_img = save_img * (mask[..., None] < 127.5)
        save_img = save_img.astype(np.uint8)
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(relight_img_path.replace('.jpg', '.png').replace('undistort', '0'), save_img)
        relight_img = self.transform(relight_img)  # [4, H, W]
        relight_rgbs = relight_img.view(3, -1).permute(1, 0)  # [H*W, 3]
        mask = self.transform(mask)  # [4, H, W]
        mask = mask.view(-1).unsqueeze(-1)

        light_idx = torch.zeros(self.img_wh[0] * self.img_wh[1], 1).to(torch.int8) 
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

        item = {
            'img_wh': self.img_wh,  # (int, int)
            'light_idx': light_idx,  # [rotation_num, H*W, 1]
            'rgbs': relight_rgbs.to(torch.float),  # [rotation_num, H*W, 3],
            'rgbs_mask': mask.to(torch.float),  # [H*W, 1]
            'rays': rays,  # [H*W, 6]
            'c2w': c2w,  # [4, 4]
            'w2c': w2c  # [4, 4]
        }
        return item

