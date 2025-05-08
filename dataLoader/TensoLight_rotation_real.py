import math
import os, random
import json
from pathlib import Path
from tkinter import NO
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import mitsuba as mi

from models.relight_utils import linear2srgb_torch


mi.set_variant('scalar_rgb')

from skimage.transform import resize
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *
import torch.nn as nn

def toindex(x, exp):
    r = int(int(x.split('_')[1]) / 90)
    # lst = [0, 60, 122, 174]
    if exp == 'chair':
        lst = [0, 84, 166, 235]
    elif exp == 'dog1':
        lst = [0, 60, 122, 174]
    elif exp == 'dog2':
        lst = [0, 70, 137, 210]
    elif exp == 'bear':
        lst = [0, 65, 135, 202]
    r = lst[r] + int(x.split('_')[-1].split('.')[0]) - 1
    return r



class TensoLight_Dataset_rotated_lights_real(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 random_test=False,
                 N_vis=-1,
                 train_ratio=0.95,
                 downsample=1.0,
                 use_hdr=False,
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
        exp = root_dir.split('/')[-1]
        self.root_dir = Path(root_dir)
        self.split = split
        self.mask_path = os.path.join(self.root_dir, 'mask')
        self.split_list = [int(toindex(x, exp)) for x in os.listdir(self.mask_path) if '.png' in x]
        
        img_num = len(self.split_list)
        split_num = int(img_num * train_ratio)
        random.shuffle(self.split_list)
        if split == 'train':
            self.split_list = self.split_list[:split_num]
        else:                     
            self.split_list = self.split_list[split_num:]        

        
        self.img_wh = (int(512 / downsample), int(512 / downsample))  
        self.white_bg = True
        self.downsample = downsample
        self.transform = self.define_transforms()
        self.near_far = [0.05, 100]  
        self.meta_path = os.path.join(self.root_dir, 'transforms.json')
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)

        if exp == 'chair':
            self.scene_bbox = torch.tensor([[-0.8, -0.6, -1.1], [0.6,  2.0000,  0.7]])
        elif exp == 'dog1':
            self.scene_bbox = torch.tensor([[-0.5, -0.9, -1.],[0.7,  0.8,  0.4]])
        elif exp == 'dog2':
            self.scene_bbox = torch.tensor([[-0.7, -0.7, -1.2], [0.7,  0.7,  0.5]])
        elif exp == 'bear':
            self.scene_bbox = torch.tensor([[-0.7, -0.9, -1.2],[0.7,  0.8,  0.5]])
        self.AABB = torch.tensor([[-1., -1., -1.], [1., 1., 1.]]) * 16

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # HDR configs
        self.scan = self.root_dir.stem  
        self.light_rotation = [0, 90, 180, 270]
        self.light_num = len(self.light_rotation)
        self.light_rotation_matrix = []
        for k, pose in self.meta['rotations'].items():
            pose = torch.tensor(pose)

            pose = torch.linalg.inv(pose)
            self.light_rotation_matrix.append(pose)
            
        self.light_rotation_matrix = torch.stack(self.light_rotation_matrix, 0)

        ## Load light data
        self.read_lights()

        self.use_hdr = use_hdr
        self.light_names = ['lythwood', 'ballroom', 'cathedral']
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
            light_rgb = torch.from_numpy(light_rgb)
            light_rgb = linear2srgb_torch(light_rgb)
            self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
            light_rgb = light_rgb.reshape(-1, 3).float()
            self.lights_probes = light_rgb
    def comb_img(self, occlusion_image, background_image, img):
        occlusion = occlusion_image[..., :3]
        occlusion_mask = occlusion_image[..., 3:]
        background = background_image[..., :3]
        # mask = background
        # img = img + (1 - mask) * background
        img = occlusion + (1 - occlusion_mask) * img
        return img
    def read_all_frames(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_light_idx = []
        
        
        for idx in tqdm(range(self.__len__()), desc=f'Loading {self.split} data, view number: {self.__len__()}, rotaion number: {self.light_num}'):
            img_idx = self.split_list[idx]
            meta = self.meta["frames"][img_idx]
            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            pose = torch.tensor(meta["transform_matrix"]).to(torch.float)

            pose = pose 
            c2w = pose
            light_rotation_idx = self.light_rotation.index(int(meta['rotation']))
            c2w = c2w @ self.blender2opencv
            # Read ray data
            directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [meta['fl_x'], meta['fl_y']], [meta['cx'], meta['cy']])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

            # Read RGB data
            
            relight_img_path = os.path.join(self.root_dir, meta['file_path']).replace('.exr', '.png')
            relight_img = np.asarray(Image.open(str(relight_img_path)))
            mask_path = relight_img_path.replace('images', 'mask')
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
        
            if self.downsample != 1.0:
                relight_img = resize(relight_img, self.img_wh)
                mask = resize(mask, self.img_wh)
                

            relight_img = self.transform(relight_img)  # [4, H, W]
            relight_rgbs = relight_img.view(3, -1).permute(1, 0)  # [H*W, 3]
            mask = self.transform(mask)  # [4, H, W]
            mask = mask.view(-1).unsqueeze(-1)
            light_idx = torch.tensor(light_rotation_idx, dtype=torch.int8).repeat((self.img_wh[0] * self.img_wh[1], 1)).to(torch.int8) # [H*W, 1], transform to in8 to save memory

            self.all_rays.append(rays)
            self.all_rgbs.append(relight_rgbs)
            self.all_light_idx.append(light_idx)
            self.all_masks.append(mask)

        self.all_rays = torch.cat(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]


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
        img_idx= self.split_list[idx]
        
        meta = self.meta["frames"][img_idx]
        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        pose = np.array(meta["transform_matrix"]).astype(np.float32)
        # pose[:3, 3] *= self.scene_scale
        pose = pose @ self.blender2opencv
        c2w = torch.from_numpy(pose)  # [4, 4]
        w2c = torch.linalg.inv(c2w)
        # Read ray data
        directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [meta['fl_x'], meta['fl_y']], [meta['cx'], meta['cy']])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

        # Read RGB data
        light_rotation_idx = self.light_rotation.index(int(meta['rotation']))
        relight_img_path = os.path.join(self.root_dir, meta['file_path']).replace('.exr', '.png')
        relight_img = np.asarray(Image.open(str(relight_img_path)))
        mask_path = relight_img_path.replace('images', 'mask')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
    
        if self.downsample != 1.0:
            relight_img = resize(relight_img, self.img_wh)
            mask = resize(mask, self.img_wh)

        relight_img = self.transform(relight_img)  # [4, H, W]
        relight_rgbs = relight_img.view(3, -1).permute(1, 0)  # [H*W, 3]
        mask = self.transform(mask)  # [4, H, W]
        mask = mask.view(-1).unsqueeze(-1)
        light_idx = torch.tensor(light_rotation_idx, dtype=torch.int8).repeat((self.img_wh[0] * self.img_wh[1], 1)).to(torch.int8) # [H*W, 1], transform to in8 to save memory

        item = {
            'img_wh': self.img_wh,  # (int, int)
            'light_idx': light_idx,  # [rotation_num, H*W, 1]
            'rgbs': relight_rgbs,  # [rotation_num, H*W, 3],
            'rgbs_mask': mask,  # [H*W, 1]
            'rays': rays,  # [H*W, 6]
            'c2w': c2w,  # [4, 4]
            'w2c': w2c  # [4, 4]
        }
        return item


