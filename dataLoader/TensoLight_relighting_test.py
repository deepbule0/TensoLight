import os, random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *
import mitsuba as mi


mi.set_variant('scalar_rgb')

class TensoLight_Relighting_test(Dataset):
    def __init__(self,
                 root_dir,
                 hdr_dir=None,
                 split='train',
                 random_test=True,
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_rotation=['000', '045', '090', '135', '180', '225', '270', '315'],
                 light_names=["sunrise"]
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
        self.root_dir = Path(root_dir)
        self.img_wh = (int(512 / downsample), int(512 / downsample))  
        self.white_bg = True
        self.downsample = downsample
        self.transform = self.define_transforms()
        
        self.light_names = [x.split('_')[-1] for x in os.listdir(root_dir)]
        prefix = os.listdir(self.root_dir)[0].rsplit('_', 1)[0]  
        self.env_root = [os.path.join(self.root_dir, prefix +'_'+ x) for x in self.light_names]
        self.split = split

        self.len = 20
        self.near_far = [0.05, 100]  
        self.scene_scale = 1.
        self.scene_bbox = torch.tensor([[-self.scene_scale, -self.scene_scale, -self.scene_scale], [self.scene_scale, self.scene_scale, self.scene_scale]]) * self.downsample
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # HDR configs
        self.scan = self.root_dir.stem 

        self.light_rotation = [0]

        self.light_num = len(self.light_names)
        
        self.meta_path = os.path.join(self.env_root[0], 'transforms.json')
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.focal = 0.5 * int(self.meta['w']) / np.tan(0.5 * np.deg2rad(self.meta['x_fov']))  # fov -> focal length
        self.focal *= self.img_wh[0] / self.meta['w']
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.focal, self.focal])  # [H, W, 3]
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.directions = self.directions.to(torch.float)
        self.use_hdr = False
        
    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms



    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        meta = self.meta["frames"][idx]
        pose = np.array(meta["transform_matrix"]).astype(np.float32)
        pose[:3, 3] *= self.scene_scale
        pose = pose @ self.blender2opencv
        c2w = torch.from_numpy(pose).float()  # [4, 4]
        w2c = torch.linalg.inv(c2w)
        # Read ray data
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
        
        relight_rgbs_list = []
        light_idx_list = []
        for root in self.env_root:
            # Read RGB data
            relight_img_path = os.path.join(root, meta['file_path'].replace('.exr', '.png'))
            # relight_img = np.asarray(mi.Bitmap(str(relight_img_path)))
            # relight_img[~np.isfinite(relight_img)] = 0

            relight_img = np.array(Image.open(relight_img_path))
            mask_path = os.path.join(root, meta['mask_path'])
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            
            if self.downsample != 1.0:
                relight_img = np.resize(relight_img, self.img_wh)
                mask = np.resize(mask, self.img_wh)
                
            save_img = relight_img * 255
            # save_img = save_img * (mask[..., None] >= 127.5) + np.ones_like(save_img) * 255 * (mask[..., None] < 127.5)
            save_img = save_img.astype(np.uint8)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(relight_img_path.replace('.exr', '.png'), save_img)
            relight_img = self.transform(relight_img)  # [4, H, W]
            relight_rgbs = relight_img.view(3, -1).permute(1, 0)  # [H*W, 3]
            mask = self.transform(mask)  # [4, H, W]
            mask = mask.view(-1).unsqueeze(-1)
            # relight_rgbs = relight_img[:, :3] 
            light_idx = torch.tensor(0, dtype=torch.int).repeat((self.img_wh[0] * self.img_wh[1], 1)) # [H*W, 1]

            relight_rgbs_list.append(relight_rgbs)
            light_idx_list.append(light_idx)
            relight_mask = mask
        
        relight_rgbs = torch.stack(relight_rgbs_list, dim=0)    # [rotation_num, H*W, 3]
        light_idx = torch.stack(light_idx_list, dim=0)          # [rotation_num, H*W, 1]
        ## Obtain background mask, bg = False

    

        item = {
            'img_wh': self.img_wh,  # (int, int)
            'light_idx': light_idx,  # [light_num, H*W, 1]
            'rgbs': relight_rgbs,  # [light_num, H*W, 3],
            'rgbs_mask': relight_mask,  # [H*W, 1]
            'rays': rays,  # [H*W, 6]
            'c2w': c2w,  # [4, 4]
            'w2c': w2c  # [4, 4]
        }
        return item
