import cv2, torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    qw2, qx2, qy2, qz2 = qw**2, qx**2, qy**2, qz**2
    return np.array([
        [1 - 2*(qy2 + qz2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx2 + qz2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx2 + qy2)]
    ])


def parse_images_file(file_path):
    extrinsics = {}
    image_names = {}
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]

        for i in range(0, len(lines), 2):
            if i >= len(lines):
                break

            params = lines[i].split()
            if len(params) < 10:
                continue

            image_id = int(params[0]) - 1
            qw, qx, qy, qz = map(float, params[1:5])
            tx, ty, tz = map(float, params[5:8])
            image_names[image_id] = params[9]

            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = [tx, ty, tz]
            
            extrinsics[image_id] = extrinsic.astype(np.float32)
            
    return extrinsics, image_names

def parse_camera_file(file_path):
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]

        for i in range(0, len(lines), 2):
            if i >= len(lines):
                break
            
            params = lines[i].split()
            if len(params) < 10:
                continue
            w, h, fx, fy, cx, cy, k1, k2, p1, p2 = map(float, params[2:12])
            w, h = int(w), int(h)
            return w, h, fx, fy, cx, cy, k1, k2, p1, p2
        
@torch.no_grad()
def fill_ret(v, shape_, masks, key):
    if v.dim() > 1:
        if key:
            bkg = torch.ones(*shape_, v.shape[-1]).to(v.device)
            bkg[masks, :] = v
        else:
            bkg = torch.zeros(*shape_, v.shape[-1]).to(v.device)
            bkg[masks, :] = v
    else:
        bkg = torch.zeros(*shape_).to(v.device)
        bkg[masks] = v
            
    return bkg

def rotation_ray(M, rays):
    rot_rays = torch.zeros_like(rays)
    rot_rays[..., :3] = rotation(M, rays[..., :3])
    rot_rays[..., 3:] = rotation(M, rays[..., 3:], False)
    return rot_rays

def get_rotation_matrix(theta_x_deg, theta_y_deg, theta_z_deg):
    theta_x = torch.deg2rad(torch.tensor(theta_x_deg))
    theta_y = torch.deg2rad(torch.tensor(theta_y_deg))
    theta_z = torch.deg2rad(torch.tensor(theta_z_deg))

    Rx = torch.tensor([
        [1, 0, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
        [0, torch.sin(theta_x), torch.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])

    Ry = torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    
    Rz = torch.tensor([
        [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
        [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
    
    return rotation_matrix


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]

def rotation(M, x, is_pos=True):
    M = M.to(x.device)
    res = (M[..., :3, :3] @ x.unsqueeze(-1).to(torch.float32)).squeeze(-1)
    if is_pos:
        res += M[..., :3, 3]
    return res

def torchinv_trans(T):
    T_inv = torch.zeros_like(T).to(T.device)
    for i in range(T.shape[0]):

        matrix = T[i]

        R = matrix[:3, :3]
        t = matrix[:3, 3]

        R_inv = R.T

        t_inv = -torch.matmul(R_inv, t)

        T_inv[i, :3, :3] = R_inv
        T_inv[i, :3, 3] = t_inv
        T_inv[i, 3, 3] = 1.0 
        
    return T_inv

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET, mask=None):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1

    # TODO change mask to white
    if mask is not None:
        x[mask] = 1.0

    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)   # total volumes / number
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        # count_w = max(self._tensor_size(x[:,:,:,1:]), 1)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


# Multi-GPU training
import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

