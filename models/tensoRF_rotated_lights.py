from .tensorBase_rotated_lights import *
from .relight_utils import grid_sample



class TensorVMSplit(TensorBase):
    def __init__(self, aabb, AABB, gridSize, light_gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, AABB, gridSize, light_gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        # used for factorize the radiance fields under different lighting conditions
        self.basis_mat1 = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        self.light_line = torch.nn.Embedding(self.light_num, sum(self.app_n_comp)).to(device) # (light_num, sum(self.app_n_comp)), such as (10, 16+16+16)

    def init_tensolight(self, device):
        self.light_density_plane, self.light_density_line = self.init_one_svd(self.density_n_comp_light, self.gridSize_light, self.scale_light, device)
        self.light_app_plane, self.light_app_line = self.init_one_svd(self.app_n_comp_light, self.gridSize_light, self.scale_light, device)
        self.light_basis_mat = torch.nn.Linear(sum(self.app_n_comp_light), self.app_dim_light, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [   {'params': self.density_line, 'lr': lr_init_spatialxyz}, 
                        {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                        {'params': self.app_line, 'lr': lr_init_spatialxyz}, 
                        {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                        {'params': self.basis_mat.parameters(), 'lr':lr_init_network},
                        {'params': self.basis_mat1.parameters(), 'lr':lr_init_network}]
        # TODO: merge the learing rates of the following two groups into config file
        grad_vars += [  {'params': self.light_line.parameters(), 'lr':0.001}]

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
            
        grad_vars += [{'params': self.light_density_line, 'lr': lr_init_spatialxyz}, 
                          {'params': self.light_density_plane, 'lr': lr_init_spatialxyz},
                          {'params': self.light_app_line, 'lr': lr_init_spatialxyz}, 
                          {'params': self.light_app_plane, 'lr': lr_init_spatialxyz},
                          {'params': self.light_basis_mat.parameters(), 'lr':lr_init_network},
                          {'params':self.light_renderModule.parameters(), 'lr':lr_init_network}]
   
        if isinstance(self.renderModule_brdf, torch.nn.Module):
            grad_vars += [{'params':self.renderModule_brdf.parameters(), 'lr':lr_init_network}]

        if (self.normals_kind == "purely_predicted" or self.normals_kind == "derived_plus_predicted" or self.normals_kind == "residue_prediction") \
                and isinstance(self.renderModule_normal, torch.nn.Module):
            grad_vars += [{'params':self.renderModule_normal.parameters(), 'lr':lr_init_network}]

        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
            # total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
            # total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total


    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n] [component_num, point_num]
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n]
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_densityfeature_with_xyz_grad(self, xyz_sampled):
        # the diffenrence this function and compute_densityfeature() is .detach() is removed, 
        # and this function replace F.grid_sample with a rewritten function, which support second order gradient

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            # The current F.grif_sample in pytorch doesn't support backpropagation
            plane_coef_point = grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]]).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n] [component_num, point_num]
            line_coef_point = grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]]).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n]
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_bothfeature(self, xyz_sampled, light_idx):
        '''
        args:
            xyz_sampled: (sampled_pts, 3)
            light_idx: (sampled_pts, )
        '''
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))

            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        
        light_coef_point = self.light_line(light_idx.to(xyz_sampled.device)).squeeze(1).permute(1,0)

        radiance_field_feat_update = self.basis_mat1((plane_coef_point * line_coef_point * light_coef_point).T)
        
        intrinsic_feat = self.basis_mat((plane_coef_point * line_coef_point).T)

        radiance_field_feat = intrinsic_feat + radiance_field_feat_update
        return radiance_field_feat, intrinsic_feat

    def compute_intrinfeature(self, xyz_sampled):
        '''
        args:
            xyz_sampled: (sampled_pts, 3)
        '''


        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))

            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        
        
        intrinsic_feat = self.basis_mat((plane_coef_point * line_coef_point).T)
        
        return intrinsic_feat

    def compute_appfeature(self, xyz_sampled, light_idx):
        '''
        args:
            xyz_sampled: (sampled_pts, 3)
            light_idx: (sampled_pts, )
        '''
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))

            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        
        light_coef_point = self.light_line(light_idx.to(xyz_sampled.device)).squeeze(1).permute(1,0)

        radiance_field_feat_o = self.basis_mat((plane_coef_point * line_coef_point).T)
        radiance_field_feat_update = self.basis_mat1((plane_coef_point * line_coef_point * light_coef_point).T)
        radiance_field_feat = radiance_field_feat_o + radiance_field_feat_update

        return radiance_field_feat

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]

            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )


            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
        
        
    @torch.no_grad()
    def shrink_light(self, new_AABB):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_AABB
        t_l, b_r = (xyz_min - self.AABB[0]) / self.units_light, (xyz_max - self.AABB[0]) / self.units_light
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize_light]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]

            self.light_density_line[i] = torch.nn.Parameter(
                self.light_density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.light_app_line[i] = torch.nn.Parameter(
                self.light_app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )


            mode0, mode1 = self.matMode[i]
            self.light_density_plane[i] = torch.nn.Parameter(
                self.light_density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.light_app_plane[i] = torch.nn.Parameter(
                self.light_app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask_light.gridSize == self.gridSize_light):
            t_l_r, b_r_r = t_l / (self.gridSize_light-1), (b_r-1) / (self.gridSize_light-1)
            correct_AABB = torch.zeros_like(new_AABB)
            correct_AABB[0] = (1-t_l_r)*self.AABB[0] + t_l_r*self.AABB[1]
            correct_AABB[1] = (1-b_r_r)*self.AABB[0] + b_r_r*self.AABB[1]
            print("AABB", new_AABB, "\ncorrect AABB", correct_AABB)
            new_AABB = correct_AABB

        newSize = b_r - t_l
        self.AABB = new_AABB
        self.update_light_stepSize((newSize[0], newSize[1], newSize[2]))
        # self.nSamples_light = 96
        
    def compute_light_appfeature(self, xyz_sampled):
        '''
        args:
            xyz_sampled: (sampled_pts, 3)
            light_idx: (sampled_pts, )
        '''
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.light_app_plane)):
            plane_coef_point.append(F.grid_sample(self.light_app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))

            line_coef_point.append(F.grid_sample(self.light_app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        radiance_field_feat = self.light_basis_mat((plane_coef_point * line_coef_point).T)
        
        return radiance_field_feat
    
    def compute_light_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.light_density_plane)):
            plane_coef_point = F.grid_sample(self.light_density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n] [component_num, point_num]
            line_coef_point = F.grid_sample(self.light_density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])  # [16, h*w*n]
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature
    
    def light_TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.light_app_plane)):
            total = total + reg(self.light_app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
            # total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3
        return total
    
    def light_TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.light_density_plane)):
            total = total + reg(self.light_density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
            # total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3
        return total
    
    def light_density_L1(self):
        total = 0
        for idx in range(len(self.light_density_plane)):
            total = total + torch.mean(torch.abs(self.light_density_plane[idx])) + torch.mean(torch.abs(self.light_density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def light_vector_comp_diffs(self):
        return self.vectorDiffs(self.light_density_line) + self.vectorDiffs(self.light_app_line)

