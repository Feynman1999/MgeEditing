from megengine.functional.tensor import repeat
import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from edit.models.builder import BACKBONES
from .camera import get_camera_mat, get_random_pose
from .bounding_box_generator import BoundingBoxGenerator
from .common import arange_pixels, origin_to_world, image_points_to_world
from scipy.spatial.transform import Rotation as Rot
import math

@BACKBONES.register_module()
class GIRAFFE(M.Module):
    def __init__(self,
                 z_dim = 256,
                 z_dim_bg = 128,
                 n_boxes = 1,
                 decoder = None,
                 range_u = (0, 0),
                 range_v = (0.25, 0.25),
                 n_ray_samples = 64,
                 range_radius = (2.732, 2.732),
                 depth_range=[0.5, 6.],
                 background_generator=None,
                 resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 backround_rotation_range=[0., 0.],
                 sample_object_existance=False,
                 use_max_composition=False,
                 **kwargs
                 ):
        super(GIRAFFE, self).__init__()
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.fov = fov
        self.backround_rotation_range = backround_rotation_range
        self.sample_object_existance = sample_object_existance
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.use_max_composition = use_max_composition

        self.camera_matrix = get_camera_mat(fov=fov) # [1,4,4]

        # get decoder

        # get bounding_box_generator
        self.bounding_box_generator = BoundingBoxGenerator(n_boxes = n_boxes)

    def get_n_boxes(self):
        if self.bounding_box_generator is not None:
            n_boxes = self.bounding_box_generator.n_boxes
        else:
            n_boxes = 1
        return n_boxes

    def sample_z(self, size, tmp=1.):
        z = megengine.random.normal(size = size) * tmp
        return z

    def get_latent_codes(self, batch_size=32, tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        n_boxes = self.get_n_boxes()

        def sample_z(x):
            return self.sample_z(x, tmp=tmp)

        z_shape_obj = sample_z((batch_size, n_boxes, z_dim))
        z_app_obj = sample_z((batch_size, n_boxes, z_dim))
        z_shape_bg = sample_z((batch_size, z_dim_bg))
        z_app_bg = sample_z((batch_size, z_dim_bg))

        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def get_random_camera(self, batch_size=32):
        camera_mat = F.repeat(self.camera_matrix, repeats=batch_size, axis=0) # [b,4,4]
        world_mat = get_random_pose(self.range_u, self.range_v, self.range_radius, batch_size)
        # world_mat:  描述物体坐标系和相机坐标系的关系（旋转 + 平移）
        # 即每次随机一个pose，即可得到一个world_mat，用来把相机坐标系转为世界坐标系
        return camera_mat, world_mat

    def get_random_transformations(self, batch_size=32):
        s, t, R = self.bounding_box_generator(batch_size)
        return s, t, R

    def get_random_bg_rotation(self, batch_size):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [megengine.tensor(Rot.from_euler('z', r_random * 2 * np.pi).as_matrix()) for i in range(batch_size)]
            R_bg = F.stack(R_bg, axis=0).reshape(batch_size, 3, 3)
        else:
            R_bg = F.repeat(F.expand_dims(F.eye(3), axis=0).astype("float32"), repeats=batch_size, axis=0) # [b, 3, 3]
        return R_bg

    def add_noise_to_interval(self, di):
        """
            di: [b, npoints, n_steps]
        """
        di_mid = .5 * (di[..., 1:] + di[..., :-1]) # n_steps-1 个中点
        di_high = F.concat([di_mid, di[..., -1:]], axis=2) # b,npoints,n-1   b,npoints,1
        di_low = F.concat([di[..., :1], di_mid], axis=2) # b,npoints,1  b,npoints,n-1
        noise = megengine.random.normal(size = di_low.shape)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def forward(self, batch_size=32, latent_codes=None, camera_matrices=None,
                transformations=None, bg_rotation=None, mode="training", it=0,
                return_alpha_map=False,
                not_render_background=False,
                only_render_background=False):
        if latent_codes is None:
            """
                获得一个batch的z_shape和z_app，且object和bg维度不一样
                正太分布 0均值 1方差
            """
            latent_codes = self.get_latent_codes(batch_size)

        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)

        if transformations is None:
            transformations = self.get_random_transformations(batch_size)

        if bg_rotation is None:
            bg_rotation = self.get_random_bg_rotation(batch_size)

        if return_alpha_map:
            rgb_v, alpha_map = self.volume_render_image(
                latent_codes, camera_matrices, transformations, bg_rotation,
                mode=mode, it=it, return_alpha_map=True,
                not_render_background=not_render_background)
            return alpha_map
        else:
            rgb_v = self.volume_render_image(
                latent_codes, camera_matrices, transformations, bg_rotation,
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background)
            if self.neural_renderer is not None:
                rgb = self.neural_renderer(rgb_v)
            else:
                rgb = rgb_v
            return rgb

    def transform_points_to_box(self, p, transformations, box_idx=0, scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)).permute(0, 2, 1)).permute(0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def get_evaluation_points(self, pixels_world, camera_world, di, transformations, i):
        """
            pixels_world: b, points, 3
            camera_world: b, points, 3
            di: b, points, nsteps
        """
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(pixels_world, transformations, i)
        camera_world_i = self.transform_points_to_box(camera_world, transformations, i)
        ray_i = pixels_world_i - camera_world_i
        # b,points, 1, 3 + b,points,n,1 * b,points, 1, 3
        p_i = camera_world_i.unsqueeze(-2) + di.unsqueeze(-1) * ray_i.unsqueeze(-2)
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def volume_render_image(self, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='training',
                            it=0, return_alpha_map=False,
                            not_render_background=False,
                            only_render_background=False):
        res = self.resolution_vol
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes
        assert(not (not_render_background and only_render_background))

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size, invert_y_axis=False)[1]
        pixels[..., -1] *= -1. # invert_y
        # Project to 3D world
        pixels_world = image_points_to_world(pixels, camera_mat=camera_matrices[0], world_mat=camera_matrices[1])
        camera_world = origin_to_world(n_points, camera_mat=camera_matrices[0], world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + F.linspace(0., 1., num = n_steps).reshape(1, 1, -1) * (depth_range[1] - depth_range[0])
        di = F.broadcast_to(di, shape=[batch_size, n_points, n_steps])
        
        if mode == 'training':
            di = self.add_noise_to_interval(di)

        n_boxes = latent_codes[0].shape[1]
        feat, sigma = [], []
        n_iter = n_boxes if not_render_background else n_boxes + 1
        if only_render_background:
            n_iter = 1
            n_boxes = 0
        for i in range(n_iter):
            if i < n_boxes:  # Object
                p_i, r_i = self.get_evaluation_points(pixels_world, camera_world, di, transformations, i)
                z_shape_i, z_app_i = z_shape_obj[:, i], z_app_obj[:, i]

                feat_i, sigma_i = self.decoder(p_i, r_i, z_shape_i, z_app_i)

                if mode == 'training':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

                # Mask out values outside
                padd = 0.1
                mask_box = torch.all(
                    p_i <= 1. + padd, dim=-1) & torch.all(
                        p_i >= -1. - padd, dim=-1)
                sigma_i[mask_box == 0] = 0.

                # Reshape
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)
            else:  # Background
                p_bg, r_bg = self.get_evaluation_points_bg(
                    pixels_world, camera_world, di, bg_rotation)

                feat_i, sigma_i = self.background_generator(
                    p_bg, r_bg, z_shape_bg, z_app_bg)
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)

                if mode == 'training':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

            feat.append(feat_i)
            sigma.append(sigma_i)
        sigma = F.relu(torch.stack(sigma, dim=0))
        feat = torch.stack(feat, dim=0)

        if self.sample_object_existance:
            object_existance = self.get_object_existance(n_boxes, batch_size)
            # add ones for bg
            object_existance = np.concatenate(
                [object_existance, np.ones_like(
                    object_existance[..., :1])], axis=-1)
            object_existance = object_existance.transpose(1, 0)
            sigma_shape = sigma.shape
            sigma = sigma.reshape(sigma_shape[0] * sigma_shape[1], -1)
            object_existance = torch.from_numpy(object_existance).reshape(-1)
            # set alpha to 0 for respective objects
            sigma[object_existance == 0] = 0.
            sigma = sigma.reshape(*sigma_shape)

        # Composite
        sigma_sum, feat_weighted = self.composite_function(sigma, feat)

        # Get Volume Weights
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        if return_alpha_map:
            n_maps = sigma.shape[0]
            acc_maps = []
            for i in range(n_maps - 1):
                sigma_obj_sum = torch.sum(sigma[i:i+1], dim=0)
                weights_obj = self.calc_volume_weights(
                    di, ray_vector, sigma_obj_sum, last_dist=0.)
                acc_map = torch.sum(weights_obj, dim=-1, keepdim=True)
                acc_map = acc_map.permute(0, 2, 1).reshape(
                    batch_size, -1, res, res)
                acc_map = acc_map.permute(0, 1, 3, 2)
                acc_maps.append(acc_map)
            acc_map = torch.cat(acc_maps, dim=1)
            return feat_map, acc_map
        else:
            return feat_map
