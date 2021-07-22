import megengine
import numpy as np
import megengine.module as M
from scipy.spatial.transform import Rotation as Rot
import megengine.functional as F
# from .camera import get_rotation_matrix

class BoundingBoxGenerator(M.Module):
    ''' Bounding box generator class

    Args:
        n_boxes (int): number of bounding boxes (excluding background)
        scale_range_min (list): min scale values for x, y, z
        scale_range_max (list): max scale values for x, y, z
        translation_range_min (list): min values for x, y, z translation
        translation_range_max (list): max values for x, y, z translation
        z_level_plane (float): value of z-plane; only relevant if
            object_on_plane is set True
        rotation_range (list): min and max rotation value (between 0 and 1)
        check_collision (bool): whether to check for collisions
        collision_padding (float): padding for collision checking
        fix_scale_ratio (bool): whether the x/y/z scale ratio should be fixed
        object_on_plane (bool): whether the objects should be placed on a plane
            with value z_level_plane
        prior_npz_file (str): path to prior npz file (used for clevr) to sample
            locations from
    '''

    def __init__(self, n_boxes=1,
                 scale_range_min=[0.5, 0.5, 0.5],
                 scale_range_max=[0.5, 0.5, 0.5],
                 translation_range_min=[-0.75, -0.75, 0.],
                 translation_range_max=[0.75, 0.75, 0.],
                 z_level_plane=0., rotation_range=[0., 1.],
                 check_collison=False, collision_padding=0.1,
                 fix_scale_ratio=True, object_on_plane=False,
                 prior_npz_file=None, **kwargs):
        super().__init__()

        self.n_boxes = n_boxes
        self.scale_min = megengine.tensor(scale_range_min, dtype="float32").reshape(1, 1, 3)
        self.scale_range = (megengine.tensor(scale_range_max, dtype="float32") - megengine.tensor(scale_range_min, dtype="float32")).reshape(1, 1, 3)

        self.translation_min = megengine.tensor(translation_range_min, dtype="float32").reshape(1, 1, 3)
        self.translation_range = (megengine.tensor(translation_range_max, dtype="float32") - megengine.tensor(translation_range_min, dtype="float32")).reshape(1, 1, 3)

        self.z_level_plane = z_level_plane
        self.rotation_range = rotation_range
        self.check_collison = check_collison
        self.collision_padding = collision_padding
        self.fix_scale_ratio = fix_scale_ratio
        self.object_on_plane = object_on_plane

        if prior_npz_file is not None:
            try:
                prior = np.load(prior_npz_file)['coordinates']
                # We multiply by ~0.23 as this is multiplier of the original clevr
                # world and our world scale
                self.prior = megengine.tensor(prior, dtype="float32") * 0.2378777237835723
            except Exception as e: 
                print("WARNING: Clevr prior location file could not be loaded!")
                print("For rendering, this is fine, but for training, please download the files using the download script.")
                self.prior = None
        else:
            self.prior = None

    def check_for_collison(self, s, t):
        n_boxes = s.shape[1]
        if n_boxes == 1:
            is_free = (F.ones_like(s[:, 0, 0]) > 0)
        elif n_boxes == 2:
            d_t = F.abs(t[:, :1] - t[:, 1:2])
            d_s = F.abs(s[:, :1] + s[:, 1:2]) + self.collision_padding
            is_free = F.sum((d_t >= d_s).astype("float32"), axis=[1,2]) > 0
        elif n_boxes == 3:
            is_free_1 = self.check_for_collison(s[:, [0, 1]], t[:, [0, 1]])
            is_free_2 = self.check_for_collison(s[:, [0, 2]], t[:, [0, 2]])
            is_free_3 = self.check_for_collison(s[:, [1, 2]], t[:, [1, 2]])
            is_free = is_free_1 & is_free_2 & is_free_3
        else:
            print("ERROR: Not implemented")
        return is_free

    def get_translation(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        t = self.translation_min + torch.tensor(val).reshape(1, n_boxes, 3) * self.translation_range
        t = t.repeat(batch_size, 1, 1)
        if self.object_on_plane:
            t[..., -1] = self.z_level_plane
        return t

    def get_rotation(self, batch_size=32, val=[0.]):
        r_range = self.rotation_range
        values = [r_range[0] + v * (r_range[1] - r_range[0]) for v in val]
        r = torch.cat([get_rotation_matrix(
            value=v, batch_size=batch_size).unsqueeze(1) for v in values],
            dim=1)
        r = r.float()
        return r

    def get_scale(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        n_boxes = len(val)
        if self.fix_scale_ratio:
            t = self.scale_min + \
                torch.tensor(val).reshape(
                    1, n_boxes, -1)[..., :1] * self.scale_range
        else:
            t = self.scale_min + \
                torch.tensor(val).reshape(1, n_boxes, 3) * self.scale_range
        t = t.repeat(batch_size, 1, 1)
        return t

    def get_random_offset(self, batch_size):
        n_boxes = self.n_boxes
        # Sample sizes
        if self.fix_scale_ratio:
            s_rand = megengine.random.uniform(size=(batch_size, n_boxes, 1))
        else:
            s_rand = megengine.random.uniform(size=(batch_size, n_boxes, 3))
        s = self.scale_min + s_rand * self.scale_range

        # Sample translations
        if self.prior is not None:
            """
                随机选batch个translation
            """
            idx = np.random.randint(self.prior.shape[0], size=(batch_size))
            t = self.prior[idx]
        else:
            t = self.translation_min + megengine.random.uniform(size=(batch_size, n_boxes, 3)) * self.translation_range
            if self.check_collison:
                is_free = self.check_for_collison(s, t)
                while not F.prod(is_free):
                    t_new = self.translation_min + megengine.random.uniform(size=(batch_size, n_boxes, 3)) * self.translation_range
                    t[is_free == 0] = t_new[is_free == 0]
                    is_free = self.check_for_collison(s, t)
            if self.object_on_plane:
                t[..., -1] = self.z_level_plane

        def r_val(): 
            return self.rotation_range[0] + np.random.rand() * (self.rotation_range[1] - self.rotation_range[0])

        R = [megengine.tensor(Rot.from_euler('z', r_val() * 2 * np.pi).as_matrix()) for i in range(batch_size * self.n_boxes)]
        R = F.stack(R, axis=0).reshape(batch_size, self.n_boxes, -1)
        return s, t, R

    def forward(self, batch_size=32):
        s, t, R = self.get_random_offset(batch_size)
        R = R.reshape(batch_size, self.n_boxes, 3, 3)
        return s, t, R


if __name__ == "__main__":
    # model = BoundingBoxGenerator(n_boxes=2, check_collison=False)
    # s,t,r = model(32)
    R = Rot.from_euler('z', 0.5 * np.pi).as_matrix()
    point = np.array([0,2,0])
    print(np.dot(R, point))