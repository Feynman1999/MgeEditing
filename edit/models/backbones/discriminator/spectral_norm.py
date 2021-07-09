"""
    Spectral Normalization from https://arxiv.org/abs/1802.05957
    ref to https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
    this is a megengine version provided by feynman1999
"""
import megengine as mge
from megengine.module import ConvTranspose2d
import megengine.functional as F
import numpy as np

class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced
    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer, (in megengine, W is a buffer too)
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front [0,1,2,3] -> [1, 0, 2, 3]
            weight_mat = weight_mat.transpose(self.dim, *[d for d in range(weight_mat.ndim) if d != self.dim])
        height = weight_mat.shape[0]
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            # detach the weight_mat from graph
            new_weight_mat = weight_mat.detach()
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = F.normalize(F.matmul(new_weight_mat.transpose(1,0), u), ord=2, axis=0, eps=self.eps)
                u = F.normalize(F.matmul(new_weight_mat, v), ord=2, axis=0, eps=self.eps)
            # in pytorch,due to compatible with DataParallel, v,u use inplace for normalize,so 
            # for loss = D(real) - D(fake), two time forward, the second forward's v and u Covering the first ones
            # pytorch will complain that variables needed to do backward for the first forward
            #    (i.e., the `u` and `v` vectors) are changed in the second forward.
            # in megengine, we do not need this.
            """                
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
            """
            # write back to module
            setattr(module, self.name + "_u", u)
            setattr(module, self.name + "_v", v)
            
        sigma = (u * F.matmul(weight_mat, v)).sum()
        # sigma = F.dot(u, F.matmul(weight_mat, v)) 

        # debug, check if every process have save weight and sigma
        # print(weight.mean(), sigma)

        weight = weight / sigma
        return weight

    def remove(self, module) -> None:
        raise NotImplementedError("not implement for spectral remove")

    def __call__(self, module, inputs) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        raise NotImplementedError("not implement for _solve_v_and_rescale")

    @staticmethod
    def apply(module, name: str, n_power_iterations: int, dim: int, eps: float):
        for k, hook in module._forward_pre_hooks.items(): # same to pytorch
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)

        has_flag = False
        for key, param in module.named_parameters(recursive=False):
            if key == name:
                weight = param
                has_flag = True
                break

        assert has_flag, "you should make sure the module object [] has the Parameter: []".format(module.__class__, name)

        weight_mat = fn.reshape_weight_to_matrix(weight)

        h, w = weight_mat.shape
        # randomly initialize `u` and `v` with seed
        # 若多进程训练，需保证每个进程的u,v生成的一样，因此暂时采用固定种子的方式
        np.random.seed(23333)
        u = mge.tensor(np.random.normal(0, 1, (h, )), dtype="float32")
        v = mge.tensor(np.random.normal(0, 1, (w, )), dtype="float32")
        u = F.normalize(u, ord=2, axis=0, eps = fn.eps)
        v = F.normalize(v, ord=2, axis=0, eps = fn.eps)

        delattr(module, fn.name)
        setattr(module, fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, mge.tensor(weight))
        setattr(module, fn.name + "_u", u)
        setattr(module, fn.name + "_v", v)
        module.register_forward_pre_hook(fn)
        # module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        # module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, ConvTranspose2d):
            raise NotImplementedError("do not support convtranspose now")
            # dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

def use_spectral_norm(module, use_sn=False):
    if use_sn:
        return spectral_norm(module)
    return module