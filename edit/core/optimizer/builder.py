import copy
import inspect
from edit.utils import Registry, build_from_cfg
import megengine.optimizer as mgeoptim
OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')

def register_mge_optimizers():
    mge_optimizers = []
    for module_name in dir(mgeoptim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(mgeoptim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, mgeoptim.optimizer.Optimizer):
            OPTIMIZERS.register_module(module=_optim, force=True)
            mge_optimizers.append(module_name)
    return mge_optimizers

MGE_OPTIMIZERS = register_mge_optimizers()

def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)

def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg)
    )
    optimizer = optim_constructor(model)
    return optimizer


def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned. (use this usually)

    For example,

    Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': mge.optimizer, 'model2': mge.optimizer)``

    Args:
        model (:obj:`Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`mge.optimizer`] The initialized optimizers.
    """
    optimizers = {}
    # determine whether 'cfgs' has several dicts for optimizers
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False
    if is_dict_of_dict:
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers
    else:
        raise RuntimeError("please use 'dict of dict' style for optimizers config")
