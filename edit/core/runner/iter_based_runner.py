import os.path as osp
import time
import megengine as mge
from megengine.module import Module
from .base_runner import BaseRunner, module_ckpt_suffix, optim_ckpt_suffix
from edit.utils import is_list_of


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)  # the epoch in distributed_sampler will +=1 automaticly
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader):
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_iter')
        data_batch = next(data_loader)
        self.losses = self.model.train_step(data_batch)
        # 搞明白这个loss是所有线程的还是单个的
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

    def test(self, data_loader):
        self.mode = 'test'
        self.data_loader = data_loader
        self.call_hook('before_test_iter')
        data_batch = next(data_loader)
        self.outputs = self.model.test_step(data_batch)
        self.call_hook('after_test_iter')
        self._inner_iter += 1

    def run(self, data_loaders, workflow, max_iters):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training and test.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('test', 1)] means running 10000 iterations for training and
                10 iterations for test, iteratively.
            max_iters (int): Total training iterations.
        """
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_iters = max_iters
        self.logger.info("Start running, work_dir: {}, workflow: {}, max iters for train: {}".format(self.work_dir, workflow, max_iters))
        self.logger.info("registered hooks: " + str(self.hooks))
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        self.model.max_iters = max_iters
        self.model.now_iter = self.iter
        while self.iter < max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError('runner has no method named "{}" to run a workflow'.format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= max_iters:
                        return
                    iter_runner(iter_loaders[i])

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    def resume(self, checkpoint, resume_optimizer=True):
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
        """
        assert 'iter_' in checkpoint
        res_dict = self.load_checkpoint(checkpoint, load_optim=resume_optimizer)
        assert res_dict['epoch_or_iter'] == 'iter'
        self._epoch = 0
        self._iter = res_dict['nums']
        self._inner_iter = 0
        self.logger.info("resumed from iter: {}, epoch set to 0".format(self._iter))
        # create optimizers
        self.create_optimizers() # 模型参数会自动同步一次
        # 加载optim的state
        if resume_optimizer:
            for submodule_name in self.optimizers_cfg.keys():
                self.model.optimizers[submodule_name].load_state_dict(res_dict[submodule_name])

    def save_checkpoint(self, out_dir, create_symlink=True):
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        filename_tmpl = 'iter_{}'
        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        self.logger.info('save checkpoint to {}'.format(filepath))
        if isinstance(self.model.optimizers, dict):
            for key in self.model.optimizers.keys():
                submodule = getattr(self.model, key, None)
                assert submodule is not None, "model should have submodule {}".format(key)
                assert isinstance(submodule, Module), "submodule should be instance of megengine.module.Module"
                mge.save(submodule.state_dict(), osp.join(filepath, key + module_ckpt_suffix))
                mge.save(self.model.optimizers[key].state_dict(), osp.join(filepath, key + optim_ckpt_suffix))
        else:
            raise TypeError(" the type of optimizers should be dict for save_checkpoint")

        if create_symlink:
            pass

    def register_training_hooks(self,
                                lr_config,
                                checkpoint_config,
                                log_config):
        """Register default hooks for iter-based training.

        Default hooks include:

        - LrUpdaterHook
        - CheckpointSaverHook
        - logHook
        """
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)
        if log_config is not None:
            log_config.setdefault('by_epoch', False)

        # self.register_lr_hook(lr_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_logger_hooks(log_config)

        # self.register_hook(IterTimerHook())


