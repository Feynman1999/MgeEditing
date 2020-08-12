from collections import OrderedDict
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class TextLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and saved in json file.

    Args:
        by_epoch (bool): Whether EpochBasedRunner is used.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
    """

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 by_epoch=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, by_epoch)
        self.by_epoch = by_epoch

    def log(self, runner):
        log_dict = OrderedDict()
        mode = runner.mode
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch
        log_dict['losses'] = runner.losses
        if self.by_epoch:
            log_dict['iter'] = runner.inner_iter
        else:
            log_dict['iter'] = runner.iter
            
        # TODO: learning rate 信息
        # TODO: 内存使用信息(cpu, gpu)

        log_items = []
        for name, val in log_dict.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            elif isinstance(val, int):
                val = str(val)
            else:
                pass
            log_items.append(f'{name}: {val}')

        log_str = ', '.join(log_items)
        runner.logger.info(log_str)
