import os
import time
from megengine.distributed.group import get_rank, get_world_size, is_distributed
from megengine.data.dataloader import DataLoader
from edit.core.hook import Hook
from edit.utils import to_list, is_list_of, get_logger, mkdir_or_exist

def gpu_gather(v):
    raise NotImplementedError("gather for gpu tensor is not implement now")

def cpu_gather(v):
    raise NotImplementedError("gather for cpu list is not implement now")

class EvalIterHook(Hook):
    """evaluation hook by iteration-based.

    This hook will regularly perform evaluation in a given interval

    Args:
        dataloader (DataLoader): A mge dataloader.
        interval (int): Evaluation interval. Default: 3000.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a mge DataLoader, but got {}'.format(type(dataloader)))
        self.dataloader = dataloader
        self.eval_kwargs = eval_kwargs
        self.interval = self.eval_kwargs.pop('interval', 10000)
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)
        self.log_path = self.eval_kwargs.pop('log_path', None)
        mkdir_or_exist(self.save_path)
        self.logger = get_logger(name = "EvalIterHook", log_file=self.log_path) # only for rank0
        
        if is_distributed():
            self.local_rank = get_rank()
            self.nranks = get_world_size()
        else:
            self.local_rank = 0
            self.nranks = 1

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        self.logger.info("start to eval for iter: {}".format(runner.iter+1))
        save_path = os.path.join(self.save_path, "iter_{}".format(runner.iter+1))
        mkdir_or_exist(save_path)
        results = []  # list of dict
        print(len(self.dataloader))
        for _, data in enumerate(self.dataloader):
            batchdata = data
            outputs = runner.model.test_step(batchdata, save_image=self.save_image, save_path=save_path)
            if self.nranks > 1:
                # TODO:
                # 一定是使用GPU，将所有线程的outputs和data收集过来
                # gathered_outputs = xxx
                # gathered_batchdata = xxx
                pass
            else:
                gathered_outputs = outputs  # list of tensor
                gathered_batchdata = batchdata  # list of numpy
            assert gathered_batchdata[0].shape[0] == gathered_outputs[0].shape[0]  # batch维度要匹配
            assert gathered_batchdata[0].shape[0] == sample_nums_for_one_thread * self.nranks  # 确保是gather后的
            sample_nums_all_threads += gathered_outputs[0].shape[0]
            # 目前是所有进程前向并保存结果，0号进程去计算metric；之后增加CPU进程通信，把计算metric也分到不同进程上
            if self.local_rank == 0:
                result = runner.model.cal_for_eval(gathered_outputs, gathered_batchdata)
                assert is_list_of(result, dict)
                # self.logger.info(result)
                results += result
            else:
                pass
        if self.local_rank == 0:
            self.evaluate(results, runner.iter+1)

        # for key, para in runner.model.generator.named_parameters():
        #     para.requires_grad = True

    def evaluate(self, results, iters):
        """Evaluation function.

        Args:
            runner (``BaseRunner``): The runner.
            results (list of dict): Model forward results.
            iter: now iter.
        """
        save_path = os.path.join(self.save_path, "iter_{}".format(iters))  # save for some information. e.g. SVG for everyframe value in VSR.
        eval_res = self.dataloader.dataset.evaluate(results, save_path)
        self.logger.info("*****   eval results for {} iters:   *****".format(iters))
        for name, val in eval_res.items():
            self.logger.info("metric: {}  average_val: {:.4f}".format(name, val))
