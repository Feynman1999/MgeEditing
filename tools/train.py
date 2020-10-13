"""
    train your model and support eval when training(hook).
    normally the workflow is ``workflow = [('train', 1)]`` for training.
    you can also append test workflow, but more formally, use tools/test.py.
"""
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import multiprocessing as mp
import time
import argparse
import megengine as mge
import megengine.distributed as dist
from megengine.jit import trace
from megengine.data import RandomSampler, SequentialSampler, DataLoader

from edit.utils import Config, mkdir_or_exist, build_from_cfg, get_root_logger
from edit.models import build_model
from edit.datasets import build_dataset
from edit.core.runner import IterBasedRunner, EpochBasedRunner
from edit.core.hook import HOOKS
from edit.core.evaluation import EvalIterHook

def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor o(*￣▽￣*)ブ')
    parser.add_argument('config', help='train config file path')
    parser.add_argument("-d", "--dynamic", default=False, action='store_true', help="enable dygraph mode")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus for one machine to use, 0 use cpu, default 1")
    parser.add_argument("--gpuid", type=str, default="9", help="spcefic one gpu")
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args

def get_loader(dataset, cfg, mode='train'):
    assert mode in ('train', 'eval')
    if mode == 'train':
        sampler = RandomSampler(dataset, batch_size=cfg.data.samples_per_gpu, drop_last=True, seed=0)
        loader = DataLoader(dataset, sampler, num_workers=cfg.data.workers_per_gpu)
    else:
        samples_per_gpu = cfg.data.get('eval_samples_per_gpu', cfg.data.samples_per_gpu)
        workers_per_gpu = cfg.data.get('eval_workers_per_gpu', cfg.data.workers_per_gpu)
        sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False)
        loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    return loader

def train(model, datasets, cfg, rank):
    data_loaders = []
    for ds in datasets:
        data_loaders.append(get_loader(ds, cfg, 'train'))

    # build runner for training
    if cfg.get('total_iters', None) is not None:
        runner = IterBasedRunner(model=model, optimizers_cfg=cfg.optimizers, work_dir=cfg.work_dir)
        total_iters_or_epochs = cfg.total_iters
    else:
        runner = EpochBasedRunner(model=model, optimizers_cfg=cfg.optimizers, work_dir=cfg.work_dir)
        assert cfg.get('total_epochs', None) is not None
        total_iters_or_epochs = cfg.total_epochs

    # resume and create optimizers
    if cfg.resume_from is not None:
        # 恢复之前的训练（包括模型参数和优化器）
        runner.resume(cfg.resume_from, cfg.get('resume_optim', False))
    elif cfg.load_from is not None:
        # 假装从头开始训练， rank0 进程加载参数，然后每个进程创建optim，调用optim init时，模型参数会自动同步
        runner.load_checkpoint(cfg.load_from, load_optim=False)
        runner.create_optimizers()
    else:
        # 不加载任何参数，每个进程直接创建optimizers
        runner.create_optimizers()

    # register hooks
    runner.register_training_hooks(lr_config=cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # visual hook
    if cfg.get('visual_config', None) is not None:
        cfg.visual_config['output_dir'] = os.path.join(cfg.work_dir, cfg.visual_config['output_dir'])
        runner.register_hook(build_from_cfg(cfg.visual_config, HOOKS))

    # evaluation hook
    if cfg.get('evaluation', None) is not None:
        dataset = build_dataset(cfg.data.eval)
        save_path = os.path.join(cfg.work_dir, 'eval_visuals')
        log_path = cfg.work_dir
        runner.register_hook(EvalIterHook(get_loader(dataset, cfg, 'eval'), 
                            save_path=save_path, 
                            log_path=log_path, 
                            **cfg.evaluation))

    runner.run(data_loaders, cfg.workflow, total_iters_or_epochs)

def worker(rank, world_size, cfg):
    logger = get_root_logger()  # 每个进程再创建一个logger
    
    # set dynamic graph for debug
    if cfg.dynamic:
        trace.enabled = False

    if world_size > 1:
        # Initialize distributed process group
        logger.info("init distributed process group {} / {}".format(rank, world_size))
        dist.init_process_group(
            master_ip="localhost",
            master_port=23333,
            world_size=world_size,
            rank=rank,
            dev=rank%8,
        )
    model = build_model(cfg.model, train_cfg=cfg.train_cfg, eval_cfg=cfg.eval_cfg)
    datasets = [build_dataset(cfg.data.train)]
    train(model, datasets, cfg, rank)

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = args.gpus
    cfg.dynamic = args.dynamic
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        assert cfg.get('work_dir', None) is not None, 'if do not set work_dir in args, please set in config file'
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.work_dir = os.path.join(cfg.work_dir, timestamp)
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # init the logger
    log_file = os.path.join(cfg.work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('training gpus num: {}'.format(args.gpus))
    logger.info('Config:\n{}'.format(cfg.text))

    # get world_size
    world_size = args.gpus
    assert world_size <= mge.get_device_count("gpu")
    if world_size == 0: # use cpu    
        mge.set_default_device(device='cpux')
    else:
        gpuid = args.gpuid
        mge.set_default_device(device='gpu' + gpuid)

    if world_size > 1:
        # scale learning rate by number of gpus
        is_dict_of_dict = True
        for _, cfg_ in cfg.optimizers.items():
            if not isinstance(cfg_, dict):
                is_dict_of_dict = False
        if is_dict_of_dict:
            for _, cfg_ in cfg.optimizers.items():
                cfg_['lr'] = cfg_['lr'] * world_size
        else:
            raise RuntimeError("please use 'dict of dict' style for optimizers config")
        
        # start distributed training, dispatch sub-processes
        mp.set_start_method("spawn")
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, world_size, cfg))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        worker(0, 1, cfg)

if __name__ == "__main__":
    main()
