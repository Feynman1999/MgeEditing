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
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus for one machine to use, 0 use cpu, default 1")
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models')
    parser.add_argument("-e", "--ensemble", default=False, action = 'store_true')
    args = parser.parse_args()
    return args

def get_loader(dataset, cfg):
    samples_per_gpu = cfg.data.get('test_samples_per_gpu', cfg.data.samples_per_gpu)
    workers_per_gpu = cfg.data.get('test_workers_per_gpu', cfg.data.workers_per_gpu)
    sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False)
    loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    return loader

def test(model, datasets, cfg, rank):
    data_loaders = []
    for ds in datasets:
        data_loaders.append(get_loader(ds, cfg))

    # build epoch runner for test
    runner = EpochBasedRunner(model=model, optimizers_cfg=cfg.optimizers, work_dir=cfg.work_dir)

    # load from
    if cfg.load_from is not None:
        runner.load_checkpoint(cfg.load_from, load_optim=False)
        runner.create_optimizers()
    else:
        raise RuntimeError("cfg.load_from should not be None for test")

    runner.run(data_loaders, cfg.workflow, 8 if cfg.ensemble else 1)

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
    model = build_model(cfg.model, eval_cfg=cfg.eval_cfg)  # eval cfg can provide some useful info, e.g. the padding multi
    datasets = [build_dataset(cfg.data.test)]
    test(model, datasets, cfg, rank)

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = args.gpus
    cfg.dynamic = args.dynamic
    cfg.ensemble = args.ensemble
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        assert cfg.get('work_dir', None) is not None, 'if do not set work_dir in args, please set in config file'

    cfg.work_dir = os.path.join(cfg.work_dir, timestamp)
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # init the logger
    log_file = os.path.join(cfg.work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('test gpus num: {}'.format(args.gpus))
    logger.info('Config:\n{}'.format(cfg.text))

    # get world_size
    world_size = args.gpus
    assert world_size <= mge.get_device_count("gpu")
    if world_size == 0: # use cpu    
        mge.set_default_device(device='cpux')
    else:
        pass  # mge默认使用GPU

    if world_size > 1:
        # start distributed test, dispatch sub-processes
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
