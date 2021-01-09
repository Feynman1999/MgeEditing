import os
import sys
import multiprocessing as mp
import time
import argparse
import megengine as mge
import megengine.distributed as dist
from megengine.jit import trace
from megengine.data import RandomSampler, SequentialSampler, DataLoader

def worker(rank, world_size, port=23333):
    if world_size > 1:
        # Initialize distributed process group
        dist.init_process_group(
            master_ip = "localhost",
            port = port,
            world_size = world_size,
            rank = rank,
            device = rank+1,
        )
    x = mge.tensor([1.])

def main():
    world_size = 3
    bug: 使用多进程时不能调用这句话 # assert world_size <= mge.get_device_count("gpu")
    
    if world_size == 0: # use cpu    
        mge.set_default_device(device='cpux')
    elif world_size == 1:
        mge.set_default_device(device='gpu' + gpu_list[0])
    else:
        pass

    if world_size > 1:
        port = dist.util.get_free_ports(1)[0]
        server = dist.Server(port)
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, world_size, port))
            p.start()
            processes.append(p)

        for rank in range(world_size):
            processes[rank].join()
            code = processes[rank].exitcode
            assert code == 0, "subprocess {} exit with code {}".format(rank, code)
    else:
        worker(0, 1, cfg)

if __name__ == "__main__":
    main()
