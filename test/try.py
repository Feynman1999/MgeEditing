import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from edit.utils import images2video


dirpath = "./workdirs/mucan_x4_mge_epoch_9_128/20200824_135619/eval_visuals/iter_570000"
images2video(dirpath)
