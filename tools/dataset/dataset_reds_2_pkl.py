import os
from tqdm import tqdm
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
import cv2
import pickle
from edit.utils import scandir,FileClient, imfrombytes

IMG_EXTENSIONS = ('.png', )

def make_ndarray_bin(ndarray, bin_path):
    with open(bin_path, 'wb') as _f:
        pickle.dump(ndarray, _f)

def get_bin_path(path):
    suffix = path.split(".")[-1]
    bin_path = path[:-len(suffix)] + "pkl"
    return bin_path

def read_bin(bin_path):
    with open(bin_path, 'rb') as _f:
        img = pickle.load(_f)
    return img

def solve(keys, folder):
    for key in tqdm(range(len(keys))):
        filepath = os.path.join(folder, keys[key])
        bin_path = get_bin_path(filepath)
        img_bytes = file_client.get(filepath)
        img = imfrombytes(img_bytes, flag="unchanged", channel_order='bgr')  # HWC, BGR
        make_ndarray_bin(img, bin_path)

def test_speed_for_pkl(keys, folder):
    for key in tqdm(range(len(keys))):
        filepath = os.path.join(folder, keys[key])
        bin_path = get_bin_path(filepath)
        # img_bytes = file_client.get(filepath)
        # img = imfrombytes(img_bytes, flag="unchanged", channel_order='bgr')  # HWC, BGR
        # make_ndarray_bin(img, bin_path)
        img = read_bin(bin_path)

if __name__ == '__main__':
    file_client = FileClient("disk")
    
    dataroot = "/work_base/datasets/REDS/train"
    lq_folder = dataroot + "/train_sharp_bicubic/X4"
    gt_folder = dataroot + "/train_sharp"
    keys = list(scandir(lq_folder, suffix=IMG_EXTENSIONS, recursive=True))
    # solve(keys, lq_folder)
    # solve(keys, gt_folder)
    test_speed_for_pkl(keys, gt_folder) # 1300 it /s
    