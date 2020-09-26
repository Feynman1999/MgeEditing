import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy
import cv2
from edit.utils import scandir, is_list_of, mkdir_or_exist, is_tuple_of, imread, imwrite

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')

do_list = ['91', '92', '93', '97', '98', '99']

path_ans = './workdirs/mucan_v5/test'

path_ref = './workdirs/mucan_v5/test1'

xiang = './workdirs/mucan_v5/test2'


for DIR in do_list:
    print("now deal {}".format(DIR))
    now_dir_ans = os.path.join(path_ans, DIR)
    images_ans = sorted(list(scandir(now_dir_ans, suffix=IMG_EXTENSIONS, recursive=True)))
    images_ans = [os.path.join(now_dir_ans, v) for v in images_ans]
    print(images_ans[:10])

    now_dir_ref = os.path.join(path_ref, DIR)
    images_ref = sorted(list(scandir(now_dir_ref, suffix=IMG_EXTENSIONS, recursive=True)))
    images_ref = [os.path.join(now_dir_ref, v) for v in images_ref]
    print(images_ref[:10])

    assert len(images_ans) == len(images_ref)

    for i in range(len(images_ans)):
        print(i)
        # read image from images_ans and ref  , 用ref的上下四个像素填补ans，并原路写回
        ans = imread(images_ans[i], flag='unchanged')
        ref = imread(images_ref[i], flag='unchanged')

        # [H,W,C]
        ans[0:3, :, :] = ref[0:3, : ,:]
        ans[-3:, :, :] = ref[-3:, :, :]

        # write back
        imwrite(ans, images_ans[i])


