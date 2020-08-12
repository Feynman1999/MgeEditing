import os.path as osp
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
sys.path.insert(0, "/home/aistudio/external-libraries")

from edit.utils import imwrite, scandir, imread, imrescale
from alive_progress import alive_bar
import argparse


def DIV2k_HR2LR(HRpath, LRsuffix, scale, IMG_EXTENSION):
    """
    :param HRpath: the HRpath to HR images
    :return: the LRpath to LR images
    """
    images = list(scandir(HRpath, suffix=IMG_EXTENSION, recursive=True))
    images = [osp.join(HRpath, v) for v in images]
    assert images, f'{HRpath} has no valid image file.'
    with alive_bar(len(images)) as bar:   # declare your expected total
        for image in images:               # iterate as usual
            HRimg = imread(image, flag='color')
            # notice: we use opencv area interpolation method by default, you can change your own method. e.g. pillow bicubic
            LRimg = imrescale(HRimg, 1.0/scale)
            dirpath = osp.dirname(image)
            dirname = osp.basename(dirpath)
            if "HR" in dirname:
                newdirname = dirname.replace("HR", "LR")
            else:
                newdirname = dirname +"_LRx" + str(scale)
            dirdirpath = osp.dirname(dirpath)
            newdirpath = osp.join(dirdirpath, newdirname)

            HR_image_name = osp.splitext(osp.basename(image))[0]
            LR_image_name = HR_image_name + LRsuffix + IMG_EXTENSION
            imwrite(LRimg, osp.join(newdirpath, LR_image_name))
            bar()                        # call after consuming one item

    

def parse_args():
    parser = argparse.ArgumentParser(description='DIV2k_HR2LR')
    parser.add_argument("--scale", type=int ,default = 4, help="the scale factor")
    parser.add_argument("--HRpath", type=str, default='/opt/data/private/datasets/Set5/HR', help="path to HR images")
    parser.add_argument("--LRsuffix", type=str, default='x4', help="suffix that will add to LR images")
    parser.add_argument("--IMG_EXTENSION", type=str, default='.png', help="img extension for HR images")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    HRpath = args.HRpath
    LRsuffix = args.LRsuffix
    scale = args.scale
    IMG_EXTENSION = args.IMG_EXTENSION
    DIV2k_HR2LR(HRpath, LRsuffix, scale, IMG_EXTENSION)


if __name__ == "__main__":
    main()