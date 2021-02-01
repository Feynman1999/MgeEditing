import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import megengine as mge
import megengine.functional as F
import cv2
import numpy as np
from edit.utils import imwrite, tensor2img  

if __name__ == "__main__":
    img_path = "/data/home/songtt/chenyuxiang/datasets/REDS/train/train_sharp/000/00000000.png"
    image = cv2.imread(img_path)
    imwrite(image, file_path="./before.png")
    image = mge.tensor(image/255.0)
    image = F.transpose(image, (2, 0, 1))
    image = F.expand_dims(image, axis = 0)
    
    M_shape = (1, 3, 3)
    M = mge.tensor(np.array([[1, 0.05, 0.],
                            [0, 1, 0.],
                            [0., 0., 1.]], dtype=np.float32).reshape(M_shape))
    
    out = F.warp_perspective(image, M, (720, 1280), border_mode='CONSTANT', border_val=0)
    out = tensor2img(out)
    imwrite(out, file_path="./after.png")