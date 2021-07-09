"""
given imgs and txt(include imgid, bboxes and id ), to generate video to disk
"""
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
import cv2
import configparser
import random
from edit.utils import scandir
from tqdm import tqdm

def get_height_and_width(path):
    conf = configparser.ConfigParser()
    conf.read(path)
    h = conf.get("Sequence", "imHeight")
    w = conf.get("Sequence", "imWidth")
    return int(h), int(w)

def num_to_color(num):
    z = num % 256
    num = num // 256
    y = num % 256
    num = num //256
    return (num, y, z)

def get_id_to_color_dict(nums = 40):
    # 随机生成nums种颜色, (x,y,z)  [0,255]
    assert nums <= 100
    res = {}
    random.seed(23333)
    res2 = random.sample(range(0, 256**3), nums)
    for id, item in enumerate(res2):
        res[id+1] = num_to_color(item)
    return res

def visualize(workdir, aim_dir, imgname_shift = 0, zero_pad_length = 6, suffix = ".PNG"):
    imgs_dir = os.path.join(workdir, "img1")
    label_txt = os.path.join(workdir, "gt", "gt.txt")

    ini_path = os.path.join(workdir, "seqinfo.ini")
    
    clip_name = os.path.basename(workdir)
    assert clip_name != ""

    h, w = get_height_and_width(ini_path)
    out = cv2.VideoWriter(os.path.join(aim_dir, f"{clip_name}.avi"), cv2.VideoWriter_fourcc(*'XVID'), 15.0, (w, h))
    color_dict = get_id_to_color_dict()

    data_dict = {} # key : [imgid, trackid]

    # 对于label_txt中的每一行
    with open(label_txt, "r") as f:
        print("reading txt......")
        for line in tqdm(f.readlines()):
            if line.strip() == "":
                continue
            data = line.split(",")
            assert len(data) == 9
            data = [ int(float(data[i])+0.5) for i in range(6)]
            data_dict[(data[0], data[1])] = data[2:] # (imgid, trackid)
    
    keys = sorted(data_dict.keys())
    # trick
    keys.append((-1, -1))
    print("writing to video...") 
    imgname = os.path.join(imgs_dir, f"{str(imgname_shift + keys[0][0]).zfill(zero_pad_length)}{suffix}")
    now_img = cv2.imread(imgname)
    for i in tqdm(range(len(keys) - 1)):
        color = color_dict[keys[i][1]]
        x,y,w,h = data_dict[keys[i]]
        now_img = cv2.rectangle(now_img, (x, y), (x+w-1, y+h-1), color, 2)
        
        if keys[i][0] != keys[i+1][0]: # last track id in img, write img
            out.write(now_img)
            now_img = cv2.imread(os.path.join(imgs_dir, f"{str(imgname_shift + keys[i+1][0]).zfill(zero_pad_length)}{suffix}"))
            # last time is None
    out.release()

if __name__ == "__main__":
    clip_names = ['p4']
    workdir = "/data/home/songtt/chenyuxiang/datasets/MOTFISH/preliminary/train"
    aim_dir = "."
    for item in clip_names:
        print("now dealing : {}".format(item))
        visualize(os.path.join(workdir, item), aim_dir)

    