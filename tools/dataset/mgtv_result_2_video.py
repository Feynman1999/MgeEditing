import os
import cv2
from matplotlib.pyplot import clim
from tqdm import tqdm

test_dataset_path = "/data/home/songtt/chenyuxiang/datasets/mgtv/test/test_a"
result_path = "/data/home/songtt/chenyuxiang/MBRVSR/workdirs/sttn_mgtv_without_mask_test/20210609_134205/test_results"
desti_dir = "."

def read_bbox(dir_name):
    bboxes = []
    with open(os.path.join(test_dataset_path, dir_name, "minbbox.txt"), "r") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            x,y,w,h = [int(item) for item in line.strip().split(" ")]
            bboxes.append((x,y,w,h))
    return bboxes

if __name__ == "__main__":
    for clip in tqdm(os.listdir(test_dataset_path)):
        # 读取result_path中的结果，根据bbox放到原本的图像中，并写入视频
        _,clip_id = clip.split("_")
        clip_id = int(clip_id) 
        out = cv2.VideoWriter(os.path.join(desti_dir, f"{clip}.avi"), cv2.VideoWriter_fourcc(*'XVID'), 25.0, (1024, 576))
        bboxes = read_bbox(clip)
        for id, img in enumerate(sorted(os.listdir(os.path.join(test_dataset_path, clip, 'frames_corr')))):
            crop_img = "crop_" + img
            crop_img = cv2.imread(os.path.join(result_path, f"video_{str(clip_id).zfill(4)}", crop_img))
            origin_img = cv2.imread(os.path.join(test_dataset_path, clip, 'frames_corr', img))
            x,y,w,h = bboxes[id]
            # origin_img[y:y+h, x:x+w, :] = crop_img
            out.write(origin_img)
        out.release()