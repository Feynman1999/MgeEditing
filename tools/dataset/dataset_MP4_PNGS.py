import cv2
import os
from tqdm import tqdm

video_dir_path = "/data/home/songtt/chenyuxiang/datasets/mgtv/train/train_3"
desti_dir_path = "/data/home/songtt/chenyuxiang/datasets/mgtv/train/train_all_pngs"

if __name__ == "__main__":
    for mp4file in tqdm(os.listdir(video_dir_path)):
        full_path = os.path.join(video_dir_path, mp4file)
        file_name = mp4file.split('.')[0] # 作为文件名
        cap = cv2.VideoCapture(full_path)
        num = 0
        desti = os.path.join(desti_dir_path, file_name)
        os.mkdir(desti)
        while(True):
            # get a frame
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(desti, f"{str(num).zfill(6)}.png"), frame) 
            num += 1

