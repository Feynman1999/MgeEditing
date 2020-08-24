"""
    use pyav to deal with .mp4 file, to generate  xxx/HR/000/frame000.png.
"""
import os.path as osp
import ntpath
import os
import av
import cv2
import sys
import random
import tarfile
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from edit.utils import imwrite, scandir, imread, imrescale
from alive_progress import alive_bar
import argparse

# TRAIN_RAW_DATA = "./dataset/dataset-1325/train.tar"
# TEST_RAW_DATA = "./dataset/dataset-1325/test.tar"
# TRAIN_DATA_STORAGE = "./train_patches"

# 统计视频帧数
def frame_count(container, video_stream=0):
    def count(generator):
        res = 0
        for _ in generator:
            res += 1
        return res

    frames = container.streams.video[video_stream].frames
    if frames != 0:
        return frames
    frame_series = container.decode(video=video_stream)
    frames = count(frame_series)
    container.seek(0)
    return frames

# tar = tarfile.open(TRAIN_RAW_DATA)
# os.makedirs(TRAIN_DATA_STORAGE, exist_ok=True)

# name_info = {}
# todo_list = []
# while True:
#     tinfo = tar.next()
#     if tinfo is None:
#         break
#     if not tinfo.isfile():
#         continue
#     tname = tinfo.name
#     name_info[tname] = tinfo
#     if tname.endswith("_down4x.mp4"):
#         todo_list.append(tname)

# count = 0
# for tname in tqdm(todo_list):
#     tinfo = name_info[tname]
#     srcinfo = name_info[tname.replace('_down4x.mp4', '')]

#     f_down4x = tar.extractfile(tinfo)    # 下采样版本的视频
#     f_origin = tar.extractfile(srcinfo)  # 原始视频

#     container_down4x = av.open(f_down4x)
#     container_origin = av.open(f_origin)

#     frames_down4x = container_down4x.decode(video=0)
#     frames_origin = container_origin.decode(video=0)

#     fc_down4x = frame_count(container_down4x)
#     fc_origin = frame_count(container_origin)
#     extra = fc_down4x - fc_origin
    
#     # 由于视频编码和 FFmpeg 实现的问题，压缩前后的帧数可能会不等，下采样版本的视频可能数量少几帧。
#     # 这时，您需要注意跳过下采样版本视频缺少的帧数。
#     if extra > 0:
#         for _ in range(extra):
#             next(frames_down4x)

#     for k, (frame_down4x,
#             frame_origin) in enumerate(zip(frames_down4x, frames_origin)):
#             img_origin = frame_origin.to_rgb().to_ndarray()
#             if img_origin.shape[0] < 256 or img_origin.shape[1] < 256:
#                 continue
                
#             img_down4x = frame_down4x.to_rgb().to_ndarray()
#             img_down4x = cv2.resize(
#                 img_down4x, (img_origin.shape[1], img_origin.shape[0]))

#             x0 = random.randrange(img_origin.shape[0] - 256 + 1)
#             y0 = random.randrange(img_origin.shape[1] - 256 + 1)

#             img_show = np.float32(
#                 np.stack((img_down4x[x0:x0 + 256, y0:y0 + 256].transpose((2, 0, 1)),
#                           img_origin[x0:x0 + 256, y0:y0 + 256].transpose((2, 0, 1))))) / 256
#             np.save(os.path.join(TRAIN_DATA_STORAGE, '%04d.npy' % count), img_show)
#             count += 1

#     container_down4x.close()
#     container_origin.close()
#     f_down4x.close()
#     f_origin.close()

def mp42png(HRpath, destidir, write_flag = True):
    LRpath = HRpath.replace("HR", "LR") + '_down4x.mp4'
    video_name = os.path.splitext(ntpath.basename(HRpath))[0]
    
    HRdir = os.path.join(destidir, "HR", video_name)
    LRdir = os.path.join(destidir, "LR", video_name)

    container_down4x = av.open(LRpath)
    container_origin = av.open(HRpath)
    frames_down4x = container_down4x.decode(video=0)
    frames_origin = container_origin.decode(video=0)

    fc_down4x = frame_count(container_down4x)
    fc_origin = frame_count(container_origin)
    extra = fc_down4x - fc_origin

    if extra>0:
        print("video: {} 's LR frames largers than HR frames nums: {}".format(video_name, extra))
    
    # 由于视频编码和 FFmpeg 实现的问题，压缩前后的帧数可能会不等，下采样版本的视频可能数量少几帧。
    # 这时，您需要注意跳过下采样版本视频缺少的帧数。
    if extra > 0:
        for _ in range(extra):
            next(frames_down4x)

    count = 0

    for _, (frame_down4x, frame_origin) in enumerate(zip(frames_down4x, frames_origin)):
        img_origin = frame_origin.to_rgb().to_ndarray() # (1920, 1080, 3)
        if img_origin.shape[0] < 256 or img_origin.shape[1] < 256:
            continue
            
        img_down4x = frame_down4x.to_rgb().to_ndarray()
        assert img_down4x.shape[2] == img_origin.shape[2] and img_origin.shape[2] == 3
        if img_down4x.shape[0] * 4 != img_origin.shape[0] or img_down4x.shape[1] * 4 != img_origin.shape[1]:
            print("video: {} 's frame {} shape do not right, do resize, down shape:{}, origin shape: {}".format(video_name, count, img_down4x.shape, img_origin.shape))
            print("resize origin shape to 1920 1080 and down shape to 480 270")
            img_down4x = cv2.resize(img_down4x, (1920 // 4, 1080 // 4))
            img_origin = cv2.resize(img_origin, (1920, 1080))

        if write_flag and not os.path.exists(os.path.join(HRdir, '%05d.png' % count)):
            imwrite(img_origin, file_path=os.path.join(HRdir, '%05d.png' % count))
            imwrite(img_down4x, file_path=os.path.join(LRdir, '%05d.png' % count))

        count += 1

    container_down4x.close()
    container_origin.close()

def mp4s2pngs(mp4dir, IMG_EXTENSION, destidir):
    videos = list(scandir(mp4dir, suffix=IMG_EXTENSION, recursive=True))
    videos = [osp.join(mp4dir, v) for v in videos]
    with alive_bar(len(videos)) as bar:  # declare your expected total
        for HRpath in videos:
            mp42png(HRpath, destidir)
            bar()                        # call after consuming one item

def parse_args():
    parser = argparse.ArgumentParser(description='trans mp4 to png')
    parser.add_argument("--mp4dir", type=str, default='/opt/data/private/datasets/mge/train/HR', help="path to HR mp4 files, make sure replace HR to LR is LR path")
    parser.add_argument("--destidir", type=str, default='/opt/data/private/datasets/mge/pngs')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    mp4dir = args.mp4dir
    IMG_EXTENSION = (".mp4", ".mkv")
    mp4s2pngs(mp4dir, IMG_EXTENSION, args.destidir)

if __name__ == "__main__":
    main()