# python tools/train.py configs/mot/centertracker_fish.py --gpuids 1 -d
nohup python -u tools/train.py configs/mot/centertracker_fish_dla_GN_v4.py --gpuids 4,5,6,7,8,9 -d >> ./mot_dla_GN_v4.log 2>&1 &