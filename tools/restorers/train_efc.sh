# python tools/train.py configs/restorers/EFC/efc_v1.py --gpuids 0,3 -d
nohup python -u tools/train.py configs/restorers/EFC/efc_v1.py --gpuids 1,2,3 -d >> /work_base/efc.log 2>&1 &