# python tools/train.py configs/restorers/BasicVSR/basicVSR_stage1.py --gpuids 0,1,2,3 -d
nohup python -u tools/train.py configs/restorers/BasicVSR/basicVSR_stage2.py --gpuids 0,1,2,3 -d >> /work_base/track1_stage2.log 2>&1 &