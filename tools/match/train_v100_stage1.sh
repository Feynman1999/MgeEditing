# python tools/train.py configs/matching/siamfcpp/sota_24G.py --gpuids 3 -d
nohup python -u tools/train.py configs/matching/siamfcpp/sota_v100_insnorm_attention.py --gpuids 0,1,2,3 -d >> /work_base/MBRVSR/insnorm_attention.log 2>&1 &
