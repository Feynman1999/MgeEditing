name="sota_v100_disweight_lr_0.2_ch_160"
# python tools/train.py configs/matching/siamfcpp/sota_24G.py --gpuids 3 -d
python tools/train.py configs/matching/siamfcpp/${name}.py --gpuids 3 -d >> /work_base/MBRVSR/${name}.log 2>&1 &
