# python tools/train.py configs/matching/siamfcpp/sota_precise_24G.py --gpuids 0,1,6,7,9 -d
nohup python -u tools/train.py configs/matching/siamfcpp/sota_precise_v100_insnorm_deform.py --gpuids 0,1,2,3 -d >> /work_base/MBRVSR/insnorm_deform_stage2_quan10.log 2>&1 &
