# python tools/train.py configs/matching/meshflow/transformer_meshflow.py --gpuids 0,1,2 -d
nohup python -u tools/train.py configs/matching/meshflow/transformer_meshflow.py --gpuids 0,1,2 -d >> /work_base/meshflow.log 2>&1 &