# python tools/train.py configs/restorers/RSDN/rsdn_v4.py --gpuid 0 -d
nohup python -u tools/train.py configs/restorers/RSDN/rsdn_v4.py --gpuid 0 -d >> /opt/data/private/rsdnv4_2021.log 2>&1 &