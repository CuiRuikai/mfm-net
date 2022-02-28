# MFM-Net


## Experiment record

---
Dev
'''
bash ./scripts/train.sh 0 --config ./cfgs/EPN3D_models/dev.yaml --exp_name dev
bash ./scripts/test.sh 0 --ckpts ./experiments/PM/EPN3D_models/pm_only/ckpt-best.pth --config ./experiments/PM/EPN3D_models/pm_only//config.yaml --exp_name dev_test --save_pred
bash ./scripts/test.sh 0 --ckpts ./experiments/PM/EPN3D_models/pm_only_watercraft/ckpt-best.pth --config ./experiments/PM/EPN3D_models/pm_only_watercraft/config.yaml --exp_name dev_watercraft_test --save_pred

---
Partial Matching
nohup bash ./scripts/train.sh 0 --config ./cfgs/EPN3D_models/MFM.yaml --exp_name baseline &

'''