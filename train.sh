#!/bin/bash
nvidia-smi
hostname
python run.py --precheckpoint './pure_RS5M_Pretrain.pth' --task 'itr_rsicd_vit' --dist "gpu0" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/rsicd/'
# 在 PowerShell 中直接运行
# 输入：python run.py --precheckpoint './pure_RS5M_Pretrain.pth' --task 'itr_rsicd_vit' --dist "gpu0" --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/rsicd/'
