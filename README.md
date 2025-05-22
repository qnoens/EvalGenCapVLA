# Evaluating the Generalisaton Capabilities of Vision-Language-Action (VLA) Models

In this repository, you can find all the used code corresponding to the work of 'Evaluating the Generalisation Capabilities of Vision-Language-Action Models', also known as my thesis. In this document I will guide you through this code by giving some additional explanation on what each of the submodules/subfolders mean in this repository and how the experiments that were run in our study can be reproduced. We will go over each of these subfolders one by one. You will see that each one of them has their own README document explaining what should be done to install the required packages, we will not repeat that here and if you want to run our experiments, we advise you to make a virtual environment for each of the submodules following their instructions. This document will then be able to guide you through the commands that must be run using these installed packages.

## OpenVLA
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /fast_storage/qnoens/OpenVLA/data \
  --dataset_name bridge_orig \
  --run_root_dir /fast_storage/qnoens/training/log \
  --adapter_tmp_dir /fast_storage/qnoens/training/weights \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4\
  --image_aug True \
  --wandb_project "OpenVLA-Finetuning" \
  --wandb_entity "quinten-noens-universiteit-gent" \
  --save_steps 1000
```
