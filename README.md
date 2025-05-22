# Evaluating the Generalisaton Capabilities of Vision-Language-Action (VLA) Models

In this repository, you can find all the used code corresponding to the work of 'Evaluating the Generalisation Capabilities of Vision-Language-Action Models', also known as my thesis. In this document I will guide you through this code by giving some additional explanation on what each of the submodules/subfolders mean in this repository and how the experiments that were run in our study can be reproduced. We will go over each of these subfolders one by one. You will see that each one of them has their own README document explaining what should be done to install the required packages, we will not repeat that here and if you want to run our experiments, we advise you to make a virtual environment for each of the submodules following their instructions. This document will then be able to guide you through the commands that must be run using these installed packages.

## OpenVLA
To start a fine-tuning session, go to the openvla subfolder and use the following command:
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
Of course you still need to change the data_root_dir, run_root_dir, adapter_tmp_dir, wadb_project, and wandb_entity yourself. In order for this to work to your own dataset you most likely also need to modify the configs.py and transforms.py file as suggested in the README of openvla. 
Deploying a model checkpoint (after it has been fine-tuned) is done with the following command:
```bash
python vla-scripts/deploy.py --openvla_path /home/qnoens/Documents/qnoens/OpenVLA/training/log/Task1B_checkpoint
```

In this folder, we also provided a mujoco-sim folder. When running the scripts/demonstration_collection.py file here, the scripted demonstrations will be collected for the push button simulation experiment (which wa spart of Chapter 4 in the thesis). In order to do the intermediate evaluation of this task to get the validation curves during fine-tuning, you should go to the vla-scripts/finetune.py file and put the bottom code outside of the comments.

## RLDS Dataset Builder
This folder enabled us to convert our LeRobot dataset to a RLDS dataset format in order for our dataset to be fine-tuned on OpenVLA. We refer to all the necessary instructions in this submodule itself as it is pretty clear based on that alone how everything works.

## OpenPi
The openpi module includes all the code to fine-tune, and deploy $\pi_0$ and $\pi_0$-FAST. Similarly as for OpenVLA, you will need to change the src/openpi/training/config.py file for your own setup, ours all still included as an example and refer already to our publicly available Hugging Face repository. 
In order to start fine-tuning we must first compute the normalisation statistics:
```bash
CUDA_VISIBLE_DEVICES=0, uv run scripts/compute_norm_stats.py --config-name pi0_fast_ur3_attach_cb_v2_joints
```
Then we start the fine-tuning as follows:
CUDA_VISIBLE_DEVICES=0, XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_ur3_attach_cb_v2_joints --exp-name=pi-zero-fast --overwrite

The instructions for pi-zero are identical, replacing pi-zero-fast by pi-zero should do the trick.

Deploying these models is as simple as running this command, after you changed it to the correct directory and config name of course:
CUDA_VISIBLE_DEVICES=0, uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_ur3_attach_cb_v2_joints --policy.dir=checkpoints/pi0_ur3_attach_cb_v2_joints/pi-zero/9999

## Robot-imitation-glue
This submodule is used for demonstration collection on a physical robot as well as the evaluation of the policies.
