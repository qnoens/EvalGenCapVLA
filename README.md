# Evaluating the Generalisation Capabilities of Vision-Language-Action (VLA) Models

This repository contains the full codebase for the thesis *“Evaluating the Generalisation Capabilities of Vision-Language-Action Models.”* It includes all scripts, tools, and configurations used in the experiments.

Each submodule (subfolder) contains its own README with detailed usage instructions and setup guides, typically using either `conda` or `uv` environments. We recommend setting up a separate environment for each submodule, following their individual guides.

This README provides an overview of what each submodule is for and how to reproduce the experiments described in the thesis.

---

## OpenVLA

This module contains code to fine-tune and deploy OpenVLA models.

### Fine-tuning

To begin fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /path/to/your/data \
  --dataset_name bridge_orig \
  --run_root_dir /path/to/logs \
  --adapter_tmp_dir /path/to/weights \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "your-project-name" \
  --wandb_entity "your-wandb-entity" \
  --save_steps 1000
```

Make sure to update the relevant paths and names (`data_root_dir`, `wandb_project`, etc.). If you're using a custom dataset, update `configs.py` and `transforms.py` as described in the OpenVLA README.

### Deployment

After training:

```bash
python vla-scripts/deploy.py --openvla_path /path/to/your/checkpoint
```

### Simulated Demonstrations

In `mujoco-sim/scripts/demonstration_collection.py`, you can collect demonstrations for the button-push simulation experiment (referenced in Chapter 4 of the thesis). To enable validation during training, uncomment the evaluation code at the bottom of `vla-scripts/finetune.py`.

---

## RLDS Dataset Builder

This submodule converts LeRobot-style datasets into RLDS format, enabling compatibility with OpenVLA. Setup and usage instructions are included in the submodule's own README.

---

## OpenPi

This module contains all code for fine-tuning and deploying the baseline policies \$\pi\_0\$ and \$\pi\_0\$-FAST.

### Preprocessing

First, compute normalization statistics:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_ur3_attach_cb_v2_joints
```

### Fine-tuning

Then, fine-tune the model:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_ur3_attach_cb_v2_joints --exp-name=pi-zero-fast --overwrite
```

To fine-tune \$\pi\_0\$, simply replace `pi-zero-fast` with `pi-zero`.

### Deployment

Deploy with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_ur3_attach_cb_v2_joints \
  --policy.dir=checkpoints/pi0_ur3_attach_cb_v2_joints/pi-zero/9999
```

---

## Robot-Imitation-Glue

Used for physical robot data collection and evaluation.

* Run `collect_data.py` to collect demonstrations.
* For OpenVLA evaluation, modify `openvla_agent.py` to set your dataset name and desired instruction, then run `eval_openvla.py`.
* For OpenPi, first launch the server, then run `eval_pi0.py`, setting the instruction inside the script.

Use `zero_action_remover.ipynb` to filter out no-op actions from OpenVLA episodes.

---

## Help Scripts

This folder includes utilities to support the experiments:

* `lerobot_dataset4pi0.ipynb`: Converts OpenVLA-style delta EEF actions to joint-angle actions for OpenPi.
* `upload_dataset.py`: Uploads a LeRobot dataset to the Hugging Face Hub.

These scripts use the same environment as `robot-imitation-glue`.
