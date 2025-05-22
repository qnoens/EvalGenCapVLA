# TODO: visualise the data that is given as input to the finetuning code
# Idea: start from the finetuning code and modify the training part to just reading out labels, pixel input, etc.

# Also take a closer look and really analyse the output of "python visualize_dataset.py openvla_finetune_dataset" in the /fast_storage/qnoens/OpenVLA/rlds_dataset_builder directory with the rlds_env activated.
# This will give you a good idea already if the RLDS data seems to be correct! => DONE: they all seem correct

# Another sudden hypothesis I have: maybe the model does not learn becuase it's not getting the correct input when screen is locked. I don't think this is the issue because we only get no input when we are evaluating so the evaluation is just not representative. (This is an alternative issue if it's not the data that is wrong, other issue might seem to be that environment cannot be created for the first time when screen is locked)

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Added from mujoco
from dm_control import viewer
from dm_control.composer import Environment
import matplotlib.pyplot as plt
from mujoco_sim.environments.tasks.robot_push_button import RobotPushButtonTask
from PIL import Image
import cv2
import json
import numpy as np
import dm_env

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"



@dataclass
class DataCheckerConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    #adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    #max_steps: int = 200_000                                        # Max number of fine-tuning steps
    #save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    #save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    # fmt: on


@draccus.wrap()
def check_data(cfg: DataCheckerConfig) -> None:

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        #vla = vla.to(device_id)
        print("skip")

    # Assigning un-normalization to the model. Note that this might not work the first time you run this script since dataset_statistics.json might not exist yet => This can be fixed by not evaluating the model in the first run
    dataset_statistics_path = os.path.join(run_dir, "dataset_statistics.json")
    print(dataset_statistics_path, os.path.isfile(dataset_statistics_path))
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    #vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes), # if DDP, then use vla.module
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Check data
    for batch_idx, batch in enumerate(dataloader):
        # print(batch["input_ids"].shape)
        #print(batch["pixel_values"].shape)
        #print(batch["labels"].shape)
        print(batch)
        # print(batch["attention_mask"].shape)

        # Preprocess image
        image = batch["pixel_values"][0][3:].numpy() 
        image = np.moveaxis(image, 0, -1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Scale image otherwise too dark
        image = scale_image(image)        

        # Display action
        # print(batch["attention_mask"][0].numpy())
        # print(batch["input_ids"][0].numpy())
        #print(batch["labels"][0].numpy())

        action_tokens = batch["labels"][0].numpy()
        mask = action_tokens > action_tokenizer.action_token_begin_idx
        filtered_action_tokens = action_tokens[mask]

        action = action_tokenizer.decode_token_ids_to_actions(filtered_action_tokens)
        #gripper_state = "closed" if action[6] == 0 else "open"


        print("X +=", action[0], "Y +=", action[1], "Z +=", action[2], "Roll +=", action[3], "Pitch +=", action[4], "Yaw +=", action[5], "Gripper =", action[6])
        #print("Quaternions:", convert_euler_to_quaternion(action[3], action[4], action[5]))

        # Display image
        plt.imshow(image)
        plt.show()

        if batch_idx > 3:
            break

    # Train!
    # with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
    #     vla.train()
    #     optimizer.zero_grad()
    #     for batch_idx, batch in enumerate(dataloader):
    #         with torch.autocast("cuda", dtype=torch.bfloat16):
    #             output: CausalLMOutputWithPast = vla(
    #                 input_ids=batch["input_ids"].to(device_id),
    #                 attention_mask=batch["attention_mask"].to(device_id),
    #                 pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
    #                 labels=batch["labels"],
    #             )
    #             loss = output.loss

    #         # Normalize loss to account for gradient accumulation
    #         normalized_loss = loss / cfg.grad_accumulation_steps

    #         # Backward pass
    #         normalized_loss.backward()

    #         # Compute gradient step index
    #         gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

    #         # Optimizer Step
    #         if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             progress.update()

    #         # Stop training when max_steps is reached
    #         if gradient_step_idx == cfg.max_steps:
    #             print(f"Max step {cfg.max_steps} reached! Stopping training...")
    #             break

def scale_image(image):
    image -= image.min()  # Shift values to start from 0
    image /= image.max()  # Scale to [0, 1]
    image *= 255  # Scale to [0, 255]
    image = image.astype(np.uint8)  # Convert back to uint8 for display
    return image

def convert_euler_to_quaternion(roll, pitch, yaw):
        w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        return np.array([w, x, y, z])


if __name__ == "__main__":
    check_data()
