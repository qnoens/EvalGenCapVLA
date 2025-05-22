"""
verify_openvla.py

Given an HF-exported OpenVLA model, attempt to load via AutoClasses, and verify forward() and predict_action().
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import json

from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
from prismatic.models.backbones.llm.prompting.vicuna_v15_prompter import VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# === Verification Arguments
MODEL_PATH = "openvla/openvla-7b"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
INSTRUCTION = "push the button"


def get_openvla_prompt(instruction: str) -> str:
    if "v01" in MODEL_PATH:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    

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


@torch.inference_mode()
def verify_openvla(cfg: DataCheckerConfig) -> None:
    #print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    open_vla_weights_path = '/fast_storage/qnoens/OpenVLA/training/log/openvla-7b+pushbutton_mujoco_rlds+b16+lr-0.0005+lora-r32+dropout-0.0'
    run_dir = '/fast_storage/qnoens/OpenVLA/training/log'

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(open_vla_weights_path, trust_remote_code=True)

    # === BFLOAT16 + FLASH-ATTN MODE ===
    vla = AutoModelForVision2Seq.from_pretrained(open_vla_weights_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)

    dataset_statistics_path = os.path.join(run_dir, "dataset_statistics.json")
    print(dataset_statistics_path, os.path.isfile(dataset_statistics_path))
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats

    # print("[*] Loading in BF16 with Flash-Attention Enabled")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # ).to(device)

    # === 8-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~9GB of VRAM Passive || 10GB of VRAM Active] ===
    # print("[*] Loading in 8-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    # === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
    # print("[*] Loading in 4-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.float16,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

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
    save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    # collator = PaddedCollatorForActionPrediction(
    #     processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    # )
    # dataloader = DataLoader(
    #     vla_dataset,
    #     batch_size=cfg.batch_size,
    #     sampler=None,
    #     collate_fn=collator,
    #     num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    # )


    print("[*] Iterating with Randomly Generated Images")
    for _ in range(100):
        prompt = get_openvla_prompt(INSTRUCTION)
        image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))

        # === BFLOAT16 MODE ===
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # === 8-BIT/4-BIT QUANTIZATION MODE ===
        # inputs = processor(prompt, image).to(device, dtype=torch.float16)

        # Run OpenVLA Inference
        start_time = time.time()
        action = vla.predict_action(**inputs, unnorm_key="pushbutton_mujoco_rlds", do_sample=False)
        print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {action}")


if __name__ == "__main__":
    verify_openvla()

