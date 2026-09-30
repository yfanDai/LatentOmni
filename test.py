import os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
from functools import partial
import torch
from latent_model import apply_latent_omni
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniConfig, AutoTokenizer, AutoProcessor,Qwen2_5OmniThinkerConfig
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
from src.omni_utils import process_mm_info
import torch.distributed as dist
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb
from time import time
import pdb
seed_everything(seed=7)
args=get_args()
USE_AUDIO_IN_VIDEO = True


logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
MODEL_PATH = os.environ.get("LATENTOMNI_MODEL_PATH")
VIDEO_PATH = os.environ.get("LATENTOMNI_VIDEO_PATH")
if not MODEL_PATH or not VIDEO_PATH:
    raise ValueError("Set LATENTOMNI_MODEL_PATH and LATENTOMNI_VIDEO_PATH before running test.py")

patch=14 # processor.image_processor.patch_size
# Use slow processor to avoid fast-processor info spam and behavioral drift
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)

processor.tokenizer.add_tokens("<Unified_Latent_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<Unified_Latent>", special_tokens=True)
processor.tokenizer.add_tokens("</Unified_Latent>", special_tokens=True)

config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL_PATH)

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
)



tokenizer = processor.tokenizer


latent_start_idx = processor.tokenizer("<Unified_Latent>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("</Unified_Latent>", return_tensors="pt")["input_ids"][0]
latent_pad_idx = processor.tokenizer("<Unified_Latent_pad>", return_tensors="pt")["input_ids"][0]
end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
img_start_idx = processor.tokenizer("<|vision_bos|>", return_tensors="pt")["input_ids"][0]
img_end_idx = processor.tokenizer("<|vision_eos|>", return_tensors="pt")["input_ids"][0]
img_pad_idx = processor.tokenizer("<|VIDEO|>", return_tensors="pt")["input_ids"][0]
audio_start_idx = processor.tokenizer("<|audio_bos|>", return_tensors="pt")["input_ids"][0]
audio_end_idx = processor.tokenizer("<|audio_eos|>", return_tensors="pt")["input_ids"][0]
audio_pad_idx = processor.tokenizer("<|AUDIO|>", return_tensors="pt")["input_ids"][0]
observation_start_idx = processor.tokenizer("<observation>", return_tensors="pt")["input_ids"][0]
observation_end_idx = processor.tokenizer("</observation>", return_tensors="pt")["input_ids"][0]


SPECIAL_id = {
    "v_start": img_start_idx,
    "v_end": img_end_idx,
    "img_pad": img_pad_idx,
    "abs_start": latent_start_idx,
    "abs_end": latent_end_idx,
    "abs_pad": latent_pad_idx,
    "ans_start": answer_start_pattern,
    "obs_start": observation_start_idx,
    "obs_end": observation_end_idx
}

model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)
model.config.video_token_id = int(img_pad_idx)
model.config.video_start_id = int(img_start_idx)
model.config.video_end_id = int(img_end_idx)
model.config.audio_token_id = int(audio_pad_idx)
model.config.audio_start_id = int(audio_start_idx)
model.config.audio_end_id = int(audio_end_idx)
model.config.answer_start_pattern = answer_start_pattern.tolist()
model.config.obs_start_id = int(observation_start_idx)
model.config.obs_end_id = int(observation_end_idx)


PROMPT = """
You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
"""


QUESTION = "What is the primary purpose of the white car maneuvering through the water-filled obstacle course with orange cones?\nA. Testing acceleration performance on wet surfaces\nB. Conducting routine vehicle safety inspections\nC. Demonstrating vehicle water resistance capabilities\nD. Simulating emergency response driving scenarios"


conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": PROMPT}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH},
            {"type": "text", "text": QUESTION}
        ],
    },
]



USE_AUDIO_IN_VIDEO = True
# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, videos = videos,images = images, audio = audios, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)
inputs["latent_mode"] = True
inputs["latent_size"] = 40
# 生成输出
# torch.save(inputs["input_features"], "single_tensor.pt")
# segments = parse_modalities(inputs["input_ids"], tokenizer)
outputs = model.generate(
    **inputs,
    temperature=0.4,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[: ,inputs["input_ids"].shape[1] :], skip_special_tokens=False)[0]
print(response)
