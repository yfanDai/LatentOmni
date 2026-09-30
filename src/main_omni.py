import os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
from functools import partial
import torch
from latent_model import apply_latent_omni
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniConfig, AutoTokenizer, AutoProcessor,Qwen2_5OmniThinkerConfig
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from src.omni_utils import process_mm_info, process_audio_info, process_vision_info
import torch.distributed as dist
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb
from time import time
import pdb
import gc
seed_everything(seed=7)
args=get_args()
USE_AUDIO_IN_VIDEO = True
os.environ["MALLOC_ARENA_MAX"] = "2"

# Optional: enable anomaly detection when debugging in-place grad issues
if os.environ.get("TORCH_ANOMALY", "0") == "1":
    try:
        torch.autograd.set_detect_anomaly(True)
        logging.info("Enabled torch.autograd anomaly detection (TORCH_ANOMALY=1)")
    except Exception:
        pass

# DDP-friendly logging: only rank0 writes file
_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
_handlers = [logging.StreamHandler()]
if _rank == 0 and getattr(args, 'log_file', None):
    _handlers.insert(0, logging.FileHandler(args.log_file, mode='a', encoding='utf-8'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=_handlers,
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor

patch=14 # processor.image_processor.patch_size
# Use slow processor to avoid fast-processor info spam and behavioral drift
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=True, trust_remote_code=True)

if _rank == 0:
    # Rewrite deprecated preprocessor.json into video_preprocessor.json by re-saving once
    try:
        processor.save_pretrained(args.load_model_path)
        if args.wandb_name is not None:
            wandb.init(project='Latent-Think',entity="Latent-Think",name=args.wandb_name,config={"ce_emphasize_factor":args.ce_emphasize_factor,"sft_analysis_ratio":args.sft_analysis_ratio})
    except Exception as _e:
        logging.debug(f"Processor save_pretrained skip: {_e}")

processor.tokenizer.add_tokens("<Unified_Latent_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<Unified_Latent>", special_tokens=True)
processor.tokenizer.add_tokens("</Unified_Latent>", special_tokens=True)
processor.tokenizer.add_tokens("<observation>", special_tokens=True)
processor.tokenizer.add_tokens("</observation>", special_tokens=True)

config = Qwen2_5OmniThinkerConfig.from_pretrained(args.load_model_path)

config.stage = args.stage
# Avoid `use_cache=True` with gradient checkpointing warnings; training doesn't need cache
config.use_cache = False
if args.sync_proj_dim is not None:
    config.sync_proj_dim = args.sync_proj_dim
config.sync_max_target_len = args.sync_max_target_len
config.detach_sync_inputs = args.detach_sync_inputs
config.lambda_sync = args.sync_weight
config.lambda_alignment = args.alignment_weight
# Some Qwen configs carry an unrecognized `loss_type=None` which triggers a warning; set explicitly
try:
    setattr(config, 'loss_type', 'ForCausalLMLoss')
except Exception:
    pass


# Prefer Trainer-managed device placement (DDP/Accelerate). Avoid device_map="auto" here.
# Enable TF32 for faster matmul on Ampere+ if available.
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,

    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

model.config.text_config.latent_token_id = int(latent_pad_idx)
model.config.text_config.latent_start_id = int(latent_start_idx)
model.config.text_config.latent_end_id = int(latent_end_idx)
model.config.text_config.video_token_id = int(img_pad_idx)
model.config.text_config.video_start_id = int(img_start_idx)
model.config.text_config.video_end_id = int(img_end_idx)
model.config.text_config.audio_token_id = int(audio_pad_idx)
model.config.text_config.audio_start_id = int(audio_start_idx)
model.config.text_config.audio_end_id = int(audio_end_idx)
model.config.text_config.answer_start_pattern = answer_start_pattern.tolist()
model.config.text_config.obs_start_id = int(observation_start_idx)
model.config.text_config.obs_end_id = int(observation_end_idx)

for param in model.visual.parameters():
    param.requires_grad = False


def collate_fn_sft_stage1(examples):
    # examples: list of {conversation: [...], sample_id: int}
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    # breakpoint()
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    texts = fill_unified_latent(texts,args.latent_size)

    # replace <Unified_Latent></Unified_Latent> with <|vision_bos|><|VIDEO|><|vision_eos|> for each <|im_start|>assistant content
    texts = [text for text in texts]
    #pdb.set_trace()
    ################################################
    # teacher
    ################################################

    #video

    audio_inputs, image_inputs, video_inputs = process_mm_info(examples,use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # if args.image_resize == "global":
    #     image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    # elif args.image_resize == "clear_question_img":
    #     image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)


    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts,audio = audio_inputs, images=image_inputs, videos = video_inputs, return_tensors="pt", padding=True,use_audio_in_video=USE_AUDIO_IN_VIDEO)
    total_video_pads = 0
    for txt in texts:
        total_video_pads += txt.count("<|VIDEO|>")
    batch['teacher_pixel_values_videos'] = teacher_batch['pixel_values_videos']
    batch['teacher_video_grid_thw'] = teacher_batch['video_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    batch['teacher_video_second_per_grid'] = teacher_batch['video_second_per_grid']
    batch['teacher_feature_attention_mask'] = teacher_batch['feature_attention_mask']
    batch['teacher_input_features'] = teacher_batch['input_features']
    batch['use_audio_in_video'] = USE_AUDIO_IN_VIDEO
    batch["student_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, img_pad_idx, img_start_idx, img_end_idx,latent_pad_idx, latent_end_idx, audio_pad_idx, audio_start_idx, audio_end_idx])
    return batch

def collate_fn_sft_stage2(examples):
    # examples: list of {conversation: [...], sample_id: int}
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    texts = replace_img_pad_with_latent_pad(texts, args.latent_size, "<Unified_Latent_pad>")
    # replace <Unified_Latent></Unified_Latent> with <|vision_bos|><|VIDEO|><|vision_eos|> for each <|im_start|>assistant content
    texts = [text for text in texts]
    #pdb.set_trace()
    ################################################
    # teacher
    ################################################

    #video


    audio_inputs, image_inputs, video_inputs = process_mm_info(examples,use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # if args.image_resize == "global":
    #     image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    # elif args.image_resize == "clear_question_img":
    #     image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)


    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts,audio = audio_inputs, images=image_inputs, videos = video_inputs, return_tensors="pt", padding=True,use_audio_in_video=USE_AUDIO_IN_VIDEO)
    total_video_pads = 0
    for txt in texts:
        total_video_pads += txt.count("<|VIDEO|>")
    batch['teacher_pixel_values_videos'] = teacher_batch['pixel_values_videos']
    batch['teacher_video_grid_thw'] = teacher_batch['video_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    batch['teacher_video_second_per_grid'] = teacher_batch['video_second_per_grid']
    batch['teacher_feature_attention_mask'] = teacher_batch['feature_attention_mask']
    batch['teacher_input_features'] = teacher_batch['input_features']
    batch['use_audio_in_video'] = USE_AUDIO_IN_VIDEO
    batch["student_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, img_pad_idx, img_start_idx, img_end_idx,latent_pad_idx, audio_pad_idx, audio_start_idx, audio_end_idx])
    return batch

def collate_fn_sft_stage3(examples):
    # examples: list of {conversation: [...], sample_id: int}
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    # breakpoint()
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    texts = fill_unified_latent(texts,args.latent_size)

    # replace <Unified_Latent></Unified_Latent> with <|vision_bos|><|VIDEO|><|vision_eos|> for each <|im_start|>assistant content
    texts = [text for text in texts]
    #pdb.set_trace()
    ################################################
    # teacher
    ################################################

    #video

    audio_inputs, image_inputs, video_inputs = process_mm_info(examples,use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # if args.image_resize == "global":
    #     image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    # elif args.image_resize == "clear_question_img":
    #     image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)


    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts,audio = audio_inputs, images=image_inputs, videos = video_inputs, return_tensors="pt", padding=True,use_audio_in_video=USE_AUDIO_IN_VIDEO)
    total_video_pads = 0
    for txt in texts:
        total_video_pads += txt.count("<|VIDEO|>")
    batch['teacher_pixel_values_videos'] = teacher_batch['pixel_values_videos']
    batch['teacher_video_grid_thw'] = teacher_batch['video_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    batch['teacher_video_second_per_grid'] = teacher_batch['video_second_per_grid']
    batch['teacher_feature_attention_mask'] = teacher_batch['feature_attention_mask']
    batch['teacher_input_features'] = teacher_batch['input_features']
    batch['use_audio_in_video'] = USE_AUDIO_IN_VIDEO
    batch["student_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["obs_poss"] = find_obs_idx(batch["teacher_input_ids"],observation_start_idx, observation_end_idx)
    batch["question_av_poss"] = find_question_AV(batch["teacher_input_ids"], img_pad_idx, img_start_idx, img_end_idx, audio_pad_idx, audio_start_idx, audio_end_idx)
    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, img_pad_idx, img_start_idx, img_end_idx,latent_pad_idx, latent_end_idx, audio_pad_idx, audio_start_idx, audio_end_idx])
    del teacher_batch
    del audio_inputs, image_inputs, video_inputs
    del teacher_texts, texts
    gc.collect()
    return batch


preprocess_function = task_preporcess_config[args.task]
all_train_dataset = []
for data_path in args.data_path:
    if data_path.endswith('.jsonl'):
        train_dataset = load_jsonl_dataset(data_path)
    elif data_path.endswith('.json'):
        train_dataset = load_json_dataset(data_path)
    all_train_dataset.extend(train_dataset[:])
if args.shuffle_train:
    random.seed(7)
    random.shuffle(all_train_dataset)

import gc
import psutil
from torch.utils.data import Dataset

# 1. 定义惰性加载的 Dataset
class LazySFTDataset(Dataset):
    def __init__(self, raw_data, preprocess_fn, dataset_root, allow_no_observation):
        self.raw_data = raw_data
        self.preprocess_fn = preprocess_fn
        self.dataset_root = dataset_root
        self.allow_no_observation = allow_no_observation

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        processed = None
        # 使用 while 循环应对 preprocess_function 返回 None 的情况（过滤无效数据）
        # 如果当前数据无效，则随机抽取一条直到有效为止，保证 batch size 不变
        while processed is None:
            sample = self.raw_data[idx]
            processed = self.preprocess_fn(
                sample, 
                dataset_root=self.dataset_root, 
                allow_no_observation=self.allow_no_observation
            )
            if processed is None:
                idx = random.randint(0, len(self.raw_data) - 1)
                
        return processed

# 2. 将原始 JSON 数据转换为 Lazy Dataset
# 注意：此时 all_train_dataset 里只有极轻量的原始 json 字典
train_dataset = LazySFTDataset(
    raw_data=all_train_dataset,
    preprocess_fn=preprocess_function,
    dataset_root=args.dataset_root,
    allow_no_observation=args.allow_no_observation
)

logging.info(f"成功构建 LazySFTDataset，共包含 {len(train_dataset)} 条数据。")

# 3. 强制进行垃圾回收，确保没有中间变量残留
gc.collect()
logging.info(f"进入 Trainer 前的纯净内存占用: {psutil.Process(os.getpid()).memory_info().rss / 1024**3:.2f} GB")

#train_dataset = [d for d in [preprocess_function(sample) for sample in all_train_dataset[:]] if d is not None]


dataset_names = ""
for data_path in args.data_path:
    dataset_name = data_path.split("/")[-2]
    dataset_names += f"-{dataset_name}"


save_dir = args.save_model_path
# breakpoint()
if args.stage == 'sft_stage1':
    CustomTrainer = CustomTrainerSFT_STAGE1
    collate_fn = partial(collate_fn_sft_stage1)
elif args.stage == 'sft_stage2':
    CustomTrainer = CustomTrainerSFT_STAGE2
    collate_fn = partial(collate_fn_sft_stage2)
elif args.stage == 'sft_stage3':
    CustomTrainer = CustomTrainerSFT_STAGE3
    collate_fn = partial(collate_fn_sft_stage3)
elif args.stage == 'sft_stage4':
    CustomTrainer = CustomTrainerSFT_STAGE4
    collate_fn = partial(collate_fn_sft_stage3)

if args.deepspeed != "":
    print(f"Note: DeepSpeed is enabled. Using the deepspeed config in {args.deepspeed} (the bsz per device and gradient_accumulation_steps will be adopted from the deepspeed config)")
is_parallel = int(os.environ.get("WORLD_SIZE", "1")) > 1
gradient_checkpointing = False



training_args = SFTConfig(
    output_dir=save_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.grad_accum_steps,
    warmup_steps=10,
    learning_rate=args.lr,
    weight_decay=0.01,
    logging_steps=args.log_freq,
    save_strategy="steps",
    save_steps=args.save_freq,
    save_total_limit=10,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=gradient_checkpointing,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=['wandb'] if args.wandb_name is not None else [],
    logging_dir='./logs/',
    logging_strategy='steps',
    # Avoid FLOPs estimation logs (set to False through env if needed)
    disable_tqdm=False,
    # DDP related
    ddp_backend="nccl" if is_parallel else None,
    ddp_find_unused_parameters=False if is_parallel else None,
    dataloader_num_workers=0 if is_parallel else 0,
    dataloader_pin_memory=False,
    # Save only on global rank 0 when running multi-node
    save_on_each_node=False,
    # DeepSpeed config (if provided via --deepspeed)
    deepspeed=(args.deepspeed if getattr(args, 'deepspeed', '') else None),
)

# ---- Inject custom SFT analysis flags into training_args so CustomTrainerSFT can access them ----
setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
setattr(training_args, 'alignment_weight', args.alignment_weight)
setattr(training_args, 'sync_weight', args.sync_weight)

if args.stage == 'sft_stage1' or args.stage == 'sft_stage2':
    setattr(training_args, 'teacher_reps_dir', args.teacher_reps_dir)
    setattr(training_args, 'alignment_layer', args.alignment_layer)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'latent_size', args.latent_size)
    setattr(training_args, 'emphasize_latent_weight', args.emphasize_latent_weight)
    setattr(training_args, 'teacher_latent_dir', args.teacher_latent_dir)
    setattr(training_args, 'image_resize', args.image_resize)
    setattr(training_args, 'sft_stage2_align_poss', args.sft_stage2_align_poss)

# Initialize the trainer (callbacks that need trainer instance will be added after)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    exp_name=args.save_model_path.split('/')[-1]
)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(training_args.output_dir)
