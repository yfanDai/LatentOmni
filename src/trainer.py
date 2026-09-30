from trl import SFTTrainer, SFTConfig
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import logging
import torch
import os, csv, torch, datetime
import gc
import numpy as np
import math
from time import time
from torch import nn
from transformers.trainer import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# 关闭 torchvision 的 INFO 日志（只保留 WARNING/ERROR）
logging.getLogger("torchvision").setLevel(logging.WARNING)

# 关闭 decord 的 INFO 日志（只保留 WARNING/ERROR）
logging.getLogger("decord").setLevel(logging.WARNING)

def omni_compute_loss(
    self,
    model: nn.Module,
    inputs: dict[str, Union[torch.Tensor, Any]],
    return_outputs: bool = False,
    num_items_in_batch: Optional[torch.Tensor] = None,
):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Args:
        model (`nn.Module`):
            The model to compute the loss for.
        inputs (`dict[str, Union[torch.Tensor, Any]]`):
            The input data for the model.
        return_outputs (`bool`, *optional*, defaults to `False`):
            Whether to return the model outputs along with the loss.
        num_items_in_batch (Optional[torch.Tensor], *optional*):
            The number of items in the batch. If num_items_in_batch is not passed,

    Returns:
        The loss of the model along with its output if return_outputs was set to True

    Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
    make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **kwargs}

    #check if Omni Model
    def _is_omni_model(m: nn.Module) -> bool:
        unwrapped = self.accelerator.unwrap_model(m)
        return "Omni" in unwrapped._get_name()

    is_omni = _is_omni_model(model)
    use_audio_in_video = inputs.pop("use_audio_in_video", False)
    if is_omni:
        outputs = model(**inputs,use_audio_in_video=use_audio_in_video)
    else:
        outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    # User-defined compute_loss function

    if self.compute_loss_func is not None:
        if labels is None:
            print(
                "Trainer: `compute_loss_func` is defined but `labels=None`. "
                "Your custom loss function will still be called with labels=None. "
            )
        loss = self.compute_loss_func(
            outputs,
            labels,
            num_items_in_batch=num_items_in_batch,
        )
    # Default HF loss handling (label smoothing) if no custom loss function
    elif labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_name = (
            unwrapped_model.base_model.model._get_name()
            if _is_peft_model(unwrapped_model)
            else unwrapped_model._get_name()
        )
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if (
        self.args.average_tokens_across_devices
        and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        and num_items_in_batch is not None
    ):
        loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

    # if torch.distributed.get_rank() == 0:
    #     breakpoint()
    # torch.distributed.barrier()

    return (loss, outputs) if return_outputs else loss

#rewrite loss compute to fit in Omni Model
Trainer.compute_loss = omni_compute_loss


def compute_latents_only_loss(latents, loss_for_latents):
    '''
    Compute a loss (`loss_for_latents`) that backpropagates only through the latent embeddings `latents`.
    '''
    def _flatten_tensors(x):
                # Flatten nested [list/tuple of Tensors] into a flat list of Tensors
                if isinstance(x, (list, tuple)):
                    out = []
                    for y in x:
                        out.extend(_flatten_tensors(y))
                    return out
                return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,   # we won't reuse the 3rd graph
        create_graph=False,   # stop higher-order graph
        allow_unused=True     # in case some ce vectors are not used
    )

    # Replace None with zeros for unused elements
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            # Create a zero tensor on the same device/dtype/shape
            g = torch.zeros_like(v)
        safe_grads.append(g.detach())  # detach to stop any 3rd-forward param pathg

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss

def load_offline_tensor(tensor_dir, batch_metadata, alignment_layer="all_layers", rep_type="rep", align_poss="obs"):
    '''
    Load precomputed teacher representations (observation tokens for the alignment in SFT stage 2 or the latent embeddings for SFT stage 3)
    '''
    teacher_reps = None
    latents_list = []
    for metadata in batch_metadata:
        dataset_name = metadata['dataset_name']
        sample_id = metadata['sample_id']
        metadata_info = f"{alignment_layer}_{dataset_name}_{sample_id}"
        if align_poss == 'obs':
            metadata_str = f"{rep_type}_{metadata_info}.pt"
        elif align_poss == 'latent_end':
            metadata_str = f"{rep_type}_latent_end_{metadata_info}.pt"
        path = os.path.join(tensor_dir, metadata_str)
        if not os.path.isfile(path):
            latents_list = []
            raise RuntimeError(f"Missing teacher latent file: {path}")
        data = torch.load(path, map_location='cpu')
        latents_list.append(data['latent'].detach())
    if batch_metadata is not None and len(latents_list) == len(batch_metadata):
        teacher_reps = latents_list
    return teacher_reps


class CustomTrainerSFT_STAGE1(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.alignment_weight = getattr(self.args, "alignment_weight", 0.5)
        self.sync_weight = getattr(self.args, "sync_weight", 1.0)
        self.ce_emphasize_factor = getattr(self.args, "ce_emphasize_factor", 4.0)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.teacher_ce_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_steps = 0
        self.teacher_ce_loss_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_loss_steps = 0
        self.sync_loss_cum = 0.0
        self.sync_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['latent_mode'] = True
        inputs['freeze_vit'] = True
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values_videos'] = inputs['teacher_pixel_values_videos']
        inputs['video_grid_thw'] = inputs['teacher_video_grid_thw']
        inputs['video_second_per_grid'] = inputs['teacher_video_second_per_grid']
        inputs['feature_attention_mask'] = inputs['teacher_feature_attention_mask']
        inputs['input_features'] = inputs['teacher_input_features']

        model.gradient_checkpointing_disable() # since we set use_cache=True in latent forward, we must disable grad checkpointing
        inputs['loss_type'] = ['alignment', 'sync']
        inputs['output_hidden_states'] = False
        inputs['alignment_poss'] = inputs['student_alignment_poss']

        student_outputs_latent = model(**inputs)
        inputs['reconstruction_av_emb'] = student_outputs_latent.reconstruction_av_emb
        inputs['attention_mask'][inputs['attention_mask'] == 0] = 1
        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['teacher_labels']
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce', 'alignment']
        inputs['compute_emphasize_acc'] = True

        inputs['labels'] = inputs['teacher_labels']

        # Dynamic warmup factor passed to model.forwar
        teacher_total_loss, teacher_outputs = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        teacher_ce_loss = None
        if hasattr(teacher_outputs, 'loss_dict') and teacher_outputs.loss_dict is not None:
            teacher_ce_loss = teacher_outputs.loss_dict.get('ce')
        if teacher_ce_loss is None:
            teacher_ce_loss = teacher_total_loss

        if getattr(student_outputs_latent, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs_latent, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        alignment_loss = student_outputs_latent.loss_dict['alignment']
        sync_loss = student_outputs_latent.loss_dict.get('sync')
        loss = teacher_ce_loss
        if alignment_loss is not None:
            loss = loss + self.alignment_weight * alignment_loss
        if sync_loss is not None:
            loss = loss + self.sync_weight * sync_loss

        outputs_student_loss = teacher_ce_loss.item()
        self.teacher_ce_loss_cum += outputs_student_loss
        self.teacher_ce_loss_steps += 1
        if sync_loss is not None:
            self.sync_loss_cum += sync_loss.item()
            self.sync_loss_steps += 1

        # if getattr(teacher_outputs, 'mean_emphasize_acc', None) is not None:
        #     self.observation_token_acc += getattr(teacher_outputs, 'mean_emphasize_acc')
        #     self.observation_token_acc_step += 1

        del teacher_outputs
        # gc.collect()
        # torch.cuda.empty_cache()
        
        return (loss, None) if return_outputs else loss

    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.teacher_ce_steps > 0:
            merged["student_ce_loss"] = round(self.teacher_ce_cum / max(1, self.teacher_ce_steps), 6)
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0
        if self.sync_loss_steps > 0:
            merged["sync_loss"] = round(self.sync_loss_cum / max(1, self.sync_loss_steps), 6)
            self.sync_loss_cum = 0.0
            self.sync_loss_steps = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)



class CustomTrainerSFT_STAGE2(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.teacher_ce_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['latent_mode'] = False
        inputs['freeze_vit'] = True
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values_videos'] = inputs['teacher_pixel_values_videos']
        inputs['video_grid_thw'] = inputs['teacher_video_grid_thw']
        inputs['video_second_per_grid'] = inputs['teacher_video_second_per_grid']
        inputs['feature_attention_mask'] = inputs['teacher_feature_attention_mask']
        inputs['input_features'] = inputs['teacher_input_features']
        inputs['labels'] = inputs['teacher_labels']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self.args.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        inputs['compute_emphasize_acc'] = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        self.teacher_ce_cum += teacher_ce_loss.item()
        self.teacher_ce_steps += 1

        if getattr(teacher_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.teacher_ce_steps > 0:
            merged["student_ce_loss"] = round(self.teacher_ce_cum / max(1, self.teacher_ce_steps), 6)
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerSFT_STAGE3(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.alignment_weight = getattr(self.args, "alignment_weight", 0.1)
        self.sync_weight = getattr(self.args, "sync_weight", 1.0)
        self.ce_emphasize_factor = getattr(self.args, "ce_emphasize_factor", 4.0)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        
        # 1. 【新增】初始化 ce_loss 和 alignment_loss 的累积变量
        self.ce_loss_cum = 0.0          # 累积 ce_loss (teacher_ce_loss)
        self.ce_loss_steps = 0           # 累积步数
        self.alignment_loss_cum = 0.0    # 累积 alignment_loss
        self.alignment_loss_steps = 0     # 累积步数
        self.sync_loss_cum = 0.0
        self.sync_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies.
        """
        inputs['latent_mode'] = False
        inputs['freeze_vit'] = True
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['labels'] = inputs['teacher_labels']
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['compute_emphasize_acc'] = True
        # 不预先传入 reconstruction_av_emb，让模型内部在单次 Forward 中自动提取
        inputs.pop('reconstruction_av_emb', None)
        inputs.pop('ce_patch_pos', None)
        inputs.pop('ce_patch_vec', None)

        inputs['pixel_values_videos'] = inputs['teacher_pixel_values_videos']
        inputs['video_grid_thw'] = inputs['teacher_video_grid_thw']
        inputs['video_second_per_grid'] = inputs['teacher_video_second_per_grid']
        inputs['feature_attention_mask'] = inputs['teacher_feature_attention_mask']
        inputs['input_features'] = inputs['teacher_input_features']
        # 对非 audio/video 区域恢复 mask
        inputs['attention_mask'] = inputs['teacher_attention_mask'].clone()
        inputs['attention_mask'][inputs['attention_mask'] == 0] = 1

        inputs['loss_type'] = ['ce', 'alignment', 'sync']  # 单次 Forward 同时计算三种 loss
        # 单次合并 Forward
        if 'ce' in inputs['loss_type']:
            teacher_total_loss, teacher_outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            teacher_ce_loss = None
            if hasattr(teacher_outputs, 'loss_dict') and teacher_outputs.loss_dict is not None:
                teacher_ce_loss = teacher_outputs.loss_dict.get('ce')
            if teacher_ce_loss is None:
                teacher_ce_loss = teacher_total_loss
        else:
            teacher_outputs = model(**inputs)
            teacher_ce_loss = None
            self.alignment_weight = 1.0

        # 从单次 Forward 的输出中读取 alignment_loss
        alignment_loss = None
        sync_loss = None
        if hasattr(teacher_outputs, 'loss_dict') and teacher_outputs.loss_dict is not None:
            alignment_loss = teacher_outputs.loss_dict.get('alignment')
            sync_loss = teacher_outputs.loss_dict.get('sync')

        loss = teacher_ce_loss if teacher_ce_loss is not None else None
        if loss is None and alignment_loss is not None:
            loss = self.alignment_weight * alignment_loss
        if alignment_loss is not None and teacher_ce_loss is not None:
            loss = loss + self.alignment_weight * alignment_loss
        if sync_loss is not None:
            loss = (0.0 if loss is None else loss) + self.sync_weight * sync_loss
        # 2. 【新增】累积 ce_loss 和 alignment_loss
        if teacher_ce_loss is not None:
            self.ce_loss_cum += teacher_ce_loss.item()
            self.ce_loss_steps += 1
        if alignment_loss is not None:
            self.alignment_loss_cum += alignment_loss.item()
            self.alignment_loss_steps += 1
        if sync_loss is not None:
            self.sync_loss_cum += sync_loss.item()
            self.sync_loss_steps += 1

        acc = getattr(teacher_outputs, 'mean_emphasize_acc', None)
        if acc is not None:
            # 确保如果是 Tensor，一定要取 .item()
            self.observation_token_acc += acc.item() if isinstance(acc, torch.Tensor) else acc

        del teacher_outputs

        return (loss, None) if return_outputs else loss

    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        
        # 3. 【新增】将累积的 ce_loss 和 alignment_loss 写入日志
        if self.ce_loss_steps > 0:
            merged["ce_loss"] = round(self.ce_loss_cum / max(1, self.ce_loss_steps), 6)
            self.ce_loss_cum = 0.0
            self.ce_loss_steps = 0
        if self.alignment_loss_steps > 0:
            merged["alignment_loss"] = round(self.alignment_loss_cum / max(1, self.alignment_loss_steps), 6)
            self.alignment_loss_cum = 0.0
            self.alignment_loss_steps = 0
        if self.sync_loss_steps > 0:
            merged["sync_loss"] = round(self.sync_loss_cum / max(1, self.sync_loss_steps), 6)
            self.sync_loss_cum = 0.0
            self.sync_loss_steps = 0

        # 保留原有的 observation_token_acc 记录
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)


#使用已经经过MSE Loss训练后的模型进行CE训练
class CustomTrainerSFT_STAGE4(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        self.alignment_weight = 0.1
        self.ce_emphasize_factor= 4.0
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        
        # 1. 【新增】初始化 ce_loss 和 alignment_loss 的累积变量
        self.ce_loss_cum = 0.0          # 累积 ce_loss (teacher_ce_loss)
        self.ce_loss_steps = 0           # 累积步数
        self.alignment_loss_cum = 0.0    # 累积 alignment_loss
        self.alignment_loss_steps = 0     # 累积步数

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies.
        """
        inputs['latent_mode'] = True
        inputs['freeze_vit'] = True
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['compute_emphasize_acc'] = True
        # 不预先传入 reconstruction_av_emb，让模型内部在单次 Forward 中自动提取
        inputs.pop('reconstruction_av_emb', None)
        inputs.pop('ce_patch_pos', None)
        inputs.pop('ce_patch_vec', None)

        inputs['pixel_values_videos'] = inputs['teacher_pixel_values_videos']
        inputs['video_grid_thw'] = inputs['teacher_video_grid_thw']
        inputs['video_second_per_grid'] = inputs['teacher_video_second_per_grid']
        inputs['feature_attention_mask'] = inputs['teacher_feature_attention_mask']
        inputs['input_features'] = inputs['teacher_input_features']
        inputs['attention_mask'] = inputs['teacher_attention_mask'].clone()
        # 对非 audio/video 区域恢复 mask
        inputs['pre_compute'] = True
        model.gradient_checkpointing_disable()
        with torch.no_grad():
            outputs = model(**inputs)
        inputs['attention_mask'][inputs['attention_mask'] == 0] = 1
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['labels'] = inputs['teacher_labels']
        inputs['latent_mode'] = False
        inputs['reconstruction_av_emb'] = outputs.reconstruction_av_emb
        inputs['loss_type'] = ['ce']  # 单次 Forward 同时计算两种 loss
        # 单次合并 Forward
        teacher_ce_loss, teacher_outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # 从单次 Forward 的输出中读取 alignment_loss
        alignment_loss = None
        if hasattr(teacher_outputs, 'loss_dict') and teacher_outputs.loss_dict is not None:
            alignment_loss = teacher_outputs.loss_dict.get('alignment')

        if alignment_loss is not None:
            loss = teacher_ce_loss + self.alignment_weight * alignment_loss
        else:
            loss = teacher_ce_loss

        # 2. 【新增】累积 ce_loss 和 alignment_loss
        self.ce_loss_cum += teacher_ce_loss.item()
        self.ce_loss_steps += 1
        if alignment_loss is not None:
            self.alignment_loss_cum += alignment_loss.item()
            self.alignment_loss_steps += 1

        acc = getattr(teacher_outputs, 'mean_emphasize_acc', None)
        if acc is not None:
            # 确保如果是 Tensor，一定要取 .item()
            self.observation_token_acc += acc.item() if isinstance(acc, torch.Tensor) else acc

        del teacher_outputs

        return (loss, None) if return_outputs else loss

    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        
        # 3. 【新增】将累积的 ce_loss 和 alignment_loss 写入日志
        if self.ce_loss_steps > 0:
            merged["ce_loss"] = round(self.ce_loss_cum / max(1, self.ce_loss_steps), 6)
            self.ce_loss_cum = 0.0
            self.ce_loss_steps = 0
        if self.alignment_loss_steps > 0:
            merged["alignment_loss"] = round(self.alignment_loss_cum / max(1, self.alignment_loss_steps), 6)
            self.alignment_loss_cum = 0.0
            self.alignment_loss_steps = 0

        # 保留原有的 observation_token_acc 记录
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)
