# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, VLRLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
    )

    if config.train_max_samples is not None:
        max_samples = min(config.train_max_samples, len(train_dataset))
        indices = list(range(len(train_dataset)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:max_samples])
    
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
    )
    
    if config.val_max_samples is not None:
        max_samples = min(config.val_max_samples, len(val_dataset))
        indices = list(range(len(val_dataset)))
        val_dataset = torch.utils.data.Subset(val_dataset, indices[:max_samples])
    
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
        shuffle=False,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader


def create_vl_dataloader(
    config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]
):
    """
    DataLoader factory for pure VL training — uses VLRLHFDataset which accepts
    any dataset format without hard-coded dataset-name routing.
    """
    def _make(data_path, max_samples, is_train):
        ds = VLRLHFDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
        )
        if max_samples is not None and max_samples > 0:
            max_samples = min(max_samples, len(ds))
            ds = torch.utils.data.Subset(ds, list(range(max_samples)))
        return ds

    train_dataset = _make(config.train_files, config.train_max_samples, is_train=True)
    val_dataset = _make(config.val_files, config.val_max_samples, is_train=False)

    if config.shuffle:
        gen = torch.Generator()
        gen.manual_seed(config.seed)
        train_sampler = RandomSampler(data_source=train_dataset, generator=gen)
    else:
        train_sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=train_sampler,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"[VL] Size of train dataloader: {len(train_dataloader)}")
    print(f"[VL] Size of val dataloader:   {len(val_dataloader)}")
    return train_dataloader, val_dataloader
