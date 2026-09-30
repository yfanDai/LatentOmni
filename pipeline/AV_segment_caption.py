#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量视频 caption 推理脚本（支持 MPI 分片 + 视音频多模态 Qwen3-Omni + 长视频按 5s 顺序分片合并）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from glob import glob
from typing import Iterable, List, Optional, Tuple

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# 【注意】请根据本地库实际名称调整
from qwen_omni_utils import process_mm_info

# 强制开启视频音频联合读取配置
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"



PROMPT_EN = (
    "Provide a **CONCISE** yet complete description of the video using both visual and audio information. "
    "Limit the description to no more than five sentences and avoid redundant details. "
    "Describe only directly observable elements: setting, people, actions, objects, camera movement, and sounds. "
    "For audio, state only clearly identifiable sound sources (e.g., music, speech, instruments if unmistakable). "
    "If speech is present, accurately report the speaker and the spoken content. "
    "Do not infer mood, intent, genre, cultural style, or add interpretation. "
    "Avoid speculation and do not use evaluative or atmospheric language."
)

def get_mpi_info() -> Tuple[int, int]:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", -1))
    if rank == -1:
        rank = int(os.environ.get("PMI_RANK", -1))

    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", -1))
    if size == -1:
        size = int(os.environ.get("PMI_SIZE", -1))

    if rank < 0:
        rank = 0
    if size < 0:
        size = 1
    return rank, size

def get_md5(text: str) -> str:
    """计算字符串的 MD5"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

class VideoCaptioner:
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.seq_max_len = max_model_len
        self.fast_max_pixels = 460800

        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10, "audio": 10}, 
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def get_max_best_max_pixels(self, frame_num: int, text: str) -> int:
        text_len = len(self.processor.tokenizer.encode(text))
        now_vision_len = self.fast_max_pixels / (16 * 16 * 4) * (frame_num / 2)
        max_vision_len = self.seq_max_len - text_len
        if now_vision_len <= 0:
            return self.fast_max_pixels
        best_fast_max_pixels = min(
            int(max_vision_len / now_vision_len * self.fast_max_pixels),
            self.fast_max_pixels,
        )
        return max(best_fast_max_pixels, 1024)

    def build_llm_input(
        self, frame_paths: List[str], audio_paths: List[str], sample_fps: float, prompt: str
    ) -> dict:
        max_pixels = self.get_max_best_max_pixels(len(frame_paths), prompt)

        content_list = []
        if frame_paths:
            content_list.append({
                "type": "video",
                "video": frame_paths,
                "max_pixels": max_pixels,
                "sample_fps": sample_fps,
            })
            
        for audio_path in audio_paths:
            content_list.append({
                "type": "audio",
                "audio": audio_path,
            })
            
        content_list.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        mm_info = process_mm_info(
            messages,
            use_audio_in_video=False,
        )
        
        a_data = mm_info[0] if len(mm_info) > 0 else None
        i_data = mm_info[1] if len(mm_info) > 1 else None
        v_data = mm_info[2] if len(mm_info) > 2 else None

        mm_data = {}
        if i_data is not None:
            mm_data["image"] = i_data
        if v_data is not None:
            mm_data["video"] = v_data
        if a_data is not None:
            mm_data["audio"] = a_data

        return {
            "prompt": formatted_prompt,
            "multi_modal_data": mm_data,
        }

def load_jsonl_data(jsonl_path: str) -> List[dict]:
    """从大 JSONL 文件中读取所有行"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def split_by_rank(data_list: List[dict], rank: int, size: int) -> List[dict]:
    if size <= 1:
        return data_list
    return [item for index, item in enumerate(data_list) if index % size == rank]

def infer_video_chunks(
    captioner: VideoCaptioner,
    video_item: dict,
    audio_dir: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> dict:
    original_video_path = video_item.get("original_video_path", "")
    frames_list = video_item.get("videos", [])
    
    if not original_video_path or not frames_list:
        raise ValueError("JSONL 缺少 original_video_path 或 videos 字段")
        
    md5_digest = get_md5(original_video_path)
    
    # 将长视频拆分为每 10 帧 (5s) 一个 chunk
    chunk_size = 10
    chunk_duration = 5 # 秒
    
    # 计算当前视频用中/英文 Prompt（保持整个视频语言一致）
    selected_prompt = args.prompt_en
    
    llm_inputs = []
    chunks_info = [] # 记录当前 chunk 的时间戳信息

    for i in range(0, len(frames_list[0]), chunk_size):
        chunk_frames = frames_list[0][i : i + chunk_size]
        chunk_index = (i // chunk_size) + 1  # 对应 1, 2, 3...
        
        # 寻找对应的音频: md5_1.wav, md5_2.wav ...
        audio_name = f"{md5_digest}_{chunk_index}.wav"
        audio_path = os.path.join(audio_dir, audio_name)
        segment_audios = [audio_path] if os.path.exists(audio_path) else []
        
        # 10 帧 = 5秒，因此 sample_fps 严格等于 2.0
        sample_fps = 2.0 
        
        llm_input = captioner.build_llm_input(
            frame_paths=chunk_frames, 
            audio_paths=segment_audios, 
            sample_fps=sample_fps, 
            prompt=selected_prompt
        )
        llm_inputs.append(llm_input)
        
        start_time = (chunk_index - 1) * chunk_duration
        end_time = chunk_index * chunk_duration
        chunks_info.append((start_time, end_time))

    # 使用 vLLM 批量推理当前视频的所有 chunks，极大提升 GPU 吞吐量
    sampling_params = SamplingParams(
        top_k=1,
        top_p=1.0,
        max_tokens=args.max_tokens,
        min_tokens=2,
        repetition_penalty=1.0,
    )
    outputs = captioner.llm.generate(llm_inputs, sampling_params=sampling_params)
    
    # 拼装按照时间戳合并的最终字符串
    timestamp_str = ""
    # 遍历 chunks_info 和 outputs，同时记录分段序号（从1开始）
    for seg_num, (info, output) in enumerate(zip(chunks_info, outputs), start=1):
        # 原代码中仍可保留时间戳变量（如果后续需要用到）
        start_t, end_t = info
        # 提取并清理字幕文本
        caption = output.outputs[0].text.strip()
        # 按 [Segment X]: 文本 的格式拼接
        timestamp_str += f"[Segment {seg_num}]: {caption}\n"

    return {
        **video_item,
        "md5": md5_digest,
        "timestamp": timestamp_str.strip(),
        "status": "success",
    }

def write_jsonl(output_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

def load_processed_md5s(output_dir: str, rank: int) -> set:
    """根据写入的新结构，读取已处理过的 md5 集合，防止中断重跑"""
    processed = set()
    patterns = [
        os.path.join(output_dir, f"{rank}.jsonl"),
        os.path.join(output_dir, f"{rank}_*.jsonl"),
        os.path.join(output_dir, f"captions_rank{rank}.jsonl"),
        os.path.join(output_dir, f"captions_rank{rank}_*.jsonl"),
    ]
    for pattern in patterns:
        for path in glob(pattern):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        md5_val = data.get("md5")
                        status = data.get("status", "success")
                        if status == "success" and md5_val:
                            processed.add(md5_val)
            except FileNotFoundError:
                continue
    return processed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM 视频 caption 按时间轴打点推理脚本")
    # 将原本的 input-txt 改成了 input-jsonl
    parser.add_argument("--input-jsonl", required=True, help="包含 videos 列表和 original_video_path 的大 jsonl 文件")
    parser.add_argument("--output-dir", required=True, help="输出 jsonl 目录，按 rank 生成文件")
    # 将原本的 frame-dir-base 替换为 audios 根路径
    parser.add_argument("--audio-dir", required=True, help="拆分后音频所在的根目录")
    
    parser.add_argument(
        "--model-path",
        default="path/to/model",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-en", default=PROMPT_EN, help="英文提示词")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rank, size = get_mpi_info()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    rng = random.Random(args.seed)

    # 1. 读取包含所有的 jsonl 数据
    all_video_data = load_jsonl_data(args.input_jsonl)
    
    # 2. 按 MPI 的 rank 分片任务
    video_data_list = split_by_rank(all_video_data, rank, size)
    total = len(video_data_list)

    # 3. 统计当前 rank 已经处理过的进度
    processed_md5s = load_processed_md5s(args.output_dir, rank)
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    output_path = os.path.join(args.output_dir, f"{rank}_caption.jsonl")
    
    print(f"Rank {rank}/{size}: {total} videos -> {output_path}")
    if processed_md5s:
        print(f"Rank {rank}: resume {len(processed_md5s)} processed videos")

    captioner = VideoCaptioner(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    processed = 0
    success = 0
    failed = 0
    for video_item in video_data_list:
        original_video_path = video_item.get("original_video_path", "")
        current_md5 = get_md5(original_video_path) if original_video_path else None
        
        # 续跑逻辑检查
        if current_md5 and current_md5 in processed_md5s:
            processed += 1
            success += 1
            if processed % 10 == 0 or processed == total:
                print(f"Rank {rank}: {processed}/{total} processed (success={success}, failed={failed})")
            continue
            
        try:
            payload = infer_video_chunks(
                captioner=captioner,
                video_item=video_item,
                audio_dir=args.audio_dir,
                rng=rng,
                args=args,
            )
        except Exception as exc:
            processed += 1
            failed += 1
            print(f"Rank {rank}: Error processing MD5 {current_md5}: {exc}")
            continue

        write_jsonl(output_path, payload)
        processed += 1
        success += 1
        if processed % 10 == 0 or processed == total:
            print(
                f"Rank {rank}: {processed}/{total} processed "
                f"(success={success}, failed={failed})"
            )

if __name__ == "__main__":
    main()