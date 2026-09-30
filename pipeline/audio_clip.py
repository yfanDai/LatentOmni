#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大规模 JSONL 音频并发切割脚本。
基于 FFmpeg 的 segment 模块，直接在底层进行流切割，内存占用极小。
"""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_md5(text: str) -> str:
    """计算字符串的 MD5，用于统一命名"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def process_single_audio(line_str: str, output_dir: str) -> tuple[bool, str]:
    """
    处理单行 JSONL：提取音频、计算 MD5，调用 FFmpeg 切割。
    """
    try:
        data = json.loads(line_str)
        original_video_path = data.get("original_video_path")
        audio_in_video = data.get("audio_in_video")

        if not original_video_path or not audio_in_video:
            return False, "Skipped: 缺少 original_video_path 或 audio_in_video 字段"
        
        # 提取第一个元素
        input_audio_path = audio_in_video[0]
        
        if not os.path.exists(input_audio_path):
            return False, f"Skipped: 文件不存在 -> {input_audio_path}"

        md5_digest = get_md5(original_video_path)
        
        # 构建输出的正则路径，例如: /output/dir/abcdef12345_%d.wav
        # %d 会被 FFmpeg 自动替换为 1, 2, 3...
        output_pattern = os.path.join(output_dir, f"{md5_digest}_%d.wav")
        
        # 使用 FFmpeg 的 segment 模块进行高效一刀切
        # -f segment: 启用分片模式
        # -segment_time 5: 每 5 秒切一刀
        # -segment_start_number 1: 序号从 1 开始 (即 _1, _2, _3...)
        # -c:a pcm_s16le: 统一重采样为标准 16-bit WAV (确保模型读取不报错)
        cmd = [
            "ffmpeg", 
            "-y",               # 覆盖输出
            "-v", "error",      # 屏蔽冗余输出，只报错误
            "-i", input_audio_path,
            "-f", "segment",
            "-segment_time", "5",
            "-segment_start_number", "1",
            "-c:a", "pcm_s16le", 
            output_pattern
        ]
        clean_env = os.environ.copy()
        clean_env.pop("LD_LIBRARY_PATH", None)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,env = clean_env)
        return True, md5_digest

    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg Error on {input_audio_path}: {e.stderr.decode('utf-8')}"
    except Exception as e:
        return False, f"Error parsing/processing line: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="并发批量按 5s 顺序切割音频")
    parser.add_argument("--input-jsonl", required=True, help="输入的大 JSONL 文件路径")
    parser.add_argument("--output-dir", required=True, help="切割后的音频保存目录")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 采用生成器逐行读取，避免大文件一次性撑爆内存
    def line_generator():
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    total_submitted = 0
    success_count = 0
    failed_count = 0

    print(f"开始处理，启动 {args.workers} 个并发线程...")
    
    # 使用 ThreadPoolExecutor，因为 FFmpeg 外部调用属于 I/O 密集型操作
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_audio, line, args.output_dir): line
            for line in line_generator()
        }
        
        total_tasks = len(futures)
        print(f"共提交了 {total_tasks} 个切割任务。")

        # 收集结果
        for future in as_completed(futures):
            total_submitted += 1
            is_success, msg = future.result()
            
            if is_success:
                success_count += 1
            else:
                failed_count += 1
                # 打印错误信息
                print(msg)
                
            # 每 100 个打印一次进度
            if total_submitted % 100 == 0 or total_submitted == total_tasks:
                print(f"进度: {total_submitted}/{total_tasks} | 成功: {success_count} | 失败: {failed_count}")

    print("✅ 所有音频切割处理完毕！")

if __name__ == "__main__":
    main()