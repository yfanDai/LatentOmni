# import os
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# os.environ["VIDEO_TOTAL_PIXELS"] = "39200000"
# import json
# import torch
# import numpy as np
# import librosa
# import av
# from typing import Any, Callable, Optional, Union, List, Tuple, Dict
# import torchaudio
# from decord import VideoReader, cpu
# from vllm import LLM, SamplingParams
# from qwen_omni_utils import process_vision_info
# from qwen_omni_utils import process_audio_info
# from transformers import AutoTokenizer,Qwen3OmniMoeProcessor
# import gc
# from tqdm import tqdm



# # --- 配置区 ---

# MODEL_PATH = "path/to/model"
# DATA_PATH = "path/to/data.jsonl"
# OUTPUT_PATH = "path/to/output.jsonl"

# Audio_PROMPT = """
# Listen the audio carefully. You MUST list all **AUDIO** events related to the QUESTION and ANSWER. For each event, you MUST only provide the precise start and end timestamps and a brief description (end timestamps MUST be smaller than Full video Duration). Do not answer the question yet. Follow below examples:\n
# audio: The voice of bird: <Unified_Latent>[00:12-00:15]</Unified_Latent>\n
# """

# Visual_PROMPT = """
# Watch the Video Carefully. You MUST list all **VISUAL** events related to the QUESTION and ANSWER. For each event, you MUST only provide the **PRECISE** start and end timestamps and a brief description (end timestamps MUST be smaller than Full video Duration). Do not answer the question yet. Follow below example (MUST Not copy example):\n
# visual: A man shoot at birds: <Unified_Latent>[00:02-00:05]</Unified_Latent>\n
# """


# SAMPLING_RATE = 16000
# BATCH_SIZE = 128

# # --- 数据预处理函数 ---

# def load_audio(audio_path, sr=SAMPLING_RATE):
#     try:
#         audio_data, _ = librosa.load(audio_path, sr=sr)
#         return audio_data
#     except Exception as e:
#         return None

# def get_unique_key(item):
#     if 'id' in item:
#         return str(item['id'])
#     return f"{item.get('original_video_path', '')}_{item.get('question', '')}"

# def batch_generator(data_list, batch_size):
#     for i in range(0, len(data_list), batch_size):
#         yield data_list[i : i + batch_size]


# def run_inference():
#     print(f"Loading input data from {DATA_PATH}...")
#     with open(DATA_PATH, 'r') as f:
#         raw_items = [json.loads(line) for line in f]
    
#     total_raw_count = len(raw_items)
#     print(f"Total raw items: {total_raw_count}")

#     finished_keys = set()
#     if os.path.exists(OUTPUT_PATH):
#         print(f"Scanning existing output file {OUTPUT_PATH} for resume...")
#         with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line: continue
#                 try:
                
#                     item = json.loads(line)
#                     key = get_unique_key(item)
#                     finished_keys.add(key)
#                 except json.JSONDecodeError:
#                     continue
    
#     print(f"Found {len(finished_keys)} items already processed.")

#     pending_items = []
#     for item in raw_items:
#         key = get_unique_key(item)
#         if key not in finished_keys:
#             pending_items.append(item)
    
#     print(f"Items remaining to process: {len(pending_items)}")

#     if len(pending_items) == 0:
#         print("All items processed! Exiting.")
#         return

#     print("Initializing LLM...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
#     llm = LLM(
#         model=MODEL_PATH,
#         tensor_parallel_size=1,
#         pipeline_parallel_size=1, 
#         trust_remote_code=True,
#         gpu_memory_utilization=0.85,
#         max_model_len=65535,
#         limit_mm_per_prompt={"video": 1, "audio": 1}
#     )
#     sampling_params = SamplingParams(
#         temperature=0.6,
#         top_p=0.95,
#         top_k=20,
#         max_tokens=4096,
#     )

#     output_file = open(OUTPUT_PATH, 'a', encoding='utf-8')

#     # 外层循环：按批次读取数据
#     for batch_items in tqdm(batch_generator(pending_items, BATCH_SIZE), total=(len(pending_items) + BATCH_SIZE - 1)//BATCH_SIZE):
        
#         batch_inputs = []
#         batch_metadata = [] 

#         # 1. 预处理循环：把当前批次的数据转换成 LLM 能识别的格式
#         for item in batch_items:
#             # item 是字典，Python中字典是引用传递，修改 item_result 里的 item 也就是修改了 batch_items 里的 item
#             item_result = {
#                 "item": item, 
#                 "video_only": "N/A",
#                 "audio_only": "N/A"
#             }
            
#             video_path = item.get('original_video_path')
#             audio_list = item.get('audio_in_video', [])
#             audio_path = audio_list[0] if audio_list else None
#             question = item.get('question', '')
#             answer = item.get('answer', '')
#             all_time = item.get('video_duration','')
#             context = f"**Full video Duration:** {all_time}\n **QUESTION:** {question}\n **ANSWER:** {answer}"

#             # --- 视频任务 ---
#             if video_path is not None and os.path.exists(video_path) and not os.path.isdir(video_path):
#                 messages_v = [
#                     {"role": "system", "content": Visual_PROMPT},
#                     {"role": "user", "content": [
#                         {"type": "video", "video": video_path},
#                         {"type": "text", "text": context}
#                     ]}
#                 ]
#                 prompt_v = processor.apply_chat_template(messages_v, tokenize=False, add_generation_prompt=True)
#                 i_data, v_data = process_vision_info(messages_v)
                
#                 # 添加到批量输入列表
#                 batch_inputs.append({
#                     "prompt": prompt_v,
#                     "multi_modal_data": {"video": v_data}
#                 })
#                 # 记录这个输入对应哪个 item，以及是视频任务
#                 batch_metadata.append({"result_obj": item_result, "type": "video_only"})
            
#             # --- 音频任务 ---
#             if audio_path:
#                 messages_a = [
#                     {"role": "system", "content": Audio_PROMPT},
#                     {"role": "user", "content": [
#                         {"type": "audio", "audio": audio_path},
#                         {"type": "text", "text": context}
#                     ]}
#                 ]
#                 a_data = process_audio_info(messages_a,use_audio_in_video = False)
#                 prompt_a = processor.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
                
#                 batch_inputs.append({
#                     "prompt": prompt_a,
#                     "multi_modal_data": {"audio": a_data}
#                 })
#                 # 记录这个输入对应哪个 item，以及是音频任务
#                 batch_metadata.append({"result_obj": item_result, "type": "audio_only"})

#         # 2. 推理阶段：如果有数据，就进行批量推理
#         if batch_inputs:
#             outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)

#             # 3. 回填阶段：把结果填回原始字典
#             for i, out in enumerate(outputs):
#                 meta = batch_metadata[i]
#                 res_obj = meta["result_obj"] # 这里拿到的引用，指向同一个 item_result
#                 task_type = meta["type"]
#                 generated_text = out.outputs[0].text.strip()
                
#                 # 直接修改原始 item 字典的内容
#                 if task_type == "video_only":
#                     res_obj["item"]["video_stamp"] = generated_text
#                 elif task_type == "audio_only":
#                     res_obj["item"]["audio_stamp"] = generated_text

#             # 4. 写入阶段（修正点）：必须遍历当前批次的所有 item 进行写入
#             for item in batch_items:
#                 # 此时 item 已经被上面的回填逻辑加上了 video_stamp 或 audio_stamp
#                 output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        
#         # 如果 batch_inputs 为空（例如这批数据既没视频也没音频），也应该把原始数据写回去（看你需求）
#         # 如果你的数据保证一定有视频或音频，上面的 if batch_inputs 逻辑就够了。
#         # 如果可能出现空数据，建议把写入逻辑移到 if batch_inputs 块的外面。
        
#         output_file.flush() 

#         # 清理内存
#         if batch_inputs: del outputs
#         del batch_inputs
#         del batch_metadata
#         gc.collect() 

#     output_file.close()
#     print("Batch inference process finished successfully.")

# if __name__ == "__main__":
#     run_inference()


import os
import sys

# --- [新增] MPI/多进程环境初始化 ---
# 必须在导入 torch/vllm 之前设置 CUDA_VISIBLE_DEVICES
# OpenMPI 通常设置 OMPI_COMM_WORLD_LOCAL_RANK，Torchrun 设置 LOCAL_RANK
def setup_distributed_env():
    # 尝试获取 Rank 信息
    if "OMPI_COMM_WORLD_RANK" in os.environ: # mpirun (OpenMPI)
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif "RANK" in os.environ: # torchrun / general
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # 非分布式环境，默认单卡
        print("Running in non-distributed mode.")
        return 0, 1, 0

    # 关键：强制当前进程只可见分配给它的那张卡
    # 这样 vLLM 初始化时会认为这台机器只有这一张卡，避免多实例资源竞争
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print(f"Rank {rank}/{world_size} (Local {local_rank}) initialized. Visible GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    return rank, world_size, local_rank

RANK, WORLD_SIZE, LOCAL_RANK = setup_distributed_env()

# --- 原有环境变量设置 ---
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# os.environ["VIDEO_TOTAL_PIXELS"] = "1568000"
# 建议关闭 Ray 的 dashboard 以减少端口冲突风险
os.environ["RAY_DISABLE_DASHBOARD"] = "1" 

import json
import torch
import numpy as np
import librosa
import av
from typing import Any, Callable, Optional, Union, List, Tuple, Dict
import torchaudio
from decord import VideoReader, cpu
from vllm import LLM, SamplingParams
# 假设这些 utils 在当前路径下可用
from omni_utils import process_vision_info
from omni_utils import process_audio_info
from transformers import AutoTokenizer, Qwen3OmniMoeProcessor
import gc
from tqdm import tqdm

# --- 配置区 ---

MODEL_PATH = "path/to/model"
DATA_PATH = "path/to/data.jsonl"
# 修改输出路径逻辑，基础路径
BASE_OUTPUT_PATH = "path/to/output"

Audio_PROMPT = """
Listen the audio carefully. You MUST list all **AUDIO** events related to the QUESTION and ANSWER. For each event, you MUST only provide the precise start and end timestamps and a brief description (end timestamps MUST be smaller than Full video Duration). Do not answer the question yet. Follow below examples:\n
audio: The voice of bird: <Unified_Latent>[00:12-00:15]</Unified_Latent>\n
"""

Visual_PROMPT = """
Watch the Video Carefully. You MUST list all **VISUAL** events related to the QUESTION and ANSWER. For each event, you MUST only provide the **PRECISE** start and end timestamps and a brief description (end timestamps MUST be smaller than Full video Duration). Do not answer the question yet. Follow below example (MUST Not copy example):\n
visual: A man shoot at birds: <Unified_Latent>[00:02-00:05]</Unified_Latent>\n
"""

SAMPLING_RATE = 16000
BATCH_SIZE = 32

# --- 数据预处理函数 ---

def load_audio(audio_path, sr=SAMPLING_RATE):
    try:
        audio_data, _ = librosa.load(audio_path, sr=sr)
        return audio_data
    except Exception as e:
        return None

def get_unique_key(item):
    if 'id' in item:
        return str(item['id'])
    return f"{item.get('original_video_path', '')}_{item.get('question', '')}"

def batch_generator(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i : i + batch_size]

def run_inference():
    # 生成当前 Rank 专属的输出文件路径
    # 例如: .../stamp_rank_0.jsonl
    rank_output_path = f"{BASE_OUTPUT_PATH}_rank_{RANK}.jsonl"
    
    if RANK == 0:
        print(f"Loading input data from {DATA_PATH}...")
    
    # 所有进程都读取完整数据（假设内存足够，JSONL通常没问题）
    with open(DATA_PATH, 'r') as f:
        all_raw_items = [json.loads(line) for line in f]
    
    # 扫描当前 Rank 的输出文件，用于断点续传
    finished_keys = set()
    if os.path.exists(rank_output_path):
        print(f"[Rank {RANK}] Scanning existing output file {rank_output_path} for resume...")
        with open(rank_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    key = get_unique_key(item)
                    finished_keys.add(key)
                except json.JSONDecodeError:
                    continue
    
    # --- [关键] 数据分片 ---
    # 使用切片操作：list[start:stop:step]
    # 例如 8卡: Rank 0 取索引 0, 8, 16... Rank 1 取索引 1, 9, 17...
    my_shard_items = all_raw_items[RANK::WORLD_SIZE]
    
    # 过滤已完成的
    pending_items = []
    for item in my_shard_items:
        key = get_unique_key(item)
        if key not in finished_keys:
            pending_items.append(item)
            
    print(f"[Rank {RANK}] Total Assigned: {len(my_shard_items)}, Already Done: {len(finished_keys)}, Pending: {len(pending_items)}")

    if len(pending_items) == 0:
        print(f"[Rank {RANK}] All items processed! Exiting.")
        return

    # 初始化 LLM
    # 注意：由于我们在开头设置了 CUDA_VISIBLE_DEVICES，这里 vLLM 只看得到一张卡
    # 所以不需要指定 devices 参数，它会自动用 device 0 (即物理上的 local_rank)
    print(f"[Rank {RANK}] Initializing LLM...")
    
    # 确保 tokenizer 加载一次
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1, # 保持为1，因为我们在做数据并行
        pipeline_parallel_size=1, 
        trust_remote_code=True,
        gpu_memory_utilization=0.90, # 稍微调高一点，因为是单卡独占
        max_model_len=65535, # 适当调整，太大会OOM，根据你的VRAM大小设定
        limit_mm_per_prompt={"video": 1, "audio": 1},
        distributed_executor_backend="mp", # 推荐：在某些环境中 mp 比 ray 更稳定，或者保持默认
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=4096, # 适当调整
    )

    output_file = open(rank_output_path, 'a', encoding='utf-8')

    # 使用 position 参数让不同 rank 的 tqdm 进度条不重叠（可选，但推荐）
    # total 计算要准确
    pbar = tqdm(batch_generator(pending_items, BATCH_SIZE), 
                total=(len(pending_items) + BATCH_SIZE - 1)//BATCH_SIZE,
                desc=f"Rank {RANK}",
                position=RANK)

    for batch_items in pbar:
        batch_inputs = []
        batch_metadata = [] 

        # 1. 预处理
        for item in batch_items:
            item_result = {
                "item": item, 
                "video_only": "N/A",
                "audio_only": "N/A"
            }
            
            video_path = item.get('original_video_path')
            audio_list = item.get('audio_in_video', [])
            audio_path = audio_list[0] if audio_list else None
            question = item.get('question', '')
            answer = item.get('answer', '')
            all_time = item.get('video_duration','')
            context = f"**Full video Duration:** {all_time}\n **QUESTION:** {question}\n **ANSWER:** {answer}"

            # --- 视频任务 ---
            if video_path is not None and os.path.exists(video_path) and not os.path.isdir(video_path):
                messages_v = [
                    {"role": "system", "content": Visual_PROMPT},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": context}
                    ]}
                ]
                # 注意：确保 processor 在此处调用也是线程安全的/进程独立的
                prompt_v = processor.apply_chat_template(messages_v, tokenize=False, add_generation_prompt=True)
                i_data, v_data = process_vision_info(messages_v)
                
                batch_inputs.append({
                    "prompt": prompt_v,
                    "multi_modal_data": {"video": v_data}
                })
                batch_metadata.append({"result_obj": item_result, "type": "video_only"})
            
            # --- 音频任务 ---
            if audio_path:
                messages_a = [
                    {"role": "system", "content": Audio_PROMPT},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": context}
                    ]}
                ]
                a_data = process_audio_info(messages_a, use_audio_in_video=False)
                prompt_a = processor.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
                
                batch_inputs.append({
                    "prompt": prompt_a,
                    "multi_modal_data": {"audio": a_data}
                })
                batch_metadata.append({"result_obj": item_result, "type": "audio_only"})

        # 2. 推理
        if batch_inputs:
            # use_tqdm=False 避免每个 batch 内部再刷屏
            outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)

            # 3. 回填
            for i, out in enumerate(outputs):
                meta = batch_metadata[i]
                res_obj = meta["result_obj"] 
                task_type = meta["type"]
                generated_text = out.outputs[0].text.strip()
                
                if task_type == "video_only":
                    res_obj["item"]["video_stamp"] = generated_text
                elif task_type == "audio_only":
                    res_obj["item"]["audio_stamp"] = generated_text

            # 4. 写入
            for item in batch_items:
                output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        output_file.flush() 
        
        # 显式清理
        if batch_inputs: del outputs
        del batch_inputs
        del batch_metadata
        gc.collect() 

    output_file.close()
    print(f"[Rank {RANK}] Finished.")

if __name__ == "__main__":
    run_inference()