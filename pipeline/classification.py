import os

# --- 环境变量设置 ---
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# os.environ["RAY_DISABLE_DASHBOARD"] = "1" 

import json
import random
import datetime
import os.path as osp
import re
import hashlib
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- MPI 环境获取 ---
def get_mpi_info():
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
    elif 'RANK' in os.environ:
        rank = int(os.environ.get('RANK'))
        size = int(os.environ.get('WORLD_SIZE'))
    elif 'MV2_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ.get('MV2_COMM_WORLD_RANK'))
        size = int(os.environ.get('MV2_COMM_WORLD_SIZE'))
    else:
        rank = 0
        size = 1
    return rank, size

# ==========================================
# 核心修改 1: 替换为分类 Prompt
# 注意：我们将 {question} 和 {answer} 留作 Python format 的占位符
# ==========================================
CLASSIFICATION_PROMPT_TEMPLATE = """You are an expert evaluator and classifier specializing in Multimodal Large Language Models (MLLMs) for Audio-Visual Question Answering (AVQA) tasks.

Your objective is to perform TWO tasks based on the provided inputs:
1. Objectively evaluate the quality of the model's Chain-of-Thought (CoT) and final answer.
2. Classify the user's question into one specific AVQA category AND determine its primary modality dependency.

### 📥 INPUT DATA
- [Standard AV Caption]: {AV_caption}
- [Question]: {question}
- [Ground Truth Answer]: {answer}
- [Model Output (CoT + Answer)]: {model_output}

---

### 📋 TASK 1: EVALUATION (1-5 Scale)
Evaluate the [Model Output] across the following 6 dimensions. 

1. Context Utilization & Relevance (Score: 1-5)
   - Goal: Does the CoT effectively and appropriately use the necessary modality (Visual, Audio, or Joint AV) required to answer the specific question?
   - 5: Perfect utilization. The CoT accurately extracts and relies strictly on the necessary information (whether pure video, pure audio, or a combination) to answer the question, without forcing irrelevant context.
   - 3: Suboptimal utilization. Uses the correct modality but misses some key details, OR unnecessarily includes irrelevant information from another modality that does not contribute to the reasoning (e.g., mentioning background noise for a purely visual question).
   - 1: Poor utilization. Fails to use the provided context, ignores the clearly required modality for the question, or relies purely on general knowledge.

2. Hallucination Absence (Score: 1-5) 
   - Goal: Are the scenes, actions, or sounds described in the CoT strictly based on the [Standard AV Caption]? (Higher score = fewer hallucinations).
   - 5: No hallucination. All details are perfectly grounded in the provided context.
   - 3: Minor hallucination. Contains slight deviations or fabricated minor details that do not disrupt the core reasoning.
   - 1: Severe hallucination. Fabricates crucial visual/audio objects, leading to an entirely ungrounded reasoning basis.

3. Logical Reasoning (Score: 1-5)
   - Goal: Is the CoT rigorous, structured, and causally sound?
   - 5: Extremely rigorous. Step-by-step deduction, clear causality, and sufficient evidence supporting the conclusion.
   - 3: Acceptable. Generally reasonable, but contains minor logical leaps or disjointed expressions.
   - 1: Chaotic. Contradictory, completely illogical, or lacks reasoning entirely (jumps straight to the answer).

4. Answer Accuracy (Score: 1-5)
   - Goal: Does the final inferred answer match the [Ground Truth Answer]?
   - 5: Perfectly accurate and semantically identical.
   - 3: Partially correct, or contains the correct answer mixed with misleading/ambiguous extra information.
   - 1: Completely incorrect.

5. Question Difficulty (Score: 1-5)
   - Goal: How inherently difficult and cognitively demanding is the [Question] based on the [Standard AV Caption]?
   - 5: Highly complex. Requires multi-step reasoning, deep temporal/causal understanding, or nuanced integration of subtle audio-visual cues that are not explicitly linked in the text.
   - 3: Moderate. Requires basic cross-modal alignment or moderately complex single-modality retrieval (e.g., matching a specific sound to a specific visible action).
   - 1: Very simple. A direct, shallow factual lookup (e.g., identifying a clearly stated object color or explicit background noise) with zero deductive effort required.

6. Observation Grounding & Deduction (Score: 1-5)
   - Goal: Does the model accurately reference the [Standard AV Caption] (e.g., within an observation block) AND perform genuine logical deduction based on it, rather than just parroting the text?
   - 5: Excellent deduction. The model accurately quotes/extracts relevant facts from the caption AND uses them as a springboard for genuine, step-by-step logical inference to reach the answer.
   - 3: Shallow deduction / Parroting. The model accurately cites the observation, but the subsequent reasoning is practically non-existent or merely rewrites/repeats the observation before jumping to the conclusion.
   - 1: Unanchored or entirely absent. The model fails to accurately reference the caption, hallucinates observations, or provides zero connection between the stated observation and the final answer.

---

### 🗂️ TASK 2: QUESTION CLASSIFICATION & MODALITY
First, analyze the [Question] and [Ground Truth Answer], and classify the question into EXACTLY ONE of the following 12 categories (Choose the most primary one):

1. Audio-Visual Joint Perception
2. Pure Visual Perception
3. Pure Audio Perception
4. Action & Behavior Recognition
5. Spatial Layout Understanding
6. Temporal Sequence Understanding
7. Attribute Comparison & Change
8. Counting & Quantification
9. Emotion & Atmosphere Perception
10. Semantic Content Summarization
11. Logical Relation Reasoning
12. Intention & Outcome Prediction

Second, determine the primary modality dependency of the question. Choose EXACTLY ONE:
- "AV-Strong": Answering requires logically combining both visual and auditory cues.
- "Video-Strong": Answering relies primarily on visual information.
- "Audio-Strong": Answering relies primarily on auditory information.

---

### 📤 OUTPUT FORMAT (STRICT JSON)
You must output ONLY a valid JSON object combining both tasks. Do not include markdown code blocks (e.g., ```json), conversational text, or any explanations outside the JSON structure. Use the exact keys below:

{{
  "evaluation": {{
    "context_utilization": <int, 1-5>,
    "hallucination_absence": <int, 1-5>,
    "logical_reasoning": <int, 1-5>,
    "answer_accuracy": <int, 1-5>,
    "question_difficulty": <int, 1-5>,
    "observation_deduction": <int, 1-5>
  }},
  "classification": {{
    "category_id": <int, 1-12>,
    "category_name": "<string, exact name from the list>",
    "modality_dependency": "<string, 'AV-Strong' | 'Video-Strong' | 'Audio-Strong'>",
    "confidence": <float, 0.0-1.0>,
    "reasoning": "<string, short 1-2 sentence explanation for the chosen category and modality>"
  }}
}}
"""

# ==========================================
# 提取 JSON 的逻辑 (保持不变，很通用)
# ==========================================
def extract_items(output_text):
    if not output_text:
        return None
    
    text = output_text.strip()
    # 策略 A: 提取 markdown 代码块
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    json_str = ""
    if match:
        json_str = match.group(1)
    else:
        # 策略 B: 寻找最外层的 { }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx : end_idx + 1]
        else:
            return None

    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
        return None
    except json.JSONDecodeError:
        return None

MODEL_PATH = "path/to/model"  

def init_llm():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=0.3, # 分类任务建议降低 temperature
        top_p=0.95,
        top_k=20,
        max_tokens=1024 # 分类输出很短，不需要很长
    )
    return llm, sampling_params

# ==========================================
# 核心修改 2: 修改构建 Prompt 的逻辑
# 使用 data['question'] 和 data['cot']
# ==========================================
def build_prompt(data) -> str:
    question = data.get('question', '')
    # 这里将 jsonl 里的 cot 字段作为 answer 传入
    cot_content = data.get('cot', '') 
    answer = data.get('answer', '') 
    caption = data.get('messages', [{}])[1].get('content', '')
    # 填充 Prompt 模板
    user_content = CLASSIFICATION_PROMPT_TEMPLATE.format(
        question=question,
        answer = answer,
        model_output=cot_content,
        AV_caption = caption
    )

    # 构造对话格式
    return f"""<|system|>
You are a helpful assistant.
<|user|>
{user_content}
<|assistant|>
```json
"""

def generate_batch(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False) 
    return [o.outputs[0].text.strip() for o in outputs]

def get_unique_id(data):
    # 依然使用 vid + question 做唯一标识
    vid = str(data.get('original_video_path', ''))
    question = str(data.get('question', ''))
    raw_str = f"{vid}_||_{question}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def get_existing_ids(save_path):
    existing_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    existing_ids.add(get_unique_id(data))
                except:
                    pass
    print(f"[Node {rank}] Found {len(existing_ids)} existing samples.")
    return existing_ids

def data_generator_sharded(data_path, existing_ids, rank, world_size):
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % world_size != rank:
                continue

            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            uid = get_unique_id(data)
            if uid in existing_ids:
                continue
            
            # 检查必要字段，现在我们需要 cot
            if 'cot' not in data and 'answer' not in data:
                # 兼容一下，如果没有 cot，看看有没有 answer，或者直接跳过
                continue
                
            yield data

if __name__ == "__main__":
    rank, world_size = get_mpi_info()
    
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # ==========================================
    # 注意：这里 DATA_PATH 应该改为你上一步生成的结果文件
    # ==========================================
    DATA_PATH = "path/to/data.jsonl" 
    # ^^^ 请确保这里指向的是包含 'cot' 字段的文件 ^^^

    SAVE_DIR = "path/to/output"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 修改输出文件名，避免覆盖
    save_path = osp.join(SAVE_DIR, f"glm_classification_{rank}.jsonl") 

    BATCH_SIZE = 32 # 分类任务比较短，Batch 可以稍微大一点

    existing_ids = get_existing_ids(save_path) 
    
    llm, sampling_params = init_llm()
    
    # 注意：如果 DATA_PATH 是多个文件，这里可能需要逻辑调整，或者手动合并成一个大文件再跑
    if not os.path.exists(DATA_PATH):
        print(f"[Error] Data path not found: {DATA_PATH}")
        # 如果是分片读取之前生成的文件，可能需要改一下读取逻辑
        # 这里假设你已经把之前生成的 cot jsonl 合并成了一个大文件，或者直接读对应的 part 文件
    
    pending_generator = data_generator_sharded(DATA_PATH, existing_ids, rank, world_size)
    
    prompt_batch = []
    data_batch = []
    total_new_processed = 0

    print(f"[Node {rank}] Start Classification. Output: {save_path}")

    with open(save_path, 'a', encoding='utf-8') as f_w: 
        disable_tqdm = (rank != 0)
        
        with tqdm(desc=f"Node {rank} Classifying", disable=disable_tqdm, unit="sample") as pbar:
            
            for data in pending_generator:
                
                # 构建 Prompt
                prompt = build_prompt(data)
                prompt_batch.append(prompt)
                data_batch.append(data)

                if len(prompt_batch) >= BATCH_SIZE:
                    texts = generate_batch(llm, prompt_batch, sampling_params)

                    for text, raw_data in zip(texts, data_batch):
                        # 解析 JSON
                        extracted_json = extract_items(text)
                        
                        if extracted_json:
                            out = raw_data.copy()
                            # 将 category_id, category_name, reasoning 等合并进去
                            out.update(extracted_json)
                            
                            f_w.write(json.dumps(out, ensure_ascii=False) + "\n")
                            total_new_processed += 1
                            if rank == 0: pbar.update(1)

                    f_w.flush() 
                    prompt_batch = []
                    data_batch = []

            # Last Batch
            if len(prompt_batch) > 0:
                texts = generate_batch(llm, prompt_batch, sampling_params)
                for text, raw_data in zip(texts, data_batch):
                    extracted_json = extract_items(text)
                    if extracted_json:
                        out = raw_data.copy()
                        out.update(extracted_json)
                        f_w.write(json.dumps(out, ensure_ascii=False) + "\n")
                        total_new_processed += 1
                        if rank == 0: pbar.update(1)
                f_w.flush()

    print(f"[Node {rank}] Done! Total classified: {total_new_processed}")