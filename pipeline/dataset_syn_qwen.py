import os
import json
import random
import datetime
import os.path as osp
import re
from tqdm import tqdm

from vllm import LLM, SamplingParams

PROMPT_QA = """
You are an expert multimodal dataset designer for Audio-Visual Question Answering (AVQA).

Input: a single audio-video caption (short paragraph) describing what is seen and heard.

Task:
1. Generate Three high-quality open-ended question-answer pairs.
2. For AV questions: it MUST be **IMPOSSIBLE** to answer from only audio or only video.
3. Questions should vary in reasoning type: causal reasoning, spatial relations, temporal sequencing, counting, sound-action attribution, object interactions, and detail comprehension etc...
4. The designed questions must involve relatively complex reasoning.

Hard Constraints:
- Questions must be concrete, specific, and grounded strictly in the caption.
- Questions must rely on both **VISUAL and AUDIO** information
- Each question must be answerable using only the provided caption.
- Answers must be concise (max 10 words) and strictly supported by caption content.
- Avoid object IDs, labels, timestamps, or XML tags.
- Ensure world/common-sense reasoning when appropriate (e.g., linking visible actions with expected sounds).

Additional Guidelines for Real world scene:
- Encourage questions requiring commonsense integration (e.g., physical causality, typical daily routines).
- Include reasoning about interactions between agents, objects, and sounds.
- **Avoid overly simplistic factual** questions that do not require cross-modal integration.

Output format:
Output must be raw JSON only. Do not wrap in markdown.
Example (**Just for reference**):
{
  "question_type": "QA",
  "items": [
{
  "question_type": "QA",
  "items": [
        {
        "id": "QA1",
        "modality": "AV",
        "question": "As the small black and white puppy tilts its head back and opens its mouth wide, what on-screen text appears and what specific vocalization is heard?",
        "answer": "嗷~嗷~ and high-pitched raspy yips"
        },
        {
        "id": "QA2",
        "modality": "AV",
        "question": "When the light grey puppy howls while looking up at the person leaning down, where is the person positioned and what sound comes next?",
        "answer": "Leaning down from left; slightly whiny howl"
        },
        {
        "id": "QA3",
        "modality": "AV",
        "question": "How many times does the pink on-screen text appear specifically to represent a puppy's howl voice during the video?",
        "answer": "Three times"
        }
    ]
    }
  ]
}

"""

PROMPT_MCQ = """
You are an expert multimodal dataset designer for Audio-Visual Question Answering (AVQA).

Input: a single audio-video caption (short paragraph) describing what is seen and heard.

Task:
1. Generate Three high-quality open-ended question-answer pairs.
2. For AV questions: it MUST be **IMPOSSIBLE** to answer from only audio or only video.
3. Questions should vary in reasoning type: causal reasoning, spatial relations, temporal sequencing, counting, sound-action attribution, object interactions, and detail comprehension etc...
4. The designed questions must involve relatively complex reasoning.

Hard Constraints:
- Each MCQ must have exactly one correct answer and three plausible distractors.
- - Questions must rely on both **VISUAL and AUDIO** information
- Options must be clearly distinct, plausible, and grounded in caption content.
- The incorrect options shouldn't be too obvious; they need to be misleading.
- Avoid subjective, speculative, or ambiguous distractors.
- Avoid object IDs, labels, timestamps, or XML tags.
- Correct answers must be strictly supported by the caption.

Additional Guidelines for Real world scene:
- Ensure distractors reflect reasonable alternative interpretations based on world/common sense (e.g., typical object usage, expected sequences of actions).
- Prioritize questions that test multimodal reasoning beyond simple visual counting or audio identification.
- Include temporal ordering and cause-effect option patterns when feasible.
- **Avoid overly simplistic factual** questions that do not require cross-modal integration.


Output format:
Output must be raw JSON only. Do not wrap in markdown.
Example:
{
  "question_type": "MCQ",
    "items": [{"id": "Q1", "modality": "AV", "question": "Based on the synchronized audio and visual cues, how does the daughter's behavior change when she begins her improvised song?", "options": ["A. She puts down her ice cream cone to clap her hands in time with the R&B beat.", "B. She remains still and focused while singing her lyrics about 'no problems at all'.", "C. She becomes more animated, pointing at the camera and gesturing while holding her cone.", "D. She stops gesturing to focus on harmonizing her baritone voice with her father's singing."], "answer": "C", "answer_text": "C. She becomes more animated, pointing at the camera and gesturing while holding her cone."}, {"id": "Q2", "modality": "AV", "question": "What specific action do the father and daughter perform together to visually celebrate the lyrics they are singing about their relationship?", "output": "A. They both make peace signs while the father sings 'Father, daughter, music, music'.", "options": ["A. They both make peace signs while the father sings 'Father, daughter, music, music'.", "B. They hold up their matching vanilla ice cream cones in a 'cheers' gesture toward the camera.", "C. The father hands his ice cream cone to the daughter so she can hold both while singing.", "D. They both point at the sunlit window behind them to emphasize the 'no problems' lyric."], "answer": "B", "answer_text": "B. They hold up their matching vanilla ice cream cones in a 'cheers' gesture toward the camera."}, {"id": "Q3", "modality": "AV", "question": "How does the father's participation in the scene evolve from the beginning of the audio track to the end of the video?", "options": ["A. He starts by singing a baritone solo and ends by silently eating his ice cream cone.", "B. He begins with a neutral expression and silent head-bobbing, then later joins in singing and smiling.", "C. He starts by holding the camera steady and ends by putting it down to gesture with both hands.", "D. He begins by encouraging his daughter verbally and ends by taking her ice cream cone away."], "answer": "B", "answer_text": "B. He begins with a neutral expression and silent head-bobbing, then later joins in singing and smiling."}]
}
"""

PROMPT_LIST = [PROMPT_QA, PROMPT_MCQ]


def choose_prompt_index():
    """50% QA, 50% MCQ"""
    return 0 if random.random() < 0.5 else 1



def extract_first_json(text: str):
    start = text.find("{")
    if start == -1:
        return None

    stack = []
    for i in range(start, len(text)):
        if text[i] == "{":
            stack.append("{")
        elif text[i] == "}":
            stack.pop()
            if not stack:
                json_str = text[start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


def extract_items(output_text):
    data = extract_first_json(output_text)
    if not data:
        return None

    if "question_type" not in data or "items" not in data:
        return None

    items = data["items"]
    if not items:
        return None

    if data["question_type"] == "MCQ":
        for item in items:
            options = item.get("options", [])
            idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(item.get("answer"))
            item["answer_text"] = options[idx] if idx is not None and idx < len(options) else None

    return {
        "question_type": data["question_type"],
        "items": items,
    }


MODEL_PATH = "path/to/model"  

def init_llm():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        enable_prefix_caching=True,
        max_num_seqs=256,  # 增大最大并发序列数
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=8192
    )
    return llm, sampling_params



def build_prompt(task_prompt: str, caption: str) -> str:
    return f"""<|system|>
{task_prompt}
<|user|>

Input caption:
{caption}
<|assistant|>
"""

def generate_batch(llm, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def iter_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_existing_ids(save_path):
    existing_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'original_video_path' in data:
                        existing_ids.add(data['original_video_path'])
                    elif 'id' in data:
                        existing_ids.add(data['id'])
                except:
                    pass
    print(f"Found {len(existing_ids)} existing samples. Creating resume filter...")
    return existing_ids


if __name__ == "__main__":

    import multiprocessing as mp
    # Linux + CUDA 推荐
    mp.set_start_method("fork", force=True)

    llm, sampling_params = init_llm()
    DATA_PATH = "path/to/data.jsonl"
    SAVE_DIR = "path/to/output"
    os.makedirs(SAVE_DIR, exist_ok=True)

    save_path = osp.join(SAVE_DIR, "qwen235b_avqa2.jsonl") 

    MAX_SAMPLES = 200000
    BATCH_SIZE = 8
    WRITE_BUFFER_SIZE = 50 

    existing_ids = get_existing_ids(save_path) 

    prompt_batch = []
    data_batch = []
    write_buffer = []
    total_processed_count = 0

    
    print(f"Start processing. Output: {save_path}")

    with open(save_path, 'a', encoding='utf-8') as f_w: # 保持文件打开状态，避免频繁 open/close
        
        for data in tqdm(iter_jsonl(DATA_PATH)):
            if total_processed_count >= MAX_SAMPLES:
                break
            
            # --- [Check] 唯一性检查/跳过已处理 ---
            vid_id = data.get('original_video_path') or data.get('id')
            if vid_id and vid_id in existing_ids:
                continue

            try:
                # 提取 caption
                caption = next(
                    x["content"] for x in data["messages"]
                    if x["role"] == "assistant"
                )
            except StopIteration:
                continue

            # 构建 Prompt
            idx = choose_prompt_index()
            sys_prompt = PROMPT_LIST[idx]
            prompt = build_prompt(sys_prompt, caption)

            prompt_batch.append(prompt)
            data_batch.append(data)

            if len(prompt_batch) >= BATCH_SIZE:
                
                texts = generate_batch(llm,prompt_batch)

                for text, raw_data in zip(texts, data_batch):
                    ret = extract_items(text)
                    if not ret:
                        continue 

                    out = raw_data.copy()
                    out.update({
                        "question_type": ret["question_type"],
                        "items": ret["items"],
                    })
                    write_buffer.append(json.dumps(out, ensure_ascii=False))
                    total_processed_count += 1


                if len(write_buffer) >= WRITE_BUFFER_SIZE:
                    f_w.write("\n".join(write_buffer) + "\n")
                    f_w.flush() 
                    write_buffer = [] 


                prompt_batch = []
                data_batch = []


        if len(prompt_batch) > 0:
            print(f"Processing remaining {len(prompt_batch)} items...")
            texts = generate_batch(llm, prompt_batch)
            for text, raw_data in zip(texts, data_batch):
                ret = extract_items(text)
                if ret:
                    out = raw_data.copy()
                    out.update({
                        "question_type": ret["question_type"],
                        "items": ret["items"],
                    })
                    write_buffer.append(json.dumps(out, ensure_ascii=False))
                    total_processed_count += 1


        if write_buffer:
            f_w.write("\n".join(write_buffer) + "\n")
            f_w.flush()

    print(f"Done! Total processed: {total_processed_count}")