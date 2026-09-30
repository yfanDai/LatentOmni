import json
from vllm import LLM, SamplingParams

# --- 配置区 ---
JUDGE_MODEL_PATH = "path/to/model"
INPUT_PATH = "path/to/data.jsonl"
FINAL_PATH = "path/to/output.jsonl"

def run_judge():
    llm = LLM(
        model=JUDGE_MODEL_PATH,
        tensor_parallel_size=8, # 32B 模型在 8 卡上飞快
        gpu_memory_utilization=0.9
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    with open(INPUT_PATH, 'r') as f:
        data = [json.loads(line) for line in f]

    judge_prompts = []
    # 每个样本产生两条判断请求
    for item in data:
        for mode in ['video_only', 'audio_only']:
            pred = item.get(f'pred_{mode}', "")
            question_type = item['question_type']
            question_content = item['question']
                # breakpoint()
            prompt = (
                f"<|im_start|>system\nYou are a strict judge. Answer only 'Yes' or 'No', you should not generate any other worlds (such as <think> </think>).<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {question_content}\n"
                f"Ground Truth: {item['answer']}\n"
                f"Model Prediction: {pred}\n"
                f"Is the prediction semantically correct? (Yes/No):<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            judge_prompts.append(prompt)

    print(f"Judging {len(judge_prompts)} predictions...")
    outputs = llm.generate(judge_prompts, sampling_params)
    
    # 结果分析
    # 每两行结果对应一个原始 Sample
    filtered_count = 0
    with open(FINAL_PATH, 'w') as f:
        for i in range(len(data)):
            # breakpoint()
            v_res = outputs[2*i].outputs[0].text.strip().lower()
            a_res = outputs[2*i+1].outputs[0].text.strip().lower()
            
            # 高质量 AVQA 定义：单模态都答不对
            v_is_wrong = "no" in v_res
            a_is_wrong = "no" in a_res
            
            if v_is_wrong and a_is_wrong:
                data[i]['judge_v'] = v_res
                data[i]['judge_a'] = a_res
                f.write(json.dumps(data[i], ensure_ascii=False) + '\n')
                filtered_count += 1
                
    print(f"Done! Saved {filtered_count} samples to {FINAL_PATH}")

if __name__ == "__main__":
    run_judge()