import json
from collections import Counter, defaultdict

jsonl_path = "path/to/data.jsonl"

# 统计容器 (新增了 question_difficulty 和 observation_deduction)
eval_counters = {
    "context_utilization": Counter(),
    "hallucination_absence": Counter(),
    "logical_reasoning": Counter(),
    "answer_accuracy": Counter(),
    "question_difficulty": Counter(),   # 新增：问题难度
    "observation_deduction": Counter(), # 新增：观察与演绎推理
}

category_counter = Counter()
category_name_counter = Counter()
modality_counter = Counter()

score_sum = defaultdict(int)
total = 0

with open(jsonl_path, "r", encoding="utf-8") as f: # 建议加上 encoding="utf-8" 防患于未然
    for line in f:
        # 跳过空行
        if not line.strip():
            continue
            
        data = json.loads(line)

        evaluation = data.get("evaluation", {})
        classification = data.get("classification", {})

        # 统计 evaluation 分布
        for k in eval_counters:
            score = evaluation.get(k)
            if score is not None:
                eval_counters[k][score] += 1
                score_sum[k] += score

        # 统计类别
        category_id = classification.get("category_id")
        category_name = classification.get("category_name")
        modality = classification.get("modality_dependency")

        if category_id is not None:
            category_counter[category_id] += 1
        if category_name is not None:
            category_name_counter[category_name] += 1
        if modality is not None:
            modality_counter[modality] += 1

        total += 1


print(f"\n总样本数: {total}\n")

print("===== Evaluation Score Distribution =====")
for k, counter in eval_counters.items():
    print(f"\n{k}")
    # 统计每个具体分数的数量
    for score in sorted(counter):
        print(f"  score {score}: {counter[score]}")
    # 计算平均分
    if total > 0:
        # 注意：这里除以的是总样本数(total)，如果有些样本缺少某个指标，平均分可能会偏低。
        # 如果你想计算“有分数的样本的平均分”，可以改成: score_sum[k] / sum(counter.values())
        valid_count = sum(counter.values())
        if valid_count > 0:
             print(f"  avg: {score_sum[k] / valid_count:.3f}")
        else:
             print("  avg: 0.000")

print("\n===== Category ID Distribution =====")
for k, v in category_counter.most_common():
    print(f"{k}: {v}")

print("\n===== Category Name Distribution =====")
for k, v in category_name_counter.most_common():
    print(f"{k}: {v}")

print("\n===== Modality Dependency Distribution =====")
for k, v in modality_counter.most_common():
    print(f"{k}: {v}")