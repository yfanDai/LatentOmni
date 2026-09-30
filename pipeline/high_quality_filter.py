import json
import os
from collections import defaultdict

def filter_sft_data_by_evaluation(input_file, output_file, thresholds):
    """
    根据大模型生成的 evaluation 评分筛选高质量的 SFT 训练数据。
    """
    total_count = 0
    passed_count = 0
    failed_count = 0
    
    # 用于统计各项指标导致淘汰的原因
    failure_reasons = defaultdict(int)
    # 统计缺失 evaluation 字段的数据
    missing_eval_count = 0 

    filtered_data = []

    print("正在加载并筛选数据，请稍候...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行 JSON 解析失败，已跳过。")
                    continue
                
                total_count += 1
                
                # 假设 evaluation 字段在 JSON 的顶层
                # 如果你的结构是嵌套的（比如存放在 sample['meta']['evaluation']），请在这里修改路径
                evaluation = sample.get("evaluation", {})
                
                if not evaluation:
                    missing_eval_count += 1
                    failed_count += 1
                    continue
                
                # 检查是否所有指标都达到阈值
                passed = True
                failed_key = None
                
                for key, min_score in thresholds.items():
                    # 如果指标不存在，默认给 0 分（会被淘汰）
                    score = evaluation.get(key, 0)
                    if score < min_score:
                        passed = False
                        failed_key = key
                        break # 一项不达标即淘汰
                
                if passed:
                    passed_count += 1
                    filtered_data.append(sample)
                else:
                    failed_count += 1
                    failure_reasons[failed_key] += 1

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return

    # 将通过筛选的数据写入新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")
        return

    # 打印详细的统计报告
    print("\n" + "="*40)
    print("📊 数据筛选统计报告")
    print("="*40)
    print(f"总处理条目: {total_count}")
    print(f"✅ 通过筛选: {passed_count} ({(passed_count/total_count)*100:.1f}%)" if total_count else "0")
    print(f"❌ 被淘汰的: {failed_count} ({(failed_count/total_count)*100:.1f}%)" if total_count else "0")
    
    print("\n📉 淘汰原因分布 (首次未达标指标):")
    if missing_eval_count > 0:
        print(f"  - 缺失 evaluation 字段: {missing_eval_count} 条")
    
    # 按淘汰数量降序排序打印
    sorted_reasons = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_reasons:
        threshold_val = thresholds.get(key)
        print(f"  - {key} (需 >= {threshold_val}, 但实际偏低): {count} 条")
    print("="*40)

# ------------------- 配置参数 -------------------
INPUT_JSONL = "path/to/data.jsonl"
OUTPUT_JSONL = "path/to/output.jsonl"

# 严格的 SFT 训练数据筛选阈值配置表
# 你可以根据终端打印出的“淘汰原因分布”，随时在这里把阈值调高或调低
THRESHOLDS_CONFIG = {
    "context_utilization": 4,     # 拒绝无端联想
    "hallucination_absence": 5,   # 拒绝严重幻觉
    "logical_reasoning": 4,       # 要求逻辑严密
    "answer_accuracy": 5,         # 答案必须正确
    "question_difficulty": 3,     # [关键] 剔除太简单(1-2分)的问题
    "observation_deduction": 4    # [关键] 拒绝单纯的“复读机”
}

# ------------------- 执行 -------------------
if __name__ == "__main__":
    filter_sft_data_by_evaluation(INPUT_JSONL, OUTPUT_JSONL, THRESHOLDS_CONFIG)