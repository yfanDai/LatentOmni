import json
import re

def remove_observation_tags(text):
    """
    使用正则表达式移除 <observation> 标签及其内部所有内容
    """
    pattern = re.compile(r'<observation>.*?</observation>', re.DOTALL)
    return pattern.sub('', text)

def clean_json_file(input_path, output_path):
    total_count = 0
    cleaned_count = 0
    all_data = []
    
    try:
        # 1. 读取文件 (自动判断是 JSON 还是 JSONL)
        with open(input_path, 'r', encoding='utf-8') as f_in:
            try:
                # 尝试作为标准 JSON 数组读取
                raw_data = json.load(f_in)
                if not isinstance(raw_data, list):
                    raw_data = [raw_data]
            except json.JSONDecodeError:
                # 如果失败，回退作为 JSONL 逐行读取
                f_in.seek(0)
                raw_data = []
                for line in f_in:
                    stripped_line = line.strip()
                    if stripped_line:
                        raw_data.append(json.loads(stripped_line))

        # 2. 遍历处理数据
        for data in raw_data:
            total_count += 1
            has_modification = False
            
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if "content" in item and isinstance(item["content"], list):
                        for content_item in item["content"]:
                            if content_item.get("type") == "text" and "text" in content_item:
                                original_text = content_item["text"]
                                cleaned_text = remove_observation_tags(original_text)
                                
                                if cleaned_text != original_text:
                                    content_item["text"] = cleaned_text
                                    has_modification = True
            
            if has_modification:
                cleaned_count += 1
                
            all_data.append(data)

        # 3. 写入标准 JSON 格式
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_data, f_out, ensure_ascii=False, indent=2)

        print(f"处理完成！")
        print(f"总处理条目数: {total_count}")
        print(f"包含 <observation> 并已清理的条目数: {cleaned_count}")
        print(f"清理后的文件已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")

# ------------------- 配置路径 -------------------
INPUT_FILE = "path/to/data.json"
OUTPUT_FILE = "path/to/output.json"

# ------------------- 执行清理 -------------------
if __name__ == "__main__":
    clean_json_file(INPUT_FILE, OUTPUT_FILE)