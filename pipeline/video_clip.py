import json
import re
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_JSONL = "path/to/data.jsonl"       
OUTPUT_JSONL = "path/to/output.jsonl"      
VIDEO_DIR = "path/to/video"  # 你的切片视频存放目录
MAX_WORKERS = 32                      
# ===========================================

import json
import os
import hashlib
import re

def process_line(line, line_idx):
    """处理单行数据：解析 Segment -> 提取并检查视频片段 -> 组装纯 Conversation 列表"""
    try:
        item = json.loads(line.strip())
        source_path = item.get("original_video_path", "")
        if not source_path or not os.path.isfile(source_path):
            return None
            
        path_md5 = hashlib.md5(source_path.encode('utf-8')).hexdigest()
        done_file = os.path.join(VIDEO_DIR, f".{path_md5}.done")
        
        if not os.path.exists(done_file):
            return None
            
        question = item.get("question", "")
        cot_text = item.get("cot", "")
        videos_list = item.get("original_video_path", [])

        conversation = []
        
        # System
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AVQA expert."}]
        })
        
        # User
        conversation.append({
            "role": "user",
            "content": [
                {"type": "video", "video": videos_list},
                {"type": "text", "text": question}
            ]
        })

        pattern = r'\[Segment\s+(\d+)\]'
        matches = list(re.finditer(pattern, cot_text))
        
        if not matches:
            if cot_text.strip():
                conversation.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": cot_text.strip()}]
                })
            return conversation  
        
        assistant_content = []
        
        for idx, match in enumerate(matches):
            n = int(match.group(1))
            
            # 1. 提取当前 Segment 前面的文本
            if idx == 0:
                text_before_segment = cot_text[:match.start()].strip()
            else:
                # 提取上一个 Segment 结束到当前 Segment 开始之间的文本
                text_before_segment = cot_text[matches[idx-1].end():match.start()].strip()
                
            # 2. 将文本与 <Unified_Latent> 标签拼在一起
            combined_text = f"{text_before_segment}<Unified_Latent></Unified_Latent>"
            
            assistant_content.append({
                "type": "text", 
                "text": combined_text
            })

            # 3. 添加视频片段
            file_idx = n - 1
            segment_video_path = os.path.join(VIDEO_DIR, f"{path_md5}_{file_idx}.mp4")
            
            if not os.path.exists(segment_video_path):
                return None
                
            assistant_content.append({
                "type": "video",
                "video": segment_video_path
            })
            
        # 4. 处理最后一个 Segment 之后的收尾文本
        last_match = matches[-1]
        final_text = cot_text[last_match.end():].strip()
        if final_text:
            assistant_content.append({"type": "text", "text": final_text})
        
        if assistant_content:
            conversation.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        return conversation
        
    except Exception as e:
        print(f"Error processing line {line_idx}: {e}")
        return None

def main():
    print("⏳ 正在读取 JSONL 文件到内存...")
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    final_data = []
    
    print(f"🚀 开始处理，共 {len(lines)} 条数据...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_line, line, idx) for idx, line in enumerate(lines)]
        
        # 实时显示进度
        for future in tqdm(as_completed(futures), total=len(lines), desc="组装数据"):
            result = future.result()
            # 只有严格验证通过的（不为None）才会保留
            if result:
                final_data.append(result)

    print(f"\n📊 过滤报告：原始数据 {len(lines)} 条，成功保留 {len(final_data)} 条。")
    print(f"💾 正在保存结果到 {OUTPUT_JSONL}...")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("✅ 处理完成！")

if __name__ == "__main__":
    main()