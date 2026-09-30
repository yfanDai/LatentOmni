from PIL import Image
from pathlib import Path
import os
from src.utils import *
from qwen_vl_utils import process_vision_info

import os  # 保留os库（若后续无其他用途可删除，此处为兼容原代码结构）

def Monet_single_input_images_preprocess_function(sample, dataset_root="", allow_no_observation=False):
    """
    Preprocess function for Monet with single input images, interleaved CoT.
    修改点：
    1. 移除"图像数量=视觉token数量"的校验规则
    2. 视频路径直接使用original_video_path的值，不再拼接dataset_root
    """
    n_img_pad = 0
    conversations = sample["data"]
    seen_observation = False
    # Process image loading for all steps first
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        # Track whether an assistant image has appeared before any observation text in this step
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "text":
                
                if step["role"] == "assistant":
                    n_img_pad += content['text'].count('<Unified_Latent></Unified_Latent>')
                    # Validate that any observation text must be preceded by an assistant image within the same step
                    if "<Observation>" in content.get("text", "") and not seen_assistant_image:
                        content['text'] = content['text'].replace("<observation>", "").replace("</observation>", "")
                    if "<Observation>" in content.get("text", ""):
                        seen_observation = True

                elif step["role"] == "user":
                    img_key = "video"
                    if 'Zebra_CoT_visual_search' not in new_step["content"][0][img_key] and 'Zebra_CoT_count' not in new_step["content"][0][img_key]: # keep boxed instructions for Zebra_CoT_visual_search
                        content["text"] = content["text"].replace("\nPut your final answer within \\boxed{}.", "")

            new_step["content"][j] = content
        conversations[i] = new_step
    sample["data"] = conversations
    
    if not seen_observation and not allow_no_observation:
        #print("[Preprocess] No observation found in assistant responses. Discard this sample")
        return None

    return sample

def Monet_single_input_images_preprocess_function_question_only(sample, dataset_root="", cur_max=-1, id=0, rank=-1):
    """
    Preprocess function for Monet with single input images, question only.
    """
    conversations = []

    # Process image loading for all steps first
    for i, step in enumerate(sample[:2]):
        new_step = step.copy()
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "video":
                content["video"] = os.path.join(dataset_root,content.pop("video")) 
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<Unified_Latent></Unified_Latent>" not in new_step["content"][j-1]["text"]:
                        return None, cur_max
                if step["role"] == "assistant":
                    seen_assistant_image = True
            elif content["type"] == "text" and step["role"] == "assistant":
                if "<observation>" in content.get("text", "") and not seen_assistant_image:
                    return None, cur_max
            
            new_step["content"][j] = content
        conversations.append(new_step)

    return conversations, cur_max


task_preporcess_config = {
    'mm-reasoning': Monet_single_input_images_preprocess_function
}

