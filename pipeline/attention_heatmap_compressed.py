import argparse
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import Qwen2_5OmniProcessor

from my_qwen_omni_utils import process_mm_info


def _mask_to_ranges(mask):
    if mask is None:
        return []
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not indices:
        return []
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = prev = idx
    ranges.append((start, prev + 1))
    return ranges


def _get_layer_attn(step_attns, layer_idx):
    if not isinstance(step_attns, dict):
        raise ValueError(f"Unexpected attention container type: {type(step_attns)}")
    if layer_idx in step_attns:
        return step_attns[layer_idx]
    if layer_idx < 0 and step_attns:
        return step_attns[max(step_attns.keys())]
    if len(step_attns) == 1:
        return next(iter(step_attns.values()))
    raise KeyError(f"Layer {layer_idx} not found in attentions: {sorted(step_attns.keys())}")


def _build_full_attention(all_self_attns, layer_idx):
    if not all_self_attns:
        raise ValueError("No attentions captured. Check attention_layer configuration.")
    prefill_attn = _get_layer_attn(all_self_attns[0], layer_idx)
    prefill_attn = prefill_attn[0].float()  # [heads, q, k]
    prefill_len = prefill_attn.shape[1]
    # total_len = prefill_len + len(all_self_attns)
    total_len = all_self_attns[-1][layer_idx].shape[-1]

    full = torch.full((total_len, total_len), float("nan"), dtype=torch.float16)
    full[:prefill_len, :prefill_len] = prefill_attn.mean(dim=0).to(full.dtype)

    for step_idx, step_attns in enumerate(all_self_attns[1:]):
        step_attn = _get_layer_attn(step_attns, layer_idx)
        step_attn = step_attn[0].float()  # [heads, 1, k]
        row = prefill_len + step_idx
        full[row, : step_attn.shape[-1]] = step_attn.mean(dim=0).squeeze(0).to(full.dtype)

    return full, prefill_len


def _downsample_matrix(matrix, max_tokens):
    total_len = matrix.shape[0]
    if max_tokens <= 0 or total_len <= max_tokens:
        return matrix, 1
    step = math.ceil(total_len / max_tokens)
    return matrix[::step, ::step], step


def _scale_ranges(ranges, step):
    if step == 1:
        return ranges
    scaled = []
    for start, end in ranges:
        scaled.append((start // step, (end + step - 1) // step))
    return scaled


def _get_sys_user_lengths(processor, conversation):
    tokenizer = processor.tokenizer
    # text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    # sys_ids = tokenizer(sys_text, add_special_tokens=True).input_ids
    # print(f"sys text: {sys_text}")
    # sys_len = len(sys_ids[0])
    # sys_len = len(sys_ids)

    sys_user_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    print(f"sys user text: {sys_user_text}")
    input_ids = tokenizer(sys_user_text, add_special_tokens=False).input_ids[0]
    print(f"input ids: {input_ids}")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_start_positions = [i for i, t in enumerate(input_ids) if t == im_start_id]
    if len(im_start_positions) < 2:
        raise ValueError(
            f"Expect at least 3 <|im_start|> (system, user, assistant), "
            f"but got {len(im_start_positions)}. conversation={conversation}"
        )
    system_start = im_start_positions[0]
    user_start = im_start_positions[1]
    sys_len = user_start + 4
    sys_user_len = len(input_ids)
    # sys_user_len = len(sys_user_ids)ty

    return sys_len, sys_user_len


from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

def _short_label(label: str) -> str:
    """把长 key 映射成短标签，只给文本类段写。"""
    mapping = {
        "sys prompt": "sys",
        "user prompt": "ins",
        "output tokens": "out",
    }
    return mapping.get(label, label)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.patches as mpatches # 不再需要图例 patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

def _plot_attention(
    matrix: np.ndarray,
    ranges: dict,
    output_path: Path,
    title: str,
    block: int = 25,
    min_leftover: int = 10,
):
    colors = {
        "sys prompt":   "#F4D9B8",
        "user prompt":  "#D8F1FF",
        "output tokens":"#D7F5D1",
        "video tokens": "#FFE06B",
        "audio tokens": "#FFB6B6",
    }
    # 保持物理序列顺序，这对构建矩阵至关重要
    # 注意：这里需要根据序列在 input_ids 里的实际出现顺序排列，通常如下：
    order = ["sys prompt", "video tokens", "audio tokens", "user prompt", "output tokens"]

    # 1. 计算每个分段如何映射到新像素
    # mapping 存储：(original_start, original_end) 对应的 new_pixel_index
    pixel_definitions = [] # 存储每个新像素对应的原始索引切片
    new_ranges = {k: [] for k in colors.keys()}
    
    current_new_idx = 0
    
    # 遍历所有逻辑分段，按照它们在原始序列中出现的先后顺序
    # 需要先合并并排序所有 ranges 里的段落
    all_segments = []
    for label, segs in ranges.items():
        for s, e in segs:
            all_segments.append((s, e, label))
    all_segments.sort() # 按起始索引排序

    for s, e, label in all_segments:
        seg_len = e - s
        num_full_blocks = seg_len // block
        leftover = seg_len % block
        
        seg_new_start = current_new_idx
        
        # 处理完整的 block (25)
        for i in range(num_full_blocks):
            b_start = s + i * block
            pixel_definitions.append(slice(b_start, b_start + block))
            current_new_idx += 1
            
        # 处理冗余 (10 <= leftover < 25)
        if leftover >= min_leftover:
            b_start = s + num_full_blocks * block
            pixel_definitions.append(slice(b_start, e))
            current_new_idx += 1
        # 如果 leftover < 10，代码逻辑会直接跳过，即“丢掉”
        
        seg_new_end = current_new_idx
        if seg_new_end > seg_new_start:
            new_ranges[label].append((seg_new_start, seg_new_end))

    L2 = len(pixel_definitions)
    if L2 == 0:
        print("Warning: No pixels to plot after grouping.")
        return

    # 2. 构建聚合后的矩阵
    # 我们不能简单 reshape，必须按定义好的 slice 平均化
    mat_down = np.full((L2, L2), np.nan, dtype=np.float32)
    orig_mat = np.asarray(matrix, dtype=np.float32)
    
    for i in range(L2):
        slice_i = pixel_definitions[i]
        for j in range(L2):
            # 只有下三角需要计算（或者根据你 matrix 的实际情况决定）
            if i >= j:
                slice_j = pixel_definitions[j]
                # 提取子矩阵并求均值，忽略 NaN
                sub_block = orig_mat[slice_i, slice_j]
                if not np.all(np.isnan(sub_block)):
                    mat_down[i, j] = np.nanmean(sub_block)

    # 3. 绘图逻辑
    iu = np.triu_indices(L2, k=1)
    valid = mat_down[~np.isnan(mat_down)]
    vmin = max(valid.min() if valid.size > 0 else 1e-4, 5e-4)
    vmax = valid.max() if valid.size > 0 else 1.0
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")
    cmap.set_over(color="white")
    mat_down[iu] = vmax + 1

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    for spine in ax.spines.values():
        spine.set_visible(False)
    im = ax.imshow(
        mat_down,
        origin="upper",
        interpolation="nearest",
        norm=norm,
        cmap=cmap,
        aspect="equal",
    )
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, L2 - 0.5)
    ax.set_ylim(L2 - 0.5, -0.5)
    ax.set_title(title, fontsize=14)

    # # 绘制分段边界线
    # boundary_points = set()
    # for segs in new_ranges.values():
    #     for (s, e) in segs:
    #         boundary_points.add(s)
    #         boundary_points.add(e)
    # boundary_points = sorted(p for p in boundary_points if 0 < p < L2)
    # for p in boundary_points:
    #     ax.axvline(p - 0.5, linestyle='--', color='black', alpha=0.3, linewidth=0.5)
    #     ax.axhline(p - 0.5, linestyle='--', color='black', alpha=0.3, linewidth=0.5)

    # 侧边色条
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size="5%", pad=0.2, sharex=ax)
    ax_left   = divider.append_axes("left",   size="5%", pad=0.2, sharey=ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention Score (Log Scale)", fontsize=10)
    
    ax_bottom.set_ylim(0, 1); ax_bottom.set_xticks([]); ax_bottom.set_yticks([])
    ax_left.set_xlim(0, 1); ax_left.set_xticks([]); ax_left.set_yticks([])

    # 按照 new_ranges 填充颜色
    for label, segs in new_ranges.items():
        for (s, e) in segs:
            ax_bottom.axvspan(s, e, color=colors[label], alpha=0.9, lw=0)
            ax_left.axhspan(s, e, color=colors[label], alpha=0.9, lw=0)

    for ax_side in (ax_bottom, ax_left):
        for spine in ax_side.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot averaged attention heatmap with segment highlights.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--modality", type=str, default="video", choices=["audio", "video", "image"])
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--sys_prompt", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_plot_tokens", type=int, default=4000)
    parser.add_argument("--output", type=str, default="outputs/attention_heatmap.png")
    parser.add_argument("--use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--compression_transformer", action="store_true")
    parser.add_argument("--omnizip", action="store_true")
    parser.add_argument("--rho_audio", type=float, default=0.3)
    parser.add_argument("--rho_video", type=float, default=0.6)
    parser.add_argument("--g", type=int, default=3)
    parser.add_argument("--contextual_ratio", type=float, default=0.05)
    args = parser.parse_args()

    if args.compression_transformer:
        from avcompression_transformer_attention_score.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
        )

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        model.thinker.compression_config = {"rho_audio": args.rho_audio, "rho_video": args.rho_video}
    elif args.omnizip:
        from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )
        model.thinker.omnizip_config = {
            "rho_audio": args.rho_audio,
            "rho_video": args.rho_video,
            "g": args.g,
            "contextual_ratio": args.contextual_ratio,
        }
    else:
        from qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
        )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": args.sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": args.modality, args.modality: args.file_path},
            ],
        },
    ]

    if args.layer < 0:
        args.layer = model.thinker.model.config.num_hidden_layers + args.layer
    model.thinker.model.attention_layer = args.layer
    model.thinker.all_self_attns = []

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio_in_video)
    if videos:
        model.thinker.nframes = videos[0].shape[0]

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            use_audio_in_video=args.use_audio_in_video,
            do_sample=False,
            return_audio=False,
            max_new_tokens=args.max_new_tokens,
            output_attentions=False,
            use_cache=True,
        )
    generated_text = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    output_length = output_ids[0].shape[0] - inputs["input_ids"].shape[1]
    # print(output_ids[0][-1])
    print(f"Output length: {output_length} tokens")
    print(f"Generated text: {generated_text}")
    all_self_attns = model.thinker.all_self_attns
    # import ipdb; ipdb.set_trace()
    for index, idx_attns in enumerate(all_self_attns):
        for layer, attention in idx_attns.items():
            print(attention.shape)
    global_mask = getattr(model.thinker, "global_mask", torch.ones([1, inputs["input_ids"].shape[1]], dtype=torch.bool)).squeeze(0).bool().cpu()
    inputs["input_ids"] = inputs["input_ids"][..., global_mask]
    input_len = inputs["input_ids"].shape[1]
    print(f"Input ids length after masking: {input_len}")
    new_tokens = output_ids.shape[1] - input_len - (~global_mask).sum().item()
    if len(all_self_attns) != new_tokens:
        print(
            f"[warn] attention steps ({len(all_self_attns) - 1}) != generated tokens ({new_tokens})."
        )
    full_attn, prefill_len = _build_full_attention(all_self_attns, args.layer)
    total_len = full_attn.shape[0]
    if prefill_len != input_len:
        print(f"[warn] prefill_len ({prefill_len}) != input_ids length ({input_len}).")


    audio_mask = getattr(model.thinker, "audio_mask", None)
    if audio_mask is not None:
        audio_mask = audio_mask[..., 0].squeeze(0).bool().cpu()
        audio_mask = audio_mask[global_mask]
    video_mask = getattr(model.thinker, "video_mask", None)
    if video_mask is not None:
        video_mask = video_mask[..., 0].squeeze(0).bool().cpu()
        video_mask = video_mask[global_mask]
    if audio_mask is not None and audio_mask.numel() != prefill_len:
        print(f"[warn] audio_mask length ({audio_mask.numel()}) != prefill_len ({prefill_len}).")
    if video_mask is not None and video_mask.numel() != prefill_len:
        print(f"[warn] video_mask length ({video_mask.numel()}) != prefill_len ({prefill_len}).")

    sys_len, sys_user_len = _get_sys_user_lengths(processor, conversation)
    print(f"System prompt length: {sys_len} tokens")
    sys_end = min(sys_len, prefill_len)
    user_end = prefill_len

    user_text_mask = torch.zeros(prefill_len, dtype=torch.bool)
    user_text_mask[sys_end:user_end] = True

    # 把其中属于audio / video的位置去掉
    if audio_mask is not None:
        audio_m = audio_mask[:prefill_len]
        user_text_mask &= ~audio_m
    if video_mask is not None:
        video_m = video_mask[:prefill_len]
        user_text_mask &= ~video_m

    user_ranges = _mask_to_ranges(user_text_mask)

    ranges = {
        "sys prompt": [(0, sys_end)],
        "user prompt": user_ranges,
        "output tokens": [(prefill_len, total_len)],
        "video tokens": _mask_to_ranges(video_mask),
        "audio tokens": _mask_to_ranges(audio_mask),
    }


    attn_for_plot = full_attn.cpu().numpy()
    attn_for_plot, step = _downsample_matrix(attn_for_plot, args.max_plot_tokens)

    if step > 1:
        ranges = {k: _scale_ranges(v, step) for k, v in ranges.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Layer {args.layer} attention (heads averaged)"
    _plot_attention(attn_for_plot, ranges, output_path, title)

    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()