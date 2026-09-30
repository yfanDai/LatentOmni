"""
Attention heatmap visualization for the Latent-Omni model.

Follows the same inference logic as test.py but hooks into a specific
decoder layer to capture attention weights during generation, then
plots them using the visualization code from attention_heatmap_compressed.py.

Usage:
    python pipeline/attention_heatmap_latent.py \
        --model_path path/to/model \
        --modality video \
        --file_path path/to/video.mp4 \
        --prompt "Your question here" \
        --sys_prompt "You are a helpful assistant." \
        --layer -1 \
        --latent_size 40 \
        --output outputs/attention_heatmap_latent.png
"""

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that local imports work
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from monet_qwen_model import apply_latent_omni  # noqa: F401 – patches the HF module
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerConfig,
    AutoProcessor,
)
from src.omni_utils import process_mm_info


# ===========================================================================
# ─── Attention helpers (adapted from attention_heatmap_compressed.py) ───────
# ===========================================================================

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


def _build_full_attention(all_self_attns, layer_idx):
    """Reconstruct the full (prefill + decode) attention matrix."""
    if not all_self_attns:
        raise ValueError("No attentions captured. Check that the hook is installed.")

    # all_self_attns[0]  -> prefill step  : dict { layer_idx -> tensor [1, heads, q, k] }
    # all_self_attns[1:] -> decode steps  : dict { layer_idx -> tensor [1, heads, q_len, k] }

    def _get(step_attns):
        if layer_idx in step_attns:
            return step_attns[layer_idx]
        if layer_idx < 0 and step_attns:
            return step_attns[max(step_attns.keys())]
        if len(step_attns) == 1:
            return next(iter(step_attns.values()))
        raise KeyError(f"Layer {layer_idx} not found in attentions: {sorted(step_attns.keys())}")

    prefill_attn = _get(all_self_attns[0])[0].float()   # [heads, prefill_len, k_prefill]
    prefill_len = prefill_attn.shape[1]
    
    # 【修复1】动态获取总长度：最后一步的 K 长度，就是全局总长度
    total_len = _get(all_self_attns[-1])[0].shape[-1]

    full = torch.full((total_len, total_len), float("nan"), dtype=torch.float16)
    full[:prefill_len, :prefill_len] = prefill_attn.mean(dim=0).to(full.dtype)

    # 【修复2】使用游标 current_row，支持 q_len > 1 的块状写入
    current_row = prefill_len
    for step_idx, step_attns in enumerate(all_self_attns[1:]):
        step_attn = _get(step_attns)[0].float()          # [heads, q_len, k_len]
        
        q_len = step_attn.shape[1]
        k_len = step_attn.shape[-1]
        
        # 将 [q_len, k_len] 的 2D 矩阵切片直接贴入 full 矩阵对应的多行区域
        full[current_row : current_row + q_len, :k_len] = step_attn.mean(dim=0).to(full.dtype)
        
        # 游标按实际输出的 token 数量向下移动
        current_row += q_len

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
    return [(start // step, (end + step - 1) // step) for start, end in ranges]


def _get_sys_user_lengths(processor, conversation):
    tokenizer = processor.tokenizer
    sys_user_text = processor.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False
    )
    input_ids = tokenizer(sys_user_text, add_special_tokens=False).input_ids
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    # breakpoint()
    im_start_positions = [i for i, t in enumerate(input_ids) if t == im_start_id]
    if len(im_start_positions) < 2:
        raise ValueError(
            f"Expected at least 2 <|im_start|> tokens (system, user), "
            f"got {len(im_start_positions)}."
        )
    user_start = im_start_positions[1]
    sys_len = user_start + 4          # include the 4 header tokens of the user turn
    sys_user_len = len(input_ids)
    return sys_len, sys_user_len


# ===========================================================================
# ─── Plotting (verbatim from attention_heatmap_compressed.py) ───────────────
# ===========================================================================

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
    order = ["sys prompt", "video tokens", "audio tokens", "user prompt", "output tokens"]

    pixel_definitions = []
    new_ranges = {k: [] for k in colors.keys()}
    current_new_idx = 0

    all_segments = []
    for label, segs in ranges.items():
        for s, e in segs:
            all_segments.append((s, e, label))
    all_segments.sort()

    for s, e, label in all_segments:
        seg_len = e - s
        num_full_blocks = seg_len // block
        leftover = seg_len % block

        seg_new_start = current_new_idx
        for i in range(num_full_blocks):
            b_start = s + i * block
            pixel_definitions.append(slice(b_start, b_start + block))
            current_new_idx += 1
        if leftover >= min_leftover:
            b_start = s + num_full_blocks * block
            pixel_definitions.append(slice(b_start, e))
            current_new_idx += 1
        seg_new_end = current_new_idx
        if seg_new_end > seg_new_start:
            new_ranges[label].append((seg_new_start, seg_new_end))

    L2 = len(pixel_definitions)
    if L2 == 0:
        print("Warning: No pixels to plot after grouping.")
        return

    mat_down = np.full((L2, L2), np.nan, dtype=np.float32)
    orig_mat = np.asarray(matrix, dtype=np.float32)

    for i in range(L2):
        slice_i = pixel_definitions[i]
        for j in range(L2):
            if i >= j:
                slice_j = pixel_definitions[j]
                sub_block = orig_mat[slice_i, slice_j]
                if not np.all(np.isnan(sub_block)):
                    mat_down[i, j] = np.nanmean(sub_block)

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

    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size="5%", pad=0.2, sharex=ax)
    ax_left   = divider.append_axes("left",   size="5%", pad=0.2, sharey=ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention Score (Log Scale)", fontsize=10)

    ax_bottom.set_ylim(0, 1); ax_bottom.set_xticks([]); ax_bottom.set_yticks([])
    ax_left.set_xlim(0, 1);   ax_left.set_xticks([]); ax_left.set_yticks([])

    for label, segs in new_ranges.items():
        for s, e in segs:
            ax_bottom.axvspan(s, e, color=colors[label], alpha=0.9, lw=0)
            ax_left.axhspan(s, e, color=colors[label], alpha=0.9, lw=0)

    for ax_side in (ax_bottom, ax_left):
        for spine in ax_side.spines.values():
            spine.set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=colors[k], label=k)
        for k in order
        if any(len(v) > 0 for v in [new_ranges.get(k, [])])
    ]
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {output_path}")

def _plot_modality_trend(
    full_attn: torch.Tensor,
    prefill_len: int,
    video_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    output_path: Path,
    smooth_window: int = 5
):
    """
    绘制生成阶段对 Audio 和 Video 的平均注意力强度趋势折线图
    """
    # 提取生成阶段对输入阶段的注意力矩阵 (形状: [num_generated_tokens, prefill_len])
    decode_attn = full_attn[prefill_len:, :prefill_len].cpu().numpy()
    num_generated = decode_attn.shape[0]

    # 获取音视频对应的列索引
    video_idx = np.where(video_mask.numpy())[0] if video_mask is not None else []
    audio_idx = np.where(audio_mask.numpy())[0] if audio_mask is not None else []

    # 计算每个生成的 Token 对 Video 和 Audio 的平均注意力 (忽略 NaN)
    # 乘以 1000 是为了将极小的平均权重放大到肉眼舒适的量级
    scale_factor = 1000 
    video_trend = np.nanmean(decode_attn[:, video_idx], axis=1) * scale_factor if len(video_idx) > 0 else np.zeros(num_generated)
    audio_trend = np.nanmean(decode_attn[:, audio_idx], axis=1) * scale_factor if len(audio_idx) > 0 else np.zeros(num_generated)

    # 平滑函数 (Moving Average)
    def smooth(y, box_pts):
        if len(y) < box_pts:
            return y
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')

    video_trend_smooth = smooth(video_trend, smooth_window)
    audio_trend_smooth = smooth(audio_trend, smooth_window)

    # 绘图
    plt.figure(figsize=(10, 4), dpi=200)
    x = np.arange(num_generated)

    # 绘制 Video 趋势 (使用你在 heatmap 里定义的暖黄色/橙色调)
    if len(video_idx) > 0:
        plt.plot(x, video_trend_smooth, label="Video Attention", color="#E66100", linewidth=2.5)
        plt.fill_between(x, video_trend_smooth, color="#E66100", alpha=0.1)

    # 绘制 Audio 趋势 (使用偏粉红/紫色的色调)
    if len(audio_idx) > 0:
        plt.plot(x, audio_trend_smooth, label="Audio Attention", color="#9B3A70", linewidth=2.5)
        plt.fill_between(x, audio_trend_smooth, color="#9B3A70", alpha=0.1)

    plt.title("Average Attention Intensity to Audio/Video during Generation", fontsize=14, fontweight='bold')
    plt.xlabel("Generated Token Index", fontsize=12)
    plt.ylabel(f"Avg Attention Weight (x{scale_factor})", fontsize=12)
    
    # 优化坐标轴和网格
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved line chart trend to: {output_path}")

# ===========================================================================
# ─── Attention capture hook ──────────────────────────────────────────────────
# ===========================================================================

class _AttentionHook:
    """
    Captures attention weights from a single decoder layer by registering a
    forward hook on that layer's self-attention module.

    Each call to the hooked layer appends:
        { layer_idx: tensor of shape [1, heads, q_len, k_len] }
    to `self.records`.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.records = []
        self._handle = None

    def install(self, model: Qwen2_5OmniThinkerForConditionalGeneration):
        """Register the hook on the specified decoder layer's self-attention."""
        num_layers = model.model.config.num_hidden_layers
        real_idx = self.layer_idx if self.layer_idx >= 0 else num_layers + self.layer_idx
        self.layer_idx = real_idx

        target_layer = model.model.layers[real_idx]
        self._handle = target_layer.self_attn.register_forward_hook(self._hook_fn)
        print(f"[hook] Installed attention hook on decoder layer {real_idx}.")

    def _hook_fn(self, module, inputs, outputs):
        # outputs of Qwen2 self-attention: (hidden, attn_weights, past_kv)
        # attn_weights is None unless output_attentions=True inside the module.
        # We therefore patch output_attentions at the module level instead.
        # See install_with_output_attentions() for the full approach.
        pass

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _install_attention_capture(model: Qwen2_5OmniThinkerForConditionalGeneration,
                                layer_idx: int):
    """
    Monkey-patches a single decoder layer to always return attention weights,
    and installs a forward hook that stores them.

    Returns a list that will be populated in-place with captured records, and
    a cleanup callable.
    """
    num_layers = model.model.config.num_hidden_layers
    real_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    print(f"[hook] Capturing attention from decoder layer {real_idx} "
          f"(out of {num_layers}).")

    captured = []          # list of {real_idx: tensor[1, H, q, k]}

    target_self_attn = model.model.layers[real_idx].self_attn
    _original_forward = target_self_attn.forward

    def _patched_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        result = _original_forward(*args, **kwargs)
        # result = (hidden, attn_weights, past_kv)  when output_attentions=True
        attn_weights = result[1]   # [batch, heads, q, k]
        if attn_weights is not None:
            captured.append({real_idx: attn_weights.detach().cpu()})
        return result

    target_self_attn.forward = _patched_forward

    def _cleanup():
        target_self_attn.forward = _original_forward

    return captured, real_idx, _cleanup


# ===========================================================================
# ─── Main ───────────────────────────────────────────────────────────────────
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attention heatmap visualization for Latent-Omni."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--modality", type=str, default="video",
                        choices=["audio", "video", "image"])
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the media file (video/audio/image).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--sys_prompt", type=str, required=True)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Decoder layer index to visualize (-1 = last layer).")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_plot_tokens", type=int, default=4000)
    parser.add_argument("--output", type=str, default="outputs/attention_heatmap_latent.png")
    parser.add_argument("--use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--latent_size", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load processor (same approach as test.py)
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    # Register latent special tokens
    processor.tokenizer.add_tokens("<Unified_Latent_pad>", special_tokens=True)
    processor.tokenizer.add_tokens("<Unified_Latent>",     special_tokens=True)
    processor.tokenizer.add_tokens("</Unified_Latent>",    special_tokens=True)

    # ------------------------------------------------------------------
    # 2. Load model (same approach as test.py)
    # ------------------------------------------------------------------
    config = Qwen2_5OmniThinkerConfig.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )

    tokenizer = processor.tokenizer

    # Resolve special token ids
    def _tok_id(tok):
        return int(processor.tokenizer(tok, return_tensors="pt")["input_ids"][0])

    latent_start_idx  = _tok_id("<Unified_Latent>")
    latent_end_idx    = _tok_id("</Unified_Latent>")
    latent_pad_idx    = _tok_id("<Unified_Latent_pad>")
    img_start_idx     = _tok_id("<|vision_bos|>")
    img_end_idx       = _tok_id("<|vision_eos|>")
    img_pad_idx       = _tok_id("<|VIDEO|>")
    audio_start_idx   = _tok_id("<|audio_bos|>")
    audio_end_idx     = _tok_id("<|audio_eos|>")
    audio_pad_idx     = _tok_id("<|AUDIO|>")

    model.config.latent_token_id   = latent_pad_idx
    model.config.latent_start_id   = latent_start_idx
    model.config.latent_end_id     = latent_end_idx
    model.config.video_token_id    = img_pad_idx
    model.config.video_start_id    = img_start_idx
    model.config.video_end_id      = img_end_idx
    model.config.audio_token_id    = audio_pad_idx
    model.config.audio_start_id    = audio_start_idx
    model.config.audio_end_id      = audio_end_idx

    # ------------------------------------------------------------------
    # 3. Build conversation (mirrors attention_heatmap_compressed.py)
    # ------------------------------------------------------------------
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": args.sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text",         "text": args.prompt},
                {"type": args.modality,  args.modality: args.file_path},
            ],
        },
    ]

    # ------------------------------------------------------------------
    # 4. Prepare inputs (same as test.py)
    # ------------------------------------------------------------------
    text   = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=args.use_audio_in_video
    )
    inputs = processor(
        text=text,
        videos=videos,
        images=images,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    inputs["latent_mode"] = False
    inputs["latent_size"] = args.latent_size

    # ------------------------------------------------------------------
    # 5. Install attention hook
    # ------------------------------------------------------------------
    all_self_attns, hooked_layer_idx, cleanup = _install_attention_capture(
        model, args.layer
    )

    # ------------------------------------------------------------------
    # 6. Inference (same flags as test.py)
    # ------------------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=args.temperature,
            use_audio_in_video=args.use_audio_in_video,
            max_new_tokens=args.max_new_tokens,
        )

    cleanup()

    generated_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )[0]
    print(f"Generated text: {generated_text}")
    print(f"Captured {len(all_self_attns)} attention steps "
          f"(1 prefill + {len(all_self_attns) - 1} decode).")

    # ------------------------------------------------------------------
    # 7. Build full attention matrix
    # ------------------------------------------------------------------
    if not all_self_attns:
        raise RuntimeError(
            "No attention weights were captured. "
            "The hook may not have fired – check that flash_attention_2 is NOT "
            "used (it does not return attention weights)."
        )
    breakpoint()
    full_attn, prefill_len = _build_full_attention(all_self_attns, hooked_layer_idx)
    total_len = full_attn.shape[0]
    input_len = inputs["input_ids"].shape[1]

    if prefill_len != input_len:
        print(f"[warn] prefill_len ({prefill_len}) != input_ids length ({input_len}).")

    # ------------------------------------------------------------------
    # 8. Derive modality masks from input_ids
    # ------------------------------------------------------------------
    input_ids_cpu = inputs["input_ids"][0].cpu()

    audio_mask = (input_ids_cpu == audio_pad_idx)   # shape [prefill_len]
    video_mask = (input_ids_cpu == img_pad_idx)

    if not audio_mask.any():
        audio_mask = None
    if not video_mask.any():
        video_mask = None

    # ------------------------------------------------------------------
    # 9. Derive segment ranges
    # ------------------------------------------------------------------
    sys_len, sys_user_len = _get_sys_user_lengths(processor, conversation)
    sys_end  = min(sys_len, prefill_len)

    user_text_mask = torch.zeros(prefill_len, dtype=torch.bool)
    user_text_mask[sys_end:prefill_len] = True
    if audio_mask is not None:
        user_text_mask &= ~audio_mask[:prefill_len]
    if video_mask is not None:
        user_text_mask &= ~video_mask[:prefill_len]

    ranges = {
        "sys prompt":    [(0, sys_end)],
        "user prompt":   _mask_to_ranges(user_text_mask),
        "output tokens": [(prefill_len, total_len)],
        "video tokens":  _mask_to_ranges(video_mask),
        "audio tokens":  _mask_to_ranges(audio_mask),
    }

    print(f"Segment lengths – sys: {sys_end}, "
          f"video: {sum(e-s for s,e in ranges['video tokens'])}, "
          f"audio: {sum(e-s for s,e in ranges['audio tokens'])}, "
          f"user text: {sum(e-s for s,e in ranges['user prompt'])}, "
          f"output: {total_len - prefill_len}")

    # ------------------------------------------------------------------
    # 10. Downsample & plot
    # ------------------------------------------------------------------
    attn_np = full_attn.cpu().numpy()
    attn_np, step = _downsample_matrix(attn_np, args.max_plot_tokens)
    if step > 1:
        ranges = {k: _scale_ranges(v, step) for k, v in ranges.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Latent-Omni  Layer {hooked_layer_idx}  (heads averaged)"
    _plot_attention(attn_np, ranges, output_path, title)
    # ------------------------------------------------------------------
    # 11. 绘制音视频注意力趋势折线图 (新增)
    # ------------------------------------------------------------------
    line_chart_path = output_path.parent / f"{output_path.stem}_trend{output_path.suffix}"
    _plot_modality_trend(
        full_attn=full_attn, 
        prefill_len=prefill_len, 
        video_mask=audio_mask if audio_mask is None else (input_ids_cpu == img_pad_idx), # 确保传入正确的 Mask
        audio_mask=audio_mask if audio_mask is None else (input_ids_cpu == audio_pad_idx),
        output_path=line_chart_path,
        smooth_window=5 # 如果线条还是很抖，可以把这个值调大到 7 或 10
    )

if __name__ == "__main__":
    main()
