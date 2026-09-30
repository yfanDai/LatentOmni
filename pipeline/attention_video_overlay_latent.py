"""
Visualize latent-reasoning attention over audio/video tokens on the original video.

This script follows the same inference path as `test.py`, but instead of plotting a
token-token heatmap, it:

1. Captures only the attention produced during the latent reasoning loop triggered by
   `<Unified_Latent>`.
2. Maps input AV tokens back to:
   - video temporal/spatial cells using `video_grid_thw` and `spatial_merge_size`
   - audio temporal bins aligned to the same video timeline
3. Produces two intuitive artifacts:
   - `top-k` standalone keyframes with attention overlays
   - an overlay video where warm colors indicate stronger attention and cool colors
     indicate weaker attention

Example:
    python pipeline/attention_video_overlay_latent.py \
        --model_path path/to/model \
        --file_path path/to/video.mp4 \
        --prompt "Describe the key event in the video." \
        --sys_prompt "You are a helpful assistant." \
        --output_dir outputs/latent_av_vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from latent_model import apply_latent_omni  # noqa: F401 - installs local latent model patch
from transformers import (
    AutoProcessor,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from src.omni_utils import process_mm_info


def _get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_inputs_to_device_dtype(
    inputs: dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    float_like_keys = {"pixel_values", "pixel_values_videos", "input_features"}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
            if key in float_like_keys:
                moved[key] = moved[key].to(model_dtype)
        else:
            moved[key] = value
    return moved


def _resolve_special_token_ids(tokenizer) -> dict[str, int]:
    return {
        "latent_start_id": tokenizer.convert_tokens_to_ids("<Unified_Latent>"),
        "latent_end_id": tokenizer.convert_tokens_to_ids("</Unified_Latent>"),
        "latent_pad_id": tokenizer.convert_tokens_to_ids("<Unified_Latent_pad>"),
        "video_start_id": tokenizer.convert_tokens_to_ids("<|vision_bos|>"),
        "video_end_id": tokenizer.convert_tokens_to_ids("<|vision_eos|>"),
        "video_token_id": tokenizer.convert_tokens_to_ids("<|VIDEO|>"),
        "audio_start_id": tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        "audio_end_id": tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        "audio_token_id": tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
    }


def _configure_model_special_ids(model, token_ids: dict[str, int]) -> None:
    model.config.latent_token_id = int(token_ids["latent_pad_id"])
    model.config.latent_start_id = int(token_ids["latent_start_id"])
    model.config.latent_end_id = int(token_ids["latent_end_id"])
    model.config.video_token_id = int(token_ids["video_token_id"])
    model.config.video_start_id = int(token_ids["video_start_id"])
    model.config.video_end_id = int(token_ids["video_end_id"])
    model.config.audio_token_id = int(token_ids["audio_token_id"])
    model.config.audio_start_id = int(token_ids["audio_start_id"])
    model.config.audio_end_id = int(token_ids["audio_end_id"])


def _to_cpu_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu().clone()


def _get_cache_seq_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[2])
    except Exception:
        return 0


@dataclass
class AttentionStepRecord:
    step_index: int
    q_len: int
    past_len: int
    attention_len: Optional[int]
    input_ids: Optional[torch.Tensor]
    attention: torch.Tensor
    stage: str


class GenerationAttentionRecorder:
    """
    Capture attention rows from one of two sources:

    - `latent`: only the latent reasoning loop after `<Unified_Latent>`
    - `text`: ordinary decode-time text-token attention rows
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        latent_start_id: int,
        latent_size: int,
        attention_source: str,
    ):
        self.model = model
        self.latent_start_id = int(latent_start_id)
        self.latent_size = int(latent_size)
        self.attention_source = attention_source

        num_layers = model.model.config.num_hidden_layers
        self.layer_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

        self._self_attn = model.model.layers[self.layer_idx].self_attn
        self._inner_model = model.model
        self._original_self_attn_forward = self._self_attn.forward
        self._original_inner_forward = self._inner_model.forward

        self._current_meta: Optional[dict[str, Any]] = None
        self._state = "idle"
        self._remaining_steps = 0
        self._active_run: Optional[dict[str, Any]] = None

        self.call_records: list[dict[str, Any]] = []
        self.step_records: list[AttentionStepRecord] = []
        self.group_summaries: list[dict[str, Any]] = []

    def install(self) -> None:
        def _patched_self_attn_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            result = self._original_self_attn_forward(*args, **kwargs)
            attn_weights = result[1]

            if (
                self._current_meta is not None
                and self._current_meta.get("stage") in {"latent_step", "text_step"}
                and attn_weights is not None
            ):
                # We only keep the target 1-token attention rows; prefill is too large
                # and not relevant for the requested visualization.
                self._current_meta["attention"] = attn_weights.detach().cpu().float()
            return result

        def _patched_inner_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            q_len = 0
            if input_ids is not None:
                q_len = int(input_ids.shape[1])
            elif inputs_embeds is not None:
                q_len = int(inputs_embeds.shape[1])

            maybe_token = None
            if input_ids is not None and input_ids.numel() == 1:
                maybe_token = int(input_ids.reshape(-1)[0].item())

            stage = "other"
            if self.attention_source == "latent":
                if maybe_token == self.latent_start_id and q_len == 1:
                    stage = "latent_start"
                elif self._state == "collect_steps" and input_ids is None and q_len == 1:
                    stage = "latent_step"
                elif self._state == "expect_final":
                    stage = "latent_final"
            elif self.attention_source == "text":
                if q_len == 1 and maybe_token != self.latent_start_id:
                    stage = "text_step"
            else:
                raise ValueError(f"Unsupported attention_source: {self.attention_source}")

            meta = {
                "stage": stage,
                "q_len": q_len,
                "past_len": _get_cache_seq_len(kwargs.get("past_key_values")),
                "attention_len": int(kwargs["attention_mask"].shape[-1])
                if isinstance(kwargs.get("attention_mask"), torch.Tensor)
                else None,
                "input_ids": _to_cpu_clone(input_ids),
            }

            if stage == "latent_step":
                meta["step_index"] = self.latent_size - self._remaining_steps
            elif stage == "text_step":
                meta["step_index"] = len(self.step_records)
            self._current_meta = meta
            outputs = self._original_inner_forward(*args, **kwargs)
            self._current_meta = None

            self.call_records.append(meta)

            if stage == "latent_start":
                self._active_run = {
                    "start": meta,
                    "steps": [],
                    "final": None,
                }
                self._state = "collect_steps"
                self._remaining_steps = self.latent_size
            elif stage == "latent_step":
                attention = meta.get("attention")
                if attention is None:
                    raise RuntimeError(
                        "Latent reasoning step was reached, but no attention weights were captured. "
                        "Make sure the model runs with eager attention."
                    )
                step_record = AttentionStepRecord(
                    step_index=int(meta["step_index"]),
                    q_len=int(meta["q_len"]),
                    past_len=int(meta["past_len"]),
                    attention_len=meta["attention_len"],
                    input_ids=meta["input_ids"],
                    attention=attention,
                    stage=stage,
                )
                self._active_run["steps"].append(step_record)
                self.step_records.append(step_record)
                self._remaining_steps -= 1
                if self._remaining_steps == 0:
                    self._state = "expect_final"
            elif stage == "text_step":
                attention = meta.get("attention")
                if attention is None:
                    raise RuntimeError(
                        "Text decode step was reached, but no attention weights were captured. "
                        "Make sure the model runs with eager attention."
                    )
                self.step_records.append(
                    AttentionStepRecord(
                        step_index=int(meta["step_index"]),
                        q_len=int(meta["q_len"]),
                        past_len=int(meta["past_len"]),
                        attention_len=meta["attention_len"],
                        input_ids=meta["input_ids"],
                        attention=attention,
                        stage=stage,
                    )
                )
            elif stage == "latent_final":
                if self._active_run is not None:
                    self._active_run["final"] = meta
                    self.group_summaries.append(
                        {
                            "group_index": len(self.group_summaries),
                            "stage": "latent",
                            "num_steps": len(self._active_run["steps"]),
                        }
                    )
                self._active_run = None
                self._state = "idle"
                self._remaining_steps = 0

            return outputs

        self._self_attn.forward = _patched_self_attn_forward
        self._inner_model.forward = _patched_inner_forward

    def remove(self) -> None:
        self._self_attn.forward = self._original_self_attn_forward
        self._inner_model.forward = self._original_inner_forward


def _extract_media_blocks(
    input_ids: torch.Tensor,
    token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    seq = input_ids.tolist()
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for idx, token in enumerate(seq):
        if token == token_ids["video_start_id"]:
            current = {
                "start_pos": idx,
                "video_positions": [],
                "audio_positions": [],
            }
            continue

        if current is None:
            continue

        if token == token_ids["video_token_id"]:
            current["video_positions"].append(idx)
        elif token == token_ids["audio_token_id"]:
            current["audio_positions"].append(idx)
        elif token == token_ids["video_end_id"]:
            current["end_pos"] = idx
            blocks.append(current)
            current = None

    return blocks


def _build_video_token_grid(
    video_positions: list[int],
    video_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[dict[int, tuple[int, int, int]], tuple[int, int, int]]:
    grid_t = int(video_grid_thw[0].item())
    grid_h = int(video_grid_thw[1].item()) // spatial_merge_size
    grid_w = int(video_grid_thw[2].item()) // spatial_merge_size
    expected = grid_t * grid_h * grid_w

    if expected != len(video_positions):
        raise ValueError(
            f"Video token count mismatch: expected {expected} from grid_thw={video_grid_thw.tolist()} "
            f"and spatial_merge_size={spatial_merge_size}, got {len(video_positions)}."
        )

    mapping: dict[int, tuple[int, int, int]] = {}
    cursor = 0
    for t in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                mapping[video_positions[cursor]] = (t, h, w)
                cursor += 1
    return mapping, (grid_t, grid_h, grid_w)


def _build_audio_token_bins(audio_positions: list[int], grid_t: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not audio_positions or grid_t <= 0:
        return mapping

    for idx, pos in enumerate(audio_positions):
        mapped_t = int(round((idx + 0.5) * grid_t / max(len(audio_positions), 1) - 0.5))
        mapped_t = max(0, min(grid_t - 1, mapped_t))
        mapping[pos] = mapped_t
    return mapping


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _contrast_rescale(
    x: np.ndarray,
    low_percentile: float = 15.0,
    high_percentile: float = 95.0,
    gamma: float = 0.8,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x

    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(valid, low_percentile))
    hi = float(np.percentile(valid, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return _safe_minmax(x)

    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    y = np.power(y, gamma)
    return y.astype(np.float32)


def _aggregate_latent_attention(
    step_records: list[AttentionStepRecord],
    group_summaries: list[dict[str, Any]],
    media_block: dict[str, Any],
    video_pos_to_grid: dict[int, tuple[int, int, int]],
    audio_pos_to_t: dict[int, int],
    grid_shape: tuple[int, int, int],
    video_score_weight: float,
    audio_score_weight: float,
) -> dict[str, Any]:
    grid_t, grid_h, grid_w = grid_shape
    video_tensor = np.zeros((grid_t, grid_h, grid_w), dtype=np.float32)
    audio_scores = np.zeros((grid_t,), dtype=np.float32)

    av_positions = media_block["video_positions"] + media_block["audio_positions"]
    if not av_positions:
        raise ValueError("The selected media block does not contain any AV tokens.")

    step_count = 0
    total_av_mass = 0.0
    for step in step_records:
        attn = step.attention
        # Shape: [batch, heads, 1, key_len] for single-token decode steps.
        if attn.ndim != 4 or attn.shape[2] != 1:
            raise ValueError(f"Unexpected attention shape: {tuple(attn.shape)}")

        row = attn[0].mean(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
        total_av_mass += float(row[av_positions].sum())
        step_count += 1

        for pos, (t, h, w) in video_pos_to_grid.items():
            if pos < row.shape[0]:
                video_tensor[t, h, w] += float(row[pos])

        for pos, t in audio_pos_to_t.items():
            if pos < row.shape[0]:
                audio_scores[t] += float(row[pos])

    if step_count == 0:
        raise ValueError("No target attention steps were captured.")

    video_tensor /= step_count
    audio_scores /= step_count

    video_temporal_scores = video_tensor.mean(axis=(1, 2)) if video_tensor.size > 0 else np.zeros((grid_t,), dtype=np.float32)
    video_temporal_norm = _contrast_rescale(video_temporal_scores, low_percentile=20.0, high_percentile=95.0, gamma=0.75)
    audio_temporal_norm = _contrast_rescale(audio_scores, low_percentile=20.0, high_percentile=95.0, gamma=0.75)
    combined_scores = (
        float(video_score_weight) * video_temporal_norm
        + float(audio_score_weight) * audio_temporal_norm
    )
    combined_scores = _contrast_rescale(combined_scores, low_percentile=10.0, high_percentile=95.0, gamma=0.75)
    video_tensor_norm = _contrast_rescale(video_tensor, low_percentile=35.0, high_percentile=99.0, gamma=0.65)

    return {
        "video_tensor": video_tensor,
        "video_tensor_norm": video_tensor_norm,
        "audio_scores": audio_scores,
        "video_temporal_scores": video_temporal_scores,
        "combined_scores": combined_scores,
        "run_summaries": group_summaries,
        "total_av_attention_mass": total_av_mass,
        "step_count": step_count,
    }


def _select_representative_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0:
        return []

    top_k = max(1, min(int(top_k), int(scores.shape[0])))
    min_gap = max(1, int(round(scores.shape[0] / max(2 * top_k, 1))))
    ranked = np.argsort(scores)[::-1].tolist()

    selected: list[int] = []
    for idx in ranked:
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(int(idx))
        if len(selected) == top_k:
            break

    if len(selected) < top_k:
        for idx in ranked:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == top_k:
                break

    return sorted(selected, key=lambda i: float(scores[i]), reverse=True)


def _read_video_meta(video_path: str) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid video metadata for {video_path}: total_frames={total_frames}, fps={fps}"
        )
    return total_frames, fps, width, height


def _build_temporal_frame_centers(
    total_frames: int,
    sampled_frame_count: int,
    grid_t: int,
    temporal_patch_size: int,
) -> np.ndarray:
    if grid_t <= 0:
        return np.zeros((0,), dtype=np.int32)

    if sampled_frame_count > 0 and sampled_frame_count == grid_t * temporal_patch_size:
        sampled_indices = np.linspace(0, total_frames - 1, sampled_frame_count).round().astype(np.int32)
        centers = []
        for t in range(grid_t):
            start = t * temporal_patch_size
            end = start + temporal_patch_size
            centers.append(int(round(float(sampled_indices[start:end].mean()))))
        return np.asarray(centers, dtype=np.int32)

    return np.linspace(0, total_frames - 1, grid_t).round().astype(np.int32)


def _build_frame_to_temporal_index(total_frames: int, temporal_centers: np.ndarray) -> np.ndarray:
    if temporal_centers.size == 0:
        return np.zeros((total_frames,), dtype=np.int32)

    if temporal_centers.size == 1:
        return np.zeros((total_frames,), dtype=np.int32)

    boundaries = ((temporal_centers[:-1] + temporal_centers[1:]) / 2.0).astype(np.float32)
    frame_ids = np.arange(total_frames, dtype=np.float32)
    return np.searchsorted(boundaries, frame_ids, side="right").astype(np.int32)


def _extract_specific_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(int(i) for i in frame_indices))
    collected: dict[int, np.ndarray] = {}
    if not wanted:
        return collected

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    wanted_cursor = 0
    frame_id = 0
    while wanted_cursor < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        target = wanted[wanted_cursor]
        if frame_id == target:
            collected[target] = frame.copy()
            wanted_cursor += 1
        frame_id += 1

    cap.release()
    missing = [idx for idx in wanted if idx not in collected]
    if missing:
        raise ValueError(f"Could not extract frames {missing} from {video_path}")
    return collected


def _colorize_scores(x: np.ndarray) -> np.ndarray:
    """
    Map [0, 1] scores to a cool->warm palette directly in BGR space for OpenCV.
    """
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    low = np.array([255, 245, 170], dtype=np.float32)   # bright cool cyan in BGR
    mid = np.array([40, 235, 255], dtype=np.float32)    # vivid yellow in BGR
    high = np.array([0, 70, 255], dtype=np.float32)     # saturated orange-red in BGR

    x3 = x[..., None]
    left_mask = (x <= 0.55).astype(np.float32)[..., None]
    left_t = np.clip(x3 / 0.55, 0.0, 1.0)
    right_t = np.clip((x3 - 0.55) / 0.45, 0.0, 1.0)

    left_color = low * (1.0 - left_t) + mid * left_t
    right_color = mid * (1.0 - right_t) + high * right_t
    return left_mask * left_color + (1.0 - left_mask) * right_color


def _blend_solid_color(frame_bgr: np.ndarray, color_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip(frame_bgr * (1.0 - alpha) + color_bgr * alpha, 0.0, 255.0)


def _draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7) -> None:
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _render_overlay_frame(
    frame_bgr: np.ndarray,
    spatial_heatmap: np.ndarray,
    frame_score: float,
    audio_score: float,
    time_sec: float,
    rank_label: Optional[str] = None,
    show_text: bool = False,
    show_audio_bar: bool = False,
    strength_scale: float = 1.0,
) -> np.ndarray:
    frame = frame_bgr.astype(np.float32).copy()
    h, w = frame.shape[:2]
    strength_scale = float(np.clip(strength_scale, 0.0, 1.0))

    frame_score = float(np.clip(frame_score, 0.0, 1.0))
    audio_score = float(np.clip(audio_score, 0.0, 1.0))
    base_strength = max(frame_score, 0.65 * audio_score)

    base_color = _colorize_scores(np.asarray(base_strength, dtype=np.float32)).reshape(3)
    frame = _blend_solid_color(
        frame,
        np.ones_like(frame) * base_color.reshape(1, 1, 3),
        strength_scale * (0.08 + 0.28 * np.power(base_strength, 0.75)),
    )

    if spatial_heatmap.size > 0:
        heat = cv2.resize(spatial_heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        heat = _contrast_rescale(heat, low_percentile=30.0, high_percentile=99.0, gamma=0.55)
        heat_color = _colorize_scores(heat)
        alpha_map = strength_scale * (0.10 + 0.78 * np.power(heat, 0.85))
        frame = frame * (1.0 - alpha_map[..., None]) + heat_color * alpha_map[..., None]

    if show_audio_bar:
        # Audio tokens do not have spatial coordinates, so we optionally render them
        # as a global temporal cue instead of a fake local heatmap.
        bar_h = max(8, h // 60)
        bar_color = _colorize_scores(np.asarray(audio_score, dtype=np.float32)).reshape(3)
        frame[-bar_h:, :, :] = _blend_solid_color(
            frame[-bar_h:, :, :],
            np.ones_like(frame[-bar_h:, :, :]) * bar_color.reshape(1, 1, 3),
            strength_scale * (0.22 + 0.38 * audio_score),
        )

    frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if show_text:
        _draw_text(frame, f"time={time_sec:.2f}s", (18, 30), scale=0.8)
        _draw_text(frame, f"score={frame_score:.3f}  audio={audio_score:.3f}", (18, 60), scale=0.6)
        if rank_label is not None:
            _draw_text(frame, rank_label, (18, 92), scale=0.8)
    return frame


def _resize_with_padding(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _save_keyframes(
    rendered_frames: list[np.ndarray],
    output_dir: Path,
    cell_size: tuple[int, int] = (420, 260),
) -> list[Path]:
    if not rendered_frames:
        raise ValueError("No rendered frames were provided for keyframe export.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for idx, frame in enumerate(rendered_frames, start=1):
        prepared = _resize_with_padding(frame, cell_size)
        path = output_dir / f"keyframe_{idx:02d}.png"
        cv2.imwrite(str(path), prepared)
        saved_paths.append(path)
    return saved_paths


def _write_overlay_video(
    video_path: str,
    output_path: Path,
    frame_to_t: np.ndarray,
    spatial_heatmaps: np.ndarray,
    frame_scores: np.ndarray,
    audio_scores: np.ndarray,
    fps: float,
    width: int,
    height: int,
    strength_scale: float,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Failed to create video writer for: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = int(frame_to_t[min(frame_idx, len(frame_to_t) - 1)])
        rendered = _render_overlay_frame(
            frame_bgr=frame,
            spatial_heatmap=spatial_heatmaps[t],
            frame_score=float(frame_scores[t]),
            audio_score=float(audio_scores[t]),
            time_sec=float(frame_idx / fps),
            rank_label=None,
            show_text=False,
            show_audio_bar=False,
            strength_scale=strength_scale,
        )
        writer.write(rendered)
        frame_idx += 1

    writer.release()
    cap.release()


def _save_debug_arrays(output_dir: Path, agg: dict[str, Any], temporal_centers: np.ndarray) -> None:
    np.savez_compressed(
        output_dir / "latent_attention_maps.npz",
        video_tensor=agg["video_tensor"],
        video_tensor_norm=agg["video_tensor_norm"],
        audio_scores=agg["audio_scores"],
        video_temporal_scores=agg["video_temporal_scores"],
        combined_scores=agg["combined_scores"],
        temporal_centers=temporal_centers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay latent reasoning attention on the original video."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--sys_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/latent_av_vis")
    parser.add_argument("--layer", type=int, default=-1, help="Decoder layer index (-1 means last layer).")
    parser.add_argument("--latent_size", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--media_index", type=int, default=0, help="Which input video block to visualize.")
    parser.add_argument("--attention_source", type=str, choices=["latent", "text"], default="latent")
    parser.add_argument("--video_score_weight", type=float, default=0.8)
    parser.add_argument("--audio_score_weight", type=float, default=0.2)
    parser.add_argument("--overlay_strength", type=float, default=None)
    parser.add_argument("--skip_overlay_video", action="store_true")
    parser.add_argument("--use_audio_in_video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    processor.tokenizer.add_tokens("<Unified_Latent_pad>", special_tokens=True)
    processor.tokenizer.add_tokens("<Unified_Latent>", special_tokens=True)
    processor.tokenizer.add_tokens("</Unified_Latent>", special_tokens=True)

    config = Qwen2_5OmniThinkerConfig.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
    )
    model.eval()

    token_ids = _resolve_special_token_ids(processor.tokenizer)
    _configure_model_special_ids(model, token_ids)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": args.sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.file_path},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio_in_video)
    if videos is None or len(videos) == 0:
        raise ValueError("No video was loaded from the provided conversation.")

    raw_inputs = processor(
        text=text,
        videos=videos,
        images=images,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    device = _get_model_device(model)
    inputs = _move_inputs_to_device_dtype(dict(raw_inputs), device, next(model.parameters()).dtype)
    inputs["latent_mode"] = args.attention_source == "latent"
    inputs["latent_size"] = args.latent_size
    overlay_strength = args.overlay_strength
    if overlay_strength is None:
        overlay_strength = 1.0 if args.attention_source == "latent" else 0.42

    recorder = GenerationAttentionRecorder(
        model=model,
        layer_idx=args.layer,
        latent_start_id=token_ids["latent_start_id"],
        latent_size=args.latent_size,
        attention_source=args.attention_source,
    )
    recorder.install()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                temperature=args.temperature,
                use_audio_in_video=args.use_audio_in_video,
                max_new_tokens=args.max_new_tokens,
            )
    finally:
        recorder.remove()

    generated_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )[0]
    print(f"Generated text: {generated_text}")
    print(
        f"Captured {len(recorder.step_records)} `{args.attention_source}` attention step(s) "
        f"on layer {recorder.layer_idx}."
    )

    if not recorder.step_records:
        raise RuntimeError(
            f"No `{args.attention_source}` attention steps were captured."
        )

    input_ids = inputs["input_ids"][0].detach().cpu()
    media_blocks = _extract_media_blocks(input_ids, token_ids)
    if args.media_index < 0 or args.media_index >= len(media_blocks):
        raise IndexError(
            f"media_index={args.media_index} is out of range. Found {len(media_blocks)} video block(s)."
        )
    media_block = media_blocks[args.media_index]

    video_grid_thw = inputs["video_grid_thw"].detach().cpu()
    if args.media_index >= video_grid_thw.shape[0]:
        raise IndexError(
            f"media_index={args.media_index} exceeds video_grid_thw entries ({video_grid_thw.shape[0]})."
        )
    grid_item = video_grid_thw[args.media_index]

    video_pos_to_grid, grid_shape = _build_video_token_grid(
        video_positions=media_block["video_positions"],
        video_grid_thw=grid_item,
        spatial_merge_size=int(model.spatial_merge_size),
    )
    audio_pos_to_t = _build_audio_token_bins(media_block["audio_positions"], grid_t=grid_shape[0])

    agg = _aggregate_latent_attention(
        step_records=recorder.step_records,
        group_summaries=recorder.group_summaries,
        media_block=media_block,
        video_pos_to_grid=video_pos_to_grid,
        audio_pos_to_t=audio_pos_to_t,
        grid_shape=grid_shape,
        video_score_weight=args.video_score_weight,
        audio_score_weight=args.audio_score_weight,
    )

    total_frames, fps, width, height = _read_video_meta(args.file_path)
    sampled_frame_count = int(videos[args.media_index].shape[0]) if torch.is_tensor(videos[args.media_index]) else 0
    temporal_patch_size = int(model.visual.patch_embed.temporal_patch_size)
    temporal_centers = _build_temporal_frame_centers(
        total_frames=total_frames,
        sampled_frame_count=sampled_frame_count,
        grid_t=grid_shape[0],
        temporal_patch_size=temporal_patch_size,
    )
    frame_to_t = _build_frame_to_temporal_index(total_frames, temporal_centers)

    selected_t = _select_representative_indices(agg["combined_scores"], args.top_k_frames)
    selected_frame_indices = [int(temporal_centers[t]) for t in selected_t]
    extracted_frames = _extract_specific_frames(args.file_path, selected_frame_indices)

    rendered_keyframes = []
    for rank, (t, frame_idx) in enumerate(zip(selected_t, selected_frame_indices), start=1):
        frame = extracted_frames[frame_idx]
        rendered = _render_overlay_frame(
            frame_bgr=frame,
            spatial_heatmap=agg["video_tensor_norm"][t],
            frame_score=float(agg["combined_scores"][t]),
            audio_score=float(_safe_minmax(agg["audio_scores"])[t]) if agg["audio_scores"].size > 0 else 0.0,
            time_sec=float(frame_idx / fps),
            rank_label=f"rank #{rank}",
            show_text=False,
            show_audio_bar=False,
            strength_scale=overlay_strength,
        )
        rendered_keyframes.append(rendered)

    keyframe_dir = output_dir / "keyframes"
    keyframe_paths = _save_keyframes(rendered_keyframes, keyframe_dir)

    overlay_video_path = output_dir / "attention_overlay.mp4"
    if not args.skip_overlay_video:
        _write_overlay_video(
            video_path=args.file_path,
            output_path=overlay_video_path,
            frame_to_t=frame_to_t,
            spatial_heatmaps=agg["video_tensor_norm"],
            frame_scores=agg["combined_scores"],
            audio_scores=_safe_minmax(agg["audio_scores"]) if agg["audio_scores"].size > 0 else np.zeros_like(agg["combined_scores"]),
            fps=fps,
            width=width,
            height=height,
            strength_scale=overlay_strength,
        )

    metadata = {
        "attention_source": args.attention_source,
        "generated_text": generated_text,
        "layer": recorder.layer_idx,
        "latent_size": int(args.latent_size),
        "num_captured_steps": len(recorder.step_records),
        "grid_shape": {
            "t": int(grid_shape[0]),
            "h": int(grid_shape[1]),
            "w": int(grid_shape[2]),
        },
        "video_grid_thw": [int(x) for x in grid_item.tolist()],
        "spatial_merge_size": int(model.spatial_merge_size),
        "temporal_patch_size": int(temporal_patch_size),
        "sampled_frame_count": int(sampled_frame_count),
        "selected_temporal_indices": [int(x) for x in selected_t],
        "selected_frame_indices": [int(x) for x in selected_frame_indices],
        "selected_times_sec": [float(x / fps) for x in selected_frame_indices],
        "combined_scores": [float(x) for x in agg["combined_scores"].tolist()],
        "video_temporal_scores": [float(x) for x in agg["video_temporal_scores"].tolist()],
        "audio_scores": [float(x) for x in agg["audio_scores"].tolist()],
        "video_token_count": len(media_block["video_positions"]),
        "audio_token_count": len(media_block["audio_positions"]),
        "overlay_strength": float(overlay_strength),
        "run_summaries": agg["run_summaries"],
        "total_av_attention_mass": float(agg["total_av_attention_mass"]),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _save_debug_arrays(output_dir, agg, temporal_centers)

    print("Saved keyframes to:")
    for path in keyframe_paths:
        print(f"  - {path}")
    if not args.skip_overlay_video:
        print(f"Saved overlay video to: {overlay_video_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
